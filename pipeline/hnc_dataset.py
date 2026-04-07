"""HNC HDF5 dataset loader for the private Dementia and MDD datasets.

The data is shipped as Fernet-encrypted HDF5 + `.pkl` pairs from a Taiwanese
hospital network. The data owner provides an `hnc` Python package whose
`function_hnc.hnc_load_data(path, key)` is the official decryption helper:

    from hnc import function_hnc
    arr  = function_hnc.hnc_load_data(h5_path,  'train/data')   # → ndarray
    info = function_hnc.hnc_load_data(pkl_path, 'train')        # → DataFrame

Same helper, two key conventions: HDF5 keys end in `/data`, `.pkl` keys are
just the split name. The decrypted EEG arrays have shape
`(n_subjects, n_channels=30, n_datapoints)` at 500 Hz, dtype float64, and
amplitude in **millivolts** (we scale ×1e3 to µV inside `_preprocess`).

Schemas (confirmed via notebooks/inspect_hnc_data.ipynb 2026-04-07):

    Dementia (308 subjects, 120s recordings)
        train: (153, 30, 60000)   info: 153 rows, 15 cols
        valid: (58,  30, 60000)   info: 58  rows
        test:  (97,  30, 60000)   info: 97  rows
        Label codes: 0=Control+SCD-Control, 1=SCD-MCI, 2=Dementia
        Binary mode drops label==1 → 154 Control vs 52 Dementia (3:1)

    MDD (400 subjects, 92s recordings)
        train: (280, 30, 46000)   info: 280 rows, 12 cols
        test:  (120, 30, 46000)   info: 120 rows
        Label codes: 0=Control, 1=MDD
        Provider-balanced: 140/140 train, 60/60 test.

The provider train/valid/test splits are concatenated by default and
re-folded subject-level downstream so the methodology stays consistent
with the cross-dataset analysis on UCSD Stress / ADFTD / TDBRAIN.

Caching strategy: each split is decrypted exactly **once** in `__init__`,
sliced per subject, and saved as a `.pt` file. Subsequent runs (or further
folds) hit the cache and never touch the encrypted file again. This is
~5× faster than per-call decryption and avoids holding multiple decrypted
splits in memory simultaneously.

Usage:
    from pipeline.hnc_dataset import HNCDataset
    HNC_CHANNEL_NAMES = ['Fp1', 'Fp2', ...]   # 30 names from data owner
    ds = HNCDataset(name='dementia', data_root='data/hnc',
                    channel_names=HNC_CHANNEL_NAMES, binary=True)
    epochs, label, n_epochs, subject_id = ds[0]
"""
from __future__ import annotations

import gc
import hashlib
import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .common_channels import COMMON_19, normalize_channel_name

warnings.filterwarnings("ignore", category=RuntimeWarning)


# Per-dataset config table — adding a third HNC dataset later means adding
# one entry here, no other code changes.
DATASET_CONFIGS: Dict[str, Dict] = {
    "dementia": {
        "h5_filename": "Dementia_paper_dataset_rawData.hdf5",
        "info_filename": "Dementia_paper_dataset_info.pkl",
        "splits": ["train", "valid", "test"],
        "task_seconds": 120.0,
        "binary_pos_label": 2,        # Dementia
        "binary_neg_label": 0,        # Control + SCD-Control
        "binary_drop_labels": [1],    # SCD-MCI
        "pos_name": "Dementia",
        "neg_name": "Control",
    },
    "mdd": {
        "h5_filename": "MDD_paper_dataset_rawData.hdf5",
        "info_filename": "MDD_paper_dataset_info.pkl",
        "splits": ["train", "test"],
        "task_seconds": 92.0,
        "binary_pos_label": 1,        # MDD
        "binary_neg_label": 0,        # Control
        "binary_drop_labels": [],
        "pos_name": "MDD",
        "neg_name": "Control",
    },
}


# ----------------------------------------------------------------------
# Decryption shim
# ----------------------------------------------------------------------
def _hnc_load(path: str, key: str):
    """Call `hnc.function_hnc.hnc_load_data(path, key)` with a clear error
    if the package is missing. Imported lazily so the rest of the pipeline
    can be unit-tested without `hnc` installed."""
    try:
        from hnc import function_hnc  # type: ignore
    except ImportError as e:
        raise ImportError(
            "HNCDataset requires the `hnc` package (the data owner's "
            "decryption helper) to be importable. Install it into the "
            "active env or ask the data owner where to obtain it. "
            f"Original ImportError: {e}"
        )
    return function_hnc.hnc_load_data(path, key)


# ----------------------------------------------------------------------
# Dataset class
# ----------------------------------------------------------------------
class HNCDataset(Dataset):
    """PyTorch Dataset for HNC Dementia and MDD HDF5 datasets.

    Each item returns a tuple matching the rest of the pipeline's loaders:

        epochs:     (M, 19, T) float32 tensor — preprocessed EEG epochs
        label:      int — 0 or 1 (binary mode)
        n_epochs:   int — number of valid epochs in this recording
        subject_id: int — stable integer ID derived from the string subject ID

    Args:
        name: ``"dementia"`` or ``"mdd"``.
        data_root: directory containing the HDF5 + .pkl files.
        channel_names: ordered list of channel names matching the HDF5
            channel axis. **REQUIRED** — the loader hard-fails without it,
            since the data has 30 channels and there is no way to recover
            the standard 10-20 ordering automatically.
        target_sfreq: resampling target (Hz). Default 200 — matches LaBraM /
            CBraMod / TDBRAIN / ADFTD.
        window_sec: epoch length in seconds. Default 5.0 (LaBraM / CBraMod).
        norm: ``"zscore"`` (default) or ``"none"``.
        binary: if True (default), drop SCD-MCI for dementia and map labels
            to {0, 1}. MDD is already binary.
        cache_dir: where to write per-subject ``.pt`` caches.
        mv_to_uv_scale: multiplier applied during preprocessing. Default
            1e3 — the HNC data is in millivolts and LaBraM expects µV.
        prepare_cache: if True (default), pre-decrypt every split in
            ``__init__`` and write per-subject caches. Set False if you
            already know caches exist (e.g. inside test loops).
    """

    def __init__(
        self,
        name: str,
        data_root: str = "data/hnc",
        channel_names: Optional[List[str]] = None,
        target_sfreq: float = 200.0,
        window_sec: float = 5.0,
        stride_sec: Optional[float] = None,
        norm: str = "zscore",
        binary: bool = True,
        cache_dir: Optional[str] = None,
        orig_sfreq: float = 500.0,
        mv_to_uv_scale: float = 1e3,
        prepare_cache: bool = True,
    ):
        if name not in DATASET_CONFIGS:
            raise ValueError(
                f"Unknown HNC dataset {name!r}. Available: {list(DATASET_CONFIGS)}"
            )
        if channel_names is None:
            raise ValueError(
                f"HNC[{name}]: channel_names is required (the data has 30 "
                f"channels and there is no way to recover the standard 10-20 "
                f"order automatically). Get the channel list from the data "
                f"owner and pass channel_names=[...30 names in HDF5 axis-1 "
                f"order...]."
            )
        if len(channel_names) != 30:
            raise ValueError(
                f"HNC[{name}]: expected 30 channel names (matching HDF5 "
                f"axis 1), got {len(channel_names)}: {channel_names}"
            )

        self.name = name
        self.cfg = DATASET_CONFIGS[name]
        self.data_root = data_root
        self.channel_names_raw = list(channel_names)
        self.target_sfreq = target_sfreq
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.norm = norm
        self.binary = binary
        self.orig_sfreq = orig_sfreq
        self.mv_to_uv_scale = mv_to_uv_scale

        if cache_dir is None:
            cache_dir = f"data/cache_hnc_{name}"
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.h5_path = os.path.join(data_root, self.cfg["h5_filename"])
        self.info_path = os.path.join(data_root, self.cfg["info_filename"])
        for p in (self.h5_path, self.info_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"HNC[{name}]: missing data file {p}. Expected "
                    f"{self.cfg['h5_filename']} and {self.cfg['info_filename']} "
                    f"under data_root={data_root!r}."
                )

        # Resolve channel selection: which raw-channel index maps to each
        # COMMON_19 slot. Done before any decryption — fail fast on bad input.
        self.channel_indices = self._resolve_channel_indices(channel_names)

        # Build per-subject record list (label filter, ID hash, cache path)
        self.records: List[Dict] = []
        self._build_records()

        # Pre-decrypt and cache every subject. Each split is decrypted once.
        if prepare_cache:
            self._prepare_cache()

        # Print summary
        n_unique = len({r["subject_str_id"] for r in self.records})
        n_pos = sum(1 for r in self.records if r["label"] == 1)
        n_neg = sum(1 for r in self.records if r["label"] == 0)
        print(
            f"HNC[{name}]: {len(self.records)} recordings from {n_unique} subjects "
            f"({self.cfg['pos_name']}={n_pos}, {self.cfg['neg_name']}={n_neg}, "
            f"30→19ch, {int(self.orig_sfreq)}→{int(self.target_sfreq)} Hz, "
            f"mV→µV ×{self.mv_to_uv_scale:.0f})"
        )
        if len(self.records) > 0:
            sample_rec = self.records[0]
            print(
                f"  row 0 sanity check: subject={sample_rec['subject_str_id']!r} "
                f"split={sample_rec['split']!r} row_idx={sample_rec['row_idx']} "
                f"label={sample_rec['label']}"
            )

    # ------------------------------------------------------------------
    # Channel resolution
    # ------------------------------------------------------------------
    def _resolve_channel_indices(self, channel_names: List[str]) -> List[int]:
        """Pick the indices of the 19 standard 10-20 channels from the
        30-channel raw montage."""
        normalized = [normalize_channel_name(ch) for ch in channel_names]
        norm_to_idx: Dict[str, int] = {}
        for i, ch in enumerate(normalized):
            norm_to_idx.setdefault(ch, i)  # first occurrence wins on dupes

        indices: List[int] = []
        missing: List[str] = []
        for target in COMMON_19:
            t_norm = normalize_channel_name(target)
            if t_norm in norm_to_idx:
                indices.append(norm_to_idx[t_norm])
            else:
                missing.append(target)
        if missing:
            raise ValueError(
                f"HNC[{self.name}]: COMMON_19 channels missing from "
                f"channel_names: {missing}. Available (normalized): "
                f"{sorted(norm_to_idx)}"
            )
        return indices

    # ------------------------------------------------------------------
    # Subject ID hashing
    # ------------------------------------------------------------------
    @staticmethod
    def _hash_subject_id(s: str) -> int:
        """Stable string→int mapping for sklearn group CV. Same string
        always maps to the same int across processes/runs."""
        return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)

    # ------------------------------------------------------------------
    # Record construction (no decryption — just metadata)
    # ------------------------------------------------------------------
    def _build_records(self) -> None:
        cfg = self.cfg
        seen_str_ids: Dict[str, int] = {}

        for split in cfg["splits"]:
            df = _hnc_load(self.info_path, split)
            df = df.reset_index(drop=True)

            for row_idx in range(len(df)):
                row = df.iloc[row_idx]
                raw_label = int(row["Label"])

                if self.binary:
                    if raw_label in cfg["binary_drop_labels"]:
                        continue
                    if raw_label == cfg["binary_pos_label"]:
                        label = 1
                    elif raw_label == cfg["binary_neg_label"]:
                        label = 0
                    else:
                        continue
                else:
                    label = raw_label

                subject_str = str(row["ID"])
                seen_str_ids.setdefault(subject_str, self._hash_subject_id(subject_str))

                # Filesystem-safe cache name
                safe_id = subject_str.replace("/", "_").replace(" ", "_")
                cache_name = (
                    f"hnc_{self.name}_{split}_{row_idx:04d}_{safe_id}"
                    f"_w{self.window_sec}_sr{int(self.target_sfreq)}_n{self.norm}.pt"
                )

                self.records.append({
                    "subject_str_id": subject_str,
                    "subject_id": seen_str_ids[subject_str],
                    "patient_id": seen_str_ids[subject_str],
                    "label": label,
                    "baseline_label": label,
                    "stress_score": 0.0,  # placeholder for WindowDataset compat
                    "split": split,
                    "row_idx": row_idx,
                    "cache_name": cache_name,
                    "cache_path": os.path.join(self.cache_dir, cache_name),
                    "group": str(row.get("Group", "")),
                    "age": float(row["Age"]) if "Age" in row else float("nan"),
                    "gender": str(row.get("Gender", "")),
                })

    # ------------------------------------------------------------------
    # Eager cache preparation
    # ------------------------------------------------------------------
    def _prepare_cache(self) -> None:
        """Decrypt each split exactly once and write per-subject .pt files.

        After this, every record has its cache file on disk and __getitem__
        is a pure cache hit. Skipped automatically if every record's cache
        already exists.
        """
        # Group records by split
        by_split: Dict[str, List[Dict]] = {}
        for rec in self.records:
            by_split.setdefault(rec["split"], []).append(rec)

        for split, recs in by_split.items():
            # Skip the split entirely if every cache file already exists
            if all(os.path.exists(r["cache_path"]) for r in recs):
                continue

            print(f"HNC[{self.name}]: decrypting {split}/data ({len(recs)} subjects)...")
            arr = _hnc_load(self.h5_path, f"{split}/data")
            # arr: (n_subjects, 30, n_samples) at orig_sfreq, in mV (float64)

            for r in recs:
                if os.path.exists(r["cache_path"]):
                    continue
                subj_arr = arr[r["row_idx"]]  # (30, n_samples)
                tensor = self._preprocess_array(subj_arr)
                torch.save(tensor, r["cache_path"])

            # Free the giant decrypted blob before moving to the next split
            del arr
            gc.collect()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _preprocess_array(self, raw: np.ndarray) -> torch.Tensor:
        """Channel select → mV→µV scale → resample → epoch → normalize.

        Input:  (30, n_samples) float64 in millivolts (raw from hnc helper)
        Output: (M, 19, T) float32 tensor in z-scored microvolts
        """
        # 30 → 19 channel select, COMMON_19 order
        arr = raw[self.channel_indices, :].astype(np.float64)

        # mV → µV
        if self.mv_to_uv_scale != 1.0:
            arr = arr * self.mv_to_uv_scale

        # Resample 500 → 200 Hz (or whatever target_sfreq is)
        if self.orig_sfreq != self.target_sfreq:
            from math import gcd
            up = int(self.target_sfreq)
            down = int(self.orig_sfreq)
            g = gcd(up, down)
            up, down = up // g, down // g
            arr = resample_poly(arr, up, down, axis=1)

        # Epoch
        n_channels, total_samples = arr.shape
        samples_per_window = int(self.target_sfreq * self.window_sec)
        stride_samples = int(self.target_sfreq * (self.stride_sec or self.window_sec))
        starts = list(range(0, total_samples - samples_per_window + 1, stride_samples))
        if len(starts) == 0:
            raise ValueError(
                f"HNC[{self.name}]: recording too short "
                f"({total_samples / self.target_sfreq:.1f}s) for {self.window_sec}s windows"
            )
        epochs = np.stack([arr[:, s:s + samples_per_window] for s in starts])
        # epochs: (M, 19, T)

        if self.norm == "zscore":
            mean = epochs.mean(axis=-1, keepdims=True)
            std = epochs.std(axis=-1, keepdims=True) + 1e-6
            epochs = (epochs - mean) / std

        return torch.tensor(epochs, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        # Cache must exist after _prepare_cache(); but in case prepare_cache=False,
        # fall back to per-call decryption (slow — only for debugging).
        if not os.path.exists(rec["cache_path"]):
            arr = _hnc_load(self.h5_path, f"{rec['split']}/data")
            tensor = self._preprocess_array(arr[rec["row_idx"]])
            torch.save(tensor, rec["cache_path"])
            del arr
            gc.collect()
        else:
            tensor = torch.load(rec["cache_path"], weights_only=True)
        return tensor, rec["label"], tensor.shape[0], rec["subject_id"]

    def get_labels(self) -> np.ndarray:
        return np.array([r["label"] for r in self.records])

    def get_patient_ids(self) -> np.ndarray:
        return np.array([r["subject_id"] for r in self.records])
