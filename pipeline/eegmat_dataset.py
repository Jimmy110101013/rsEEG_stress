"""EEGMAT Dataset Loader (PhysioNet eegmat 1.0.0).

Mental arithmetic EEG dataset (Zyma et al. 2019, Frontiers in Neuroscience).
36 subjects × 2 EDF files each:
    SubjectNN_1.edf  → ~3 min baseline rest (eyes closed, before task)
    SubjectNN_2.edf  → ~1 min arithmetic task (serial subtraction, eyes closed)

Within-subject binary task: 0 = baseline rest, 1 = arithmetic.

Crucially, every subject contributes BOTH labels — unlike Stress / ADFTD /
TDBRAIN where the label is a subject-level property. This makes EEGMAT the
within-subject positive control for the variance-decomposition pipeline:
under our hypothesis, fine-tuning should cleanly rewrite the LaBraM
representation here even at small n, in contrast to the projection-only
failure mode on Stress.

Channels: 19 standard 10-20 (Fp1, Fp2, F3, F4, F7, F8, T3, T4, T5, T6,
C3, C4, P3, P4, O1, O2, Fz, Cz, Pz). Native sampling rate 500 Hz.

Usage:
    ds = EEGMATDataset("data/eegmat")
    epochs, label, n_epochs, subject_id = ds[0]
"""
import os
import warnings
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .common_channels import COMMON_19, normalize_channel_name

warnings.filterwarnings("ignore", message=".*boundary.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*annotation.*", category=RuntimeWarning)

# NOTE: This loader does NOT use `mne` because `mne.io.edf` eagerly imports
# `scipy.interpolate`, which is broken in the `timm_eeg` conda env.
# Instead we read EDFs with `pyedflib` (no scipy dep) and resample with
# `scipy.fft.rfft/irfft` (separate from the broken `scipy.interpolate`
# chain). The dataset therefore runs natively in `timm_eeg` for both
# cache build and training. It also runs cleanly in the `stress` env
# (scipy 1.17 has no ABI issue).


def _fft_resample(data: np.ndarray, src_sfreq: float, dst_sfreq: float) -> np.ndarray:
    """Bandlimited resample along the last axis using rfft/irfft.

    Equivalent to scipy.signal.resample (which we cannot use because
    scipy.signal pulls in scipy.interpolate via filter helpers, and
    scipy.interpolate is broken in timm_eeg). scipy.fft is a separate
    submodule and imports cleanly.

    Args:
        data: array of shape (..., n_samples) at src_sfreq.
        src_sfreq: original sampling frequency in Hz.
        dst_sfreq: target sampling frequency in Hz.

    Returns:
        Resampled array of shape (..., new_n_samples) where
        new_n_samples = round(n_samples * dst_sfreq / src_sfreq).
    """
    from scipy.fft import rfft, irfft  # noqa: PLC0415

    n = data.shape[-1]
    new_n = int(round(n * dst_sfreq / src_sfreq))
    if new_n == n:
        return data.astype(np.float32, copy=False)
    X = rfft(data, axis=-1)
    new_keep = new_n // 2 + 1
    if new_keep <= X.shape[-1]:
        X_new = X[..., :new_keep]
    else:
        # Upsampling: zero-pad in frequency domain
        pad_shape = list(X.shape)
        pad_shape[-1] = new_keep - X.shape[-1]
        X_new = np.concatenate(
            [X, np.zeros(pad_shape, dtype=X.dtype)], axis=-1
        )
    y = irfft(X_new, n=new_n, axis=-1)
    # rfft/irfft scale by length: irfft(rfft(x), n=n) returns x.
    # When we change n, amplitude scales by new_n / n; correct it back.
    y = y * (new_n / n)
    return y.astype(np.float32, copy=False)


class EEGMATDataset(Dataset):
    """PyTorch Dataset for EEGMAT (PhysioNet eegmat 1.0.0).

    Each item returns:
        epochs: (M, 19, T) float32 tensor — z-scored EEG epochs
        label: int — 0 (baseline rest) or 1 (arithmetic task)
        n_epochs: int — actual number of valid epochs
        subject_id: int — unique subject identifier (0..35)

    Notes:
    - Both recordings of every subject are included; subject-level CV must
      keep them in the same fold (the within-subject label structure is
      what makes EEGMAT the positive control).
    - Arithmetic recordings are ~60 s, so at 5 s windows you get ~12 epochs;
      baseline recordings are ~180 s for ~36 epochs. Class imbalance in
      *epoch counts* is OK because we pool to one feature vector per
      recording downstream.
    """

    def __init__(
        self,
        data_root: str,
        target_sfreq: float = 200.0,
        window_sec: float = 5.0,
        stride_sec: float | None = None,
        norm: str = "zscore",
        cache_dir: str = "data/cache_eegmat",
    ):
        self.data_root = data_root
        self.target_sfreq = target_sfreq
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.norm = norm
        self.cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)

        records_path = os.path.join(data_root, "RECORDS")
        if not os.path.exists(records_path):
            raise FileNotFoundError(
                f"RECORDS not found at {records_path}. "
                f"Download EEGMAT from PhysioNet first."
            )

        self.records: List[Dict] = []
        with open(records_path, "r") as f:
            for line in f:
                fname = line.strip()
                if not fname.endswith(".edf"):
                    continue
                # SubjectNN_K.edf → subject_num=NN, label=0 if K=1 else 1
                stem = fname[:-4]  # SubjectNN_K
                subj_part, k_part = stem.rsplit("_", 1)
                sub_num = int(subj_part.replace("Subject", ""))
                k = int(k_part)
                label = 0 if k == 1 else 1  # 1=baseline rest, 2=task

                eeg_path = os.path.join(data_root, fname)
                if not os.path.isfile(eeg_path):
                    print(f"[WARN] missing {eeg_path}")
                    continue

                stride_tag = "" if stride_sec is None else f"_s{stride_sec}"
                norm_tag = "" if norm == "zscore" else f"_n{norm}"
                cache_name = (
                    f"eegmat_{stem}_w{window_sec}{stride_tag}"
                    f"_sr{target_sfreq}{norm_tag}.pt"
                )
                self.records.append({
                    "subject_id": sub_num,
                    "patient_id": sub_num,  # alias for WindowDataset compat
                    "label": label,
                    "baseline_label": label,  # alias for WindowDataset compat
                    "stress_score": 0.0,  # placeholder for WindowDataset compat
                    "eeg_path": eeg_path,
                    "cache_name": cache_name,
                    "recording_kind": "rest" if label == 0 else "task",
                })

        n_unique = len(set(r["subject_id"] for r in self.records))
        n_task = sum(1 for r in self.records if r["label"] == 1)
        n_rest = sum(1 for r in self.records if r["label"] == 0)
        print(
            f"EEGMAT: {len(self.records)} recordings from {n_unique} subjects "
            f"(rest={n_rest}, task={n_task})"
        )

    def _preprocess(self, record: Dict) -> torch.Tensor:
        """Load EDF, select 19 channels, resample to target_sfreq, epoch,
        z-score, and cache.

        Uses pyedflib (no scipy.interpolate dependency) for EDF reading and
        scipy.fft.rfft/irfft for bandlimited resampling — both work in the
        `timm_eeg` env where the standard mne+scipy.interpolate path fails.
        """
        cache_path = os.path.join(self.cache_dir, record["cache_name"])
        if os.path.exists(cache_path):
            return torch.load(cache_path, weights_only=True)

        import pyedflib  # noqa: PLC0415

        reader = pyedflib.EdfReader(record["eeg_path"])
        try:
            n_signals = reader.signals_in_file
            raw_labels = reader.getSignalLabels()
            raw_sfreq = float(reader.getSampleFrequency(0))
            n_samples = reader.getNSamples()[0]

            # Build map: normalized 10-20 name → signal index
            sig_idx_by_name = {}
            for i, lbl in enumerate(raw_labels):
                norm = normalize_channel_name(lbl)
                if norm in sig_idx_by_name:
                    continue  # first occurrence wins
                sig_idx_by_name[norm] = i

            picks = []
            missing = []
            for tgt in COMMON_19:
                tgt_norm = normalize_channel_name(tgt)
                if tgt_norm in sig_idx_by_name:
                    picks.append(sig_idx_by_name[tgt_norm])
                else:
                    missing.append(tgt)
            if missing:
                raise ValueError(
                    f"{record['eeg_path']}: missing channels {missing}. "
                    f"Available (normalized): {sorted(sig_idx_by_name.keys())}"
                )

            data = np.stack([
                reader.readSignal(i).astype(np.float32) for i in picks
            ])  # (19, n_samples) in µV (pyedflib applies physical scaling)
        finally:
            reader.close()

        # Bandlimited resample to target_sfreq via scipy.fft.
        if raw_sfreq != self.target_sfreq:
            data = _fft_resample(data, raw_sfreq, self.target_sfreq)
        sfreq = self.target_sfreq

        # Epoch into fixed-length windows.
        samples_per_window = int(sfreq * self.window_sec)
        stride_samples = int(sfreq * (self.stride_sec or self.window_sec))
        total = data.shape[1]
        starts = list(range(0, total - samples_per_window + 1, stride_samples))
        if not starts:
            raise ValueError(
                f"{record['eeg_path']}: too short ({total/sfreq:.1f}s) "
                f"for {self.window_sec}s windows"
            )
        all_epochs = np.stack([
            data[:, s : s + samples_per_window] for s in starts
        ])  # (M, 19, T)

        if self.norm == "zscore":
            mean = all_epochs.mean(axis=-1, keepdims=True)
            std = all_epochs.std(axis=-1, keepdims=True) + 1e-6
            all_epochs = (all_epochs - mean) / std

        tensor = torch.tensor(all_epochs, dtype=torch.float32)
        torch.save(tensor, cache_path)
        return tensor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        epochs = self._preprocess(rec)
        n_epochs = epochs.shape[0]
        return epochs, rec["label"], n_epochs, rec["subject_id"]

    def get_labels(self) -> np.ndarray:
        return np.array([r["label"] for r in self.records])

    def get_patient_ids(self) -> np.ndarray:
        return np.array([r["subject_id"] for r in self.records])
