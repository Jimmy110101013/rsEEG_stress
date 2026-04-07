"""TUAB Dataset Loader (Temple University Abnormal EEG Corpus).

~2,329 subjects, abnormal vs normal clinical EEG, .edf files, 250Hz, ~22 channels.
Requires TUH Corpus registration: https://isip.piconepress.com/projects/tuh_eeg/

Usage:
    ds = TUABDataset("data/tuab", max_subjects=150)  # 75 normal + 75 abnormal
    epochs, label, n_epochs, subject_id = ds[0]
"""

import os
import re
import warnings
from typing import Dict, List

import mne
import numpy as np
import torch
from torch.utils.data import Dataset

from .epoching import epoch_raw
from .common_channels import COMMON_19, select_channels

warnings.filterwarnings("ignore", message=".*boundary.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*annotation.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*stim.*", category=RuntimeWarning)


class TUABDataset(Dataset):
    """PyTorch Dataset for TUAB (Temple University Abnormal EEG).

    Each item returns:
        epochs: (M, 19, T) float32 tensor — Z-scored EEG epochs
        label: int — 0 (normal) or 1 (abnormal)
        n_epochs: int — actual number of valid epochs
        subject_id: int — unique subject identifier
    """

    def __init__(
        self,
        data_root: str,
        target_sfreq: float = 200.0,
        window_sec: float = 10.0,
        stride_sec: float | None = None,
        norm: str = "zscore",
        max_subjects: int = 150,
        seed: int = 42,
        cache_dir: str = "data/cache_tuab",
        split: str = "train",
    ):
        self.data_root = data_root
        self.target_sfreq = target_sfreq
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.norm = norm
        self.cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)

        # Scan directory structure: {root}/edf/{split}/{normal|abnormal}/.../*.edf
        all_files = []
        for label_name in ["normal", "abnormal"]:
            label = 0 if label_name == "normal" else 1
            label_dir = os.path.join(data_root, "edf", split, label_name)
            if not os.path.isdir(label_dir):
                # Try without edf/ prefix
                label_dir = os.path.join(data_root, split, label_name)
            if not os.path.isdir(label_dir):
                print(f"[WARN] Directory not found: {label_dir}")
                continue

            for root, dirs, files in os.walk(label_dir):
                for fname in files:
                    if not fname.endswith(".edf"):
                        continue
                    fpath = os.path.join(root, fname)
                    # Parse subject ID from TUAB filename: aaaaaXXX_sYYY_tZZZ.edf
                    match = re.match(r"(\w+?)(\d{4,})_s(\d+)_t(\d+)\.edf", fname)
                    if match:
                        subj_id = int(match.group(2))
                    else:
                        # Fallback: hash filename to get unique ID
                        subj_id = hash(fname) % 100000
                    all_files.append({
                        "subject_id": subj_id,
                        "label": label,
                        "eeg_path": fpath,
                        "filename": fname,
                    })

        if not all_files:
            raise FileNotFoundError(
                f"No .edf files found in {data_root}. "
                f"Expected structure: {{root}}/edf/{{train|eval}}/{{normal|abnormal}}/.../*.edf"
            )

        # Group by subject, take first recording per subject
        subj_map = {}
        for f in all_files:
            sid = f["subject_id"]
            if sid not in subj_map:
                subj_map[sid] = f

        # Stratified subsample
        rng = np.random.RandomState(seed)
        normal_subs = [v for v in subj_map.values() if v["label"] == 0]
        abnormal_subs = [v for v in subj_map.values() if v["label"] == 1]

        n_per_class = max_subjects // 2
        if len(normal_subs) > n_per_class:
            rng.shuffle(normal_subs)
            normal_subs = normal_subs[:n_per_class]
        if len(abnormal_subs) > n_per_class:
            rng.shuffle(abnormal_subs)
            abnormal_subs = abnormal_subs[:n_per_class]

        selected = normal_subs + abnormal_subs

        # Build records
        self.records: List[Dict] = []
        for s in selected:
            stride_tag = "" if stride_sec is None else f"_s{stride_sec}"
            norm_tag = "" if norm == "zscore" else f"_n{norm}"
            cache_name = (
                f"tuab_{s['subject_id']}_w{window_sec}{stride_tag}"
                f"_sr{target_sfreq}{norm_tag}.pt"
            )
            self.records.append({
                "subject_id": s["subject_id"],
                "label": s["label"],
                "eeg_path": s["eeg_path"],
                "cache_name": cache_name,
            })

        n_norm = sum(1 for r in self.records if r["label"] == 0)
        n_abnorm = sum(1 for r in self.records if r["label"] == 1)
        print(
            f"TUAB: {len(self.records)} subjects "
            f"(normal={n_norm}, abnormal={n_abnorm}) "
            f"from {len(subj_map)} total"
        )

    def _preprocess(self, record: Dict) -> torch.Tensor:
        """Load, select channels, resample, epoch, normalize, and cache."""
        cache_path = os.path.join(self.cache_dir, record["cache_name"])

        if os.path.exists(cache_path):
            return torch.load(cache_path, weights_only=True)

        raw = mne.io.read_raw_edf(record["eeg_path"], preload=True, verbose=False)

        # Select common 19 channels (handles "EEG FP1-REF" naming)
        raw = select_channels(raw, COMMON_19)

        # Epoch
        epochs = epoch_raw(
            raw,
            target_sfreq=self.target_sfreq,
            window_sec=self.window_sec,
            stride_sec=self.stride_sec,
        )

        # Scale to µV if needed
        if np.abs(epochs).max() < 0.01:
            epochs = epochs * 1e6

        # Normalize
        if self.norm == "zscore":
            mean = epochs.mean(axis=-1, keepdims=True)
            std = epochs.std(axis=-1, keepdims=True) + 1e-6
            epochs = (epochs - mean) / std

        tensor = torch.tensor(epochs, dtype=torch.float32)
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
