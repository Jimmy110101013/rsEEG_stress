import os
import warnings
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .epoching import epoch_raw

# Silence MNE boundary/annotation warnings (handled by our epoching logic)
warnings.filterwarnings("ignore", message=".*boundary.*events.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*annotation.*outside data range.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*annotation.*expanding outside.*", category=RuntimeWarning)


class StressEEGDataset(Dataset):
    """PyTorch Dataset for stress EEG recordings.

    Each item returns:
        epochs: (M, C, T) float32 tensor — Z-scored EEG epochs
        baseline_label: int — 0 (normal) or 1 (increase)
        stress_score: float32 — Stress_Score / 100.0 (normalized to ~[0,1])
        n_epochs: int — actual number of valid epochs (before padding)

    Preprocessed tensors are cached to disk on first run.
    Delete the cache dir to force reprocessing.
    """

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        target_sfreq: float = 200.0,
        window_sec: float = 10.0,
        stride_sec: float | None = None,
        norm: str = "zscore",
        cache_dir: str = "data/cache",
    ):
        self.data_root = data_root
        self.target_sfreq = target_sfreq
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.norm = norm
        self.cache_dir = cache_dir

        df = pd.read_csv(csv_path)
        self.records: List[Dict] = []
        for _, row in df.iterrows():
            raw_path = row["File_Path"]
            rel = raw_path.lstrip("./").lstrip("/")
            if rel.startswith("data/"):
                rel = rel[len("data/"):]
            file_path = os.path.join(data_root, rel)

            if not os.path.isfile(file_path):
                print(f"[WARN] File not found, skipping: {file_path}")
                continue

            group = row["Group"]
            patient_id = int(row["Patient_ID"])
            trial_name = os.path.splitext(os.path.basename(file_path))[0]

            stride_tag = "" if stride_sec is None else f"_s{stride_sec}"
            norm_tag = "" if norm == "zscore" else f"_n{norm}"
            self.records.append(
                {
                    "file_path": file_path,
                    "baseline_label": 1 if group == "increase" else 0,
                    "stress_score": float(row["Stress_Score"]) / 100.0,
                    "patient_id": patient_id,
                    "cache_name": f"{group}_p{patient_id:02d}_{trial_name}{stride_tag}{norm_tag}.pt",
                }
            )

        print(f"Dataset: {len(self.records)} recordings loaded "
              f"({sum(r['baseline_label'] for r in self.records)} increase, "
              f"{sum(1 - r['baseline_label'] for r in self.records)} normal)")

        self._build_cache()

    def _build_cache(self):
        """Preprocess and cache any recordings not yet cached."""
        os.makedirs(self.cache_dir, exist_ok=True)
        n_cached, n_new = 0, 0
        for rec in self.records:
            cache_path = os.path.join(self.cache_dir, rec["cache_name"])
            if os.path.isfile(cache_path):
                n_cached += 1
                continue

            raw = mne.io.read_raw_eeglab(rec["file_path"], preload=True, verbose=False)
            epochs = epoch_raw(raw, self.target_sfreq, self.window_sec, self.stride_sec)  # (M, C, T)
            epochs = epochs * 1e6  # V → µV

            if self.norm == "zscore":
                mean = epochs.mean(axis=2, keepdims=True)
                std = epochs.std(axis=2, keepdims=True)
                std[std < 1e-6] = 1.0
                epochs = (epochs - mean) / std

            torch.save(torch.from_numpy(epochs).float(), cache_path)
            n_new += 1

        if n_new > 0:
            print(f"Cache: preprocessed {n_new} new recordings → {self.cache_dir}")
        print(f"Cache: {n_cached + n_new} recordings ready ({n_cached} cached, {n_new} new)")

    def __len__(self) -> int:
        return len(self.records)

    def get_labels(self) -> np.ndarray:
        return np.array([r["baseline_label"] for r in self.records])

    def get_patient_ids(self) -> np.ndarray:
        return np.array([r["patient_id"] for r in self.records])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float, int]:
        rec = self.records[idx]
        cache_path = os.path.join(self.cache_dir, rec["cache_name"])
        epochs_tensor = torch.load(cache_path, weights_only=True)

        return (
            epochs_tensor,
            rec["baseline_label"],
            rec["stress_score"],
            epochs_tensor.shape[0],
        )


def stress_collate_fn(
    batch: List[Tuple[torch.Tensor, int, float, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length epoch sequences by padding to the max M in the batch.

    Returns:
        epochs: (B, M_max, C, T)
        baseline_labels: (B,) long
        stress_scores: (B,) float32
        mask: (B, M_max) bool — True for valid epochs, False for padding
    """
    epochs_list, labels, scores, n_epochs_list = zip(*batch)

    max_m = max(n_epochs_list)
    B = len(batch)
    C, T = epochs_list[0].shape[1], epochs_list[0].shape[2]

    padded = torch.zeros(B, max_m, C, T)
    mask = torch.zeros(B, max_m, dtype=torch.bool)

    for i, (ep, n) in enumerate(zip(epochs_list, n_epochs_list)):
        padded[i, :n] = ep
        mask[i, :n] = True

    baseline_labels = torch.tensor(labels, dtype=torch.long)
    stress_scores = torch.tensor(scores, dtype=torch.float32)

    return padded, baseline_labels, stress_scores, mask
