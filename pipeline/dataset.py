import os
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .epoching import epoch_raw


class StressEEGDataset(Dataset):
    """PyTorch Dataset for stress EEG recordings.

    Each item returns:
        epochs: (M, C, T) float32 tensor — Z-scored EEG epochs
        baseline_label: int — 0 (normal) or 1 (increase)
        stress_score: float32 — Stress_Score / 100.0 (normalized to ~[0,1])
        n_epochs: int — actual number of valid epochs (before padding)
    """

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        target_sfreq: float = 200.0,
        window_sec: float = 10.0,
    ):
        self.data_root = data_root
        self.target_sfreq = target_sfreq
        self.window_sec = window_sec

        df = pd.read_csv(csv_path)
        self.records: List[Dict] = []
        for _, row in df.iterrows():
            # Resolve file path relative to data_root
            raw_path = row["File_Path"]
            # CSV paths look like "../data/increase/2.set" — strip the leading "../"
            rel = raw_path.lstrip("./").lstrip("/")
            if rel.startswith("data/"):
                rel = rel[len("data/"):]
            file_path = os.path.join(data_root, rel)

            if not os.path.isfile(file_path):
                print(f"[WARN] File not found, skipping: {file_path}")
                continue

            self.records.append(
                {
                    "file_path": file_path,
                    "baseline_label": 1 if row["Group"] == "increase" else 0,
                    "stress_score": float(row["Stress_Score"]) / 100.0,
                    "patient_id": int(row["Patient_ID"]),
                }
            )

        print(f"Dataset: {len(self.records)} recordings loaded "
              f"({sum(r['baseline_label'] for r in self.records)} increase, "
              f"{sum(1 - r['baseline_label'] for r in self.records)} normal)")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float, int]:
        rec = self.records[idx]

        # Load EEG
        raw = mne.io.read_raw_eeglab(rec["file_path"], preload=True, verbose=False)
        epochs = epoch_raw(raw, self.target_sfreq, self.window_sec)  # (M, C, T)

        # MNE converts EEGLAB µV data to Volts (multiplies by 1e-6).
        # Restore to µV scale for FM input compatibility, then Z-score.
        epochs = epochs * 1e6  # V → µV

        # Z-score normalize per channel (across time within each epoch)
        mean = epochs.mean(axis=2, keepdims=True)
        std = epochs.std(axis=2, keepdims=True)
        std[std < 1e-6] = 1.0  # avoid division by zero
        epochs = (epochs - mean) / std

        epochs_tensor = torch.from_numpy(epochs).float()  # (M, C, T)

        return (
            epochs_tensor,
            rec["baseline_label"],
            rec["stress_score"],
            epochs_tensor.shape[0],  # n_epochs
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
