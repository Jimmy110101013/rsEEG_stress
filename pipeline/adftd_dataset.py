"""ADFTD Dataset Loader (OpenNeuro ds004504).

Alzheimer's Disease / Frontotemporal Dementia EEG dataset.
88 subjects (36 AD, 23 FTD, 29 healthy controls), 19ch, 500Hz, eyes-closed resting-state.

Usage:
    ds = ADFTDDataset("data/adftd", binary=True)  # AD vs healthy (65 subjects)
    epochs, label, n_epochs, subject_id = ds[0]
"""

import os
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


class ADFTDDataset(Dataset):
    """PyTorch Dataset for ADFTD (OpenNeuro ds004504).

    Each item returns:
        epochs: (M, 19, T) float32 tensor — Z-scored EEG epochs
        label: int — 0 (healthy) or 1 (AD)
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
        binary: bool = True,
        cache_dir: str = "data/cache_adftd",
        n_splits: int = 1,
    ):
        self.data_root = data_root
        self.target_sfreq = target_sfreq
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.norm = norm
        self.cache_dir = cache_dir
        self.n_splits = n_splits

        os.makedirs(cache_dir, exist_ok=True)

        # Parse participants.tsv
        participants_path = os.path.join(data_root, "participants.tsv")
        if not os.path.exists(participants_path):
            raise FileNotFoundError(f"participants.tsv not found at {participants_path}")

        self.records: List[Dict] = []
        with open(participants_path, "r") as f:
            header = f.readline().strip().split("\t")
            for line in f:
                fields = line.strip().split("\t")
                row = dict(zip(header, fields))

                sub_id = row["participant_id"]  # e.g., "sub-001"
                group = row.get("Group", row.get("group", ""))

                # Map group labels
                if group in ("A", "AD"):
                    label = 1  # Alzheimer's
                elif group in ("C", "CN", "HC"):
                    label = 0  # Healthy control
                elif group in ("F", "FTD"):
                    if binary:
                        continue  # Skip FTD for binary comparison
                    label = 2  # FTD (if 3-class)
                else:
                    print(f"[WARN] Unknown group '{group}' for {sub_id}, skipping")
                    continue

                # Find .set file
                sub_num = int(sub_id.split("-")[1])
                eeg_path = os.path.join(
                    data_root, sub_id, "eeg",
                    f"{sub_id}_task-eyesclosed_eeg.set"
                )

                if not os.path.isfile(eeg_path):
                    print(f"[WARN] File not found: {eeg_path}")
                    continue

                stride_tag = "" if stride_sec is None else f"_s{stride_sec}"
                norm_tag = "" if norm == "zscore" else f"_n{norm}"

                # Create n_splits pseudo-recordings per subject
                for split_idx in range(n_splits):
                    split_tag = "" if n_splits == 1 else f"_split{split_idx}"
                    cache_name = (
                        f"adftd_{sub_id}_w{window_sec}{stride_tag}"
                        f"_sr{target_sfreq}{norm_tag}{split_tag}.pt"
                    )
                    self.records.append({
                        "subject_id": sub_num,
                        "patient_id": sub_num,  # alias for WindowDataset compat
                        "label": label,
                        "baseline_label": label,  # alias for WindowDataset compat
                        "stress_score": 0.0,  # placeholder for WindowDataset compat
                        "eeg_path": eeg_path,
                        "cache_name": cache_name,
                        "group": group,
                        "split_idx": split_idx,
                    })

        n_unique = len(set(r["subject_id"] for r in self.records))
        n_ad = sum(1 for r in self.records if r["label"] == 1)
        n_hc = sum(1 for r in self.records if r["label"] == 0)
        print(
            f"ADFTD: {len(self.records)} recordings from {n_unique} subjects "
            f"(AD={n_ad}, HC={n_hc})"
            + (f", {n_splits} splits/subject" if n_splits > 1 else "")
        )

    def _preprocess(self, record: Dict) -> torch.Tensor:
        """Load, select channels, resample, epoch, normalize, and cache.

        When n_splits > 1, divides epochs into non-overlapping segments
        so each split gets a contiguous portion of the recording.
        """
        cache_path = os.path.join(self.cache_dir, record["cache_name"])

        if os.path.exists(cache_path):
            return torch.load(cache_path, weights_only=True)

        raw = mne.io.read_raw_eeglab(record["eeg_path"], preload=True, verbose=False)

        # Select common 19 channels
        raw = select_channels(raw, COMMON_19)

        # Epoch the full recording
        all_epochs = epoch_raw(
            raw,
            target_sfreq=self.target_sfreq,
            window_sec=self.window_sec,
            stride_sec=self.stride_sec,
        )

        # Scale to µV if needed
        if np.abs(all_epochs).max() < 0.01:
            all_epochs = all_epochs * 1e6

        # Normalize
        if self.norm == "zscore":
            mean = all_epochs.mean(axis=-1, keepdims=True)
            std = all_epochs.std(axis=-1, keepdims=True) + 1e-6
            all_epochs = (all_epochs - mean) / std

        # Split into segments if n_splits > 1
        if self.n_splits > 1:
            M = all_epochs.shape[0]
            chunk_size = M // self.n_splits
            split_idx = record["split_idx"]
            start = split_idx * chunk_size
            # Last split gets remaining epochs
            end = M if split_idx == self.n_splits - 1 else (split_idx + 1) * chunk_size
            epochs = all_epochs[start:end]
        else:
            epochs = all_epochs

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
