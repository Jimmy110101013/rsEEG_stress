"""TDBRAIN Dataset Loader (Brainclinics Foundation).

Two Decades Brainclinics Research Archive for Insights in Neurophysiology.
1,274 psychiatric patients (MDD, ADHD, OCD, etc.) + 47 healthy controls.
26-channel EEG, 500 Hz, resting-state eyes-open (EO) and eyes-closed (EC).

Reference: van Dijk et al. (2022) Nature Scientific Data.

Usage:
    ds = TDBRAINDataset("data/tdbrain", target_dx="MDD", condition="both")
    epochs, label, n_epochs, subject_id = ds[0]
"""

import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .common_channels import COMMON_19

warnings.filterwarnings("ignore", category=RuntimeWarning)


# TDBRAIN uses standard 10-20 names; map to our uppercase COMMON_19
# The CSV headers may use mixed-case (Fp1, Fz, etc.)
_TDBRAIN_TO_COMMON = {
    'Fp1': 'FP1', 'Fp2': 'FP2',
    'F7': 'F7', 'F3': 'F3', 'Fz': 'FZ', 'F4': 'F4', 'F8': 'F8',
    'T3': 'T3', 'C3': 'C3', 'Cz': 'CZ', 'C4': 'C4', 'T4': 'T4',
    'T5': 'T5', 'P3': 'P3', 'Pz': 'PZ', 'P4': 'P4', 'T6': 'T6',
    'O1': 'O1', 'O2': 'O2',
    # Also handle uppercase variants
    'FP1': 'FP1', 'FP2': 'FP2', 'FZ': 'FZ', 'CZ': 'CZ', 'PZ': 'PZ',
    # 10-10 aliases (if present)
    'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6',
}


class TDBRAINDataset(Dataset):
    """PyTorch Dataset for TDBRAIN (MDD vs Healthy classification).

    Each item returns:
        epochs: (M, 19, T) float32 tensor — preprocessed EEG epochs
        label: int — 0 (healthy) or 1 (MDD)
        n_epochs: int — actual number of valid epochs
        subject_id: int — unique subject identifier
    """

    def __init__(
        self,
        data_root: str = "data/tdbrain",
        target_sfreq: float = 200.0,
        window_sec: float = 10.0,
        stride_sec: Optional[float] = None,
        norm: str = "zscore",
        condition: str = "both",  # "EO", "EC", or "both"
        target_dx: str = "MDD",  # diagnosis to classify vs HEALTHY
        cache_dir: str = "data/cache_tdbrain",
        n_splits: int = 1,
        orig_sfreq: float = 500.0,
    ):
        self.data_root = data_root
        self.target_sfreq = target_sfreq
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.norm = norm
        self.condition = condition.upper()
        self.target_dx = target_dx.upper()
        self.cache_dir = cache_dir
        self.n_splits = n_splits
        self.orig_sfreq = orig_sfreq

        os.makedirs(cache_dir, exist_ok=True)

        # Parse participants metadata
        self.records: List[Dict] = []
        self._parse_participants()

        n_unique = len(set(r["subject_id"] for r in self.records))
        n_pos = sum(1 for r in self.records if r["label"] == 1)
        n_neg = sum(1 for r in self.records if r["label"] == 0)
        cond_str = self.condition if self.condition != "BOTH" else "EO+EC"
        print(
            f"TDBRAIN: {len(self.records)} recordings from {n_unique} subjects "
            f"({self.target_dx}={n_pos}, HC={n_neg}, condition={cond_str})"
            + (f", {n_splits} splits/rec" if n_splits > 1 else "")
        )

    def _parse_participants(self):
        """Parse participants TSV and build record list."""
        import csv

        tsv_path = os.path.join(self.data_root, "TDBRAIN_participants_V2.tsv")
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(
                f"Participants file not found: {tsv_path}\n"
                "Download from https://brainclinics.com/resources/tdbrain-dataset/"
            )

        def _as_int(v, default=1):
            try:
                return int(float(v))
            except (ValueError, TypeError):
                return default

        def _is_one(v):
            try:
                return int(float(v)) == 1
            except (ValueError, TypeError):
                return False

        with open(tsv_path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                pid = row["participants_ID"]  # e.g., "sub-19681349"
                indication = row.get("indication", "").strip().upper()
                sess_id = _as_int(row.get("sessID", ""), default=1)
                has_ec = _is_one(row.get("EC", ""))
                has_eo = _is_one(row.get("EO", ""))

                # Determine label
                if indication == self.target_dx:
                    label = 1
                elif indication == "HEALTHY":
                    label = 0
                else:
                    continue  # Skip non-target diagnoses

                # Extract numeric subject ID from "sub-XXXXXXXX"
                sub_num = int(pid.replace("sub-", ""))

                # Determine which conditions to include
                conditions = []
                if self.condition in ("EO", "BOTH") and has_eo:
                    conditions.append("EO")
                if self.condition in ("EC", "BOTH") and has_ec:
                    conditions.append("EC")

                for cond in conditions:
                    # Find EEG CSV file path
                    eeg_path = self._find_eeg_path(pid, sess_id, cond)
                    if eeg_path is None:
                        continue

                    stride_tag = "" if self.stride_sec is None else f"_s{self.stride_sec}"
                    norm_tag = "" if self.norm == "zscore" else f"_n{self.norm}"
                    sess_tag = f"_sess{sess_id}" if sess_id > 1 else ""

                    for split_idx in range(self.n_splits):
                        split_tag = "" if self.n_splits == 1 else f"_split{split_idx}"
                        cache_name = (
                            f"tdbrain_{pid}_{cond}{sess_tag}"
                            f"_w{self.window_sec}{stride_tag}"
                            f"_sr{self.target_sfreq}{norm_tag}{split_tag}.pt"
                        )
                        self.records.append({
                            "subject_id": sub_num,
                            "patient_id": sub_num,
                            "label": label,
                            "baseline_label": label,
                            "stress_score": 0.0,  # placeholder for WindowDataset compat
                            "eeg_path": eeg_path,
                            "cache_name": cache_name,
                            "condition": cond,
                            "session": sess_id,
                            "split_idx": split_idx,
                            "indication": indication,
                        })

    def _find_eeg_path(self, pid: str, sess_id: int, condition: str) -> Optional[str]:
        """Locate the EEG CSV file for a given subject/session/condition.

        TDBRAIN derivatives structure:
            derivatives/{pid}/ses-{N}/eeg/{pid}_ses-{N}_task-rest{EC|EO}_eeg.csv
        """
        path = os.path.join(
            self.data_root, "derivatives",
            pid, f"ses-{sess_id}", "eeg",
            f"{pid}_ses-{sess_id}_task-rest{condition}_eeg.csv"
        )
        if os.path.isfile(path):
            return path
        return None

    def _load_csv_eeg(self, csv_path: str) -> np.ndarray:
        """Load EEG data from CSV, select 19 channels, resample.

        Returns:
            np.ndarray of shape (19, n_samples_resampled)
        """
        import pandas as pd

        df = pd.read_csv(csv_path)

        # Build channel mapping from CSV columns to COMMON_19
        csv_cols = df.columns.tolist()
        selected_cols = []
        channel_order = []

        for target_ch in COMMON_19:
            found = False
            for csv_col in csv_cols:
                normalized = _TDBRAIN_TO_COMMON.get(csv_col.strip())
                if normalized == target_ch:
                    selected_cols.append(csv_col)
                    channel_order.append(target_ch)
                    found = True
                    break
            if not found:
                # Try case-insensitive match
                for csv_col in csv_cols:
                    if csv_col.strip().upper() == target_ch:
                        selected_cols.append(csv_col)
                        channel_order.append(target_ch)
                        found = True
                        break
            if not found:
                raise ValueError(
                    f"Channel {target_ch} not found in CSV columns: {csv_cols[:30]}"
                )

        # Extract selected channels as numpy array (n_samples, 19) → (19, n_samples)
        data = df[selected_cols].values.T.astype(np.float64)

        # Resample from orig_sfreq to target_sfreq
        if self.orig_sfreq != self.target_sfreq:
            # Use rational resampling: target/orig ratio
            up = int(self.target_sfreq)
            down = int(self.orig_sfreq)
            # Simplify ratio
            from math import gcd
            g = gcd(up, down)
            up, down = up // g, down // g
            data = resample_poly(data, up, down, axis=1)

        return data.astype(np.float32)

    def _preprocess(self, record: Dict) -> torch.Tensor:
        """Load CSV, select channels, resample, epoch, normalize, cache."""
        cache_path = os.path.join(self.cache_dir, record["cache_name"])

        if os.path.exists(cache_path):
            return torch.load(cache_path, weights_only=True)

        # Load and resample
        data = self._load_csv_eeg(record["eeg_path"])  # (19, n_samples)

        # Epoch into windows
        n_channels, total_samples = data.shape
        samples_per_window = int(self.target_sfreq * self.window_sec)
        stride_samples = int(self.target_sfreq * (self.stride_sec or self.window_sec))

        starts = list(range(0, total_samples - samples_per_window + 1, stride_samples))
        if len(starts) == 0:
            raise ValueError(
                f"Recording too short ({total_samples / self.target_sfreq:.1f}s) "
                f"for {self.window_sec}s windows at {record['eeg_path']}"
            )

        all_epochs = np.stack([data[:, s:s + samples_per_window] for s in starts])
        # all_epochs shape: (M, 19, T)

        # Scale to µV if needed (TDBRAIN derivatives may be in V)
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
