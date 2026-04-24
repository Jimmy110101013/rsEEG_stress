"""Meditation Dataset Loader (OpenNeuro ds001787).

EEG meditation study (Delorme & Brandmeyer, 2016).
24 subjects (12 expert meditators, 12 novice) × 2 sessions = up to 48 recordings.
BioSemi 64-channel @ 256 Hz, ~45 min per session.

Between-subject binary classification: expert=1, novice=0.
Label is a subject-level property (group column in participants.tsv).

Channels: BioSemi 64ch → COMMON_19 via BioSemi→10-20 + 10-10→10-20 mapping.
Recording cropped to middle 300s by default for consistency with HHSA pipeline.

Usage:
    ds = MeditationDataset("data/meditation")
    epochs, label, n_epochs, subject_id = ds[0]
"""
import csv
import glob
import os
import re
import warnings
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .common_channels import COMMON_19, normalize_channel_name

warnings.filterwarnings("ignore", message=".*boundary.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*annotation.*", category=RuntimeWarning)

# BioSemi internal channel names → standard 10-20 channel names.
# Only the 19 channels we need are mapped (covers COMMON_19 after
# applying the 10-10→10-20 alias T7→T3, T8→T4, P7→T5, P8→T6).
BIOSEMI_TO_1020 = {
    'A1': 'Fp1', 'A5': 'F3', 'A7': 'F7', 'A13': 'C3', 'A15': 'T7',
    'A21': 'P3', 'A23': 'P7', 'A27': 'O1', 'A31': 'Pz',
    'B2': 'Fp2', 'B6': 'Fz', 'B8': 'F4', 'B10': 'F8', 'B16': 'Cz',
    'B18': 'C4', 'B20': 'T8', 'B26': 'P4', 'B28': 'P8', 'B32': 'O2',
}

# 10-10 → 10-20 aliases
_MAP_1010_TO_1020 = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def _fft_resample(data: np.ndarray, src_sfreq: float, dst_sfreq: float) -> np.ndarray:
    """Bandlimited resample along the last axis using rfft/irfft."""
    from scipy.fft import rfft, irfft

    n = data.shape[-1]
    new_n = int(round(n * dst_sfreq / src_sfreq))
    if new_n == n:
        return data.astype(np.float32, copy=False)
    X = rfft(data, axis=-1)
    new_keep = new_n // 2 + 1
    if new_keep <= X.shape[-1]:
        X_new = X[..., :new_keep]
    else:
        pad_shape = list(X.shape)
        pad_shape[-1] = new_keep - X.shape[-1]
        X_new = np.concatenate(
            [X, np.zeros(pad_shape, dtype=X.dtype)], axis=-1
        )
    y = irfft(X_new, n=new_n, axis=-1)
    y = y * (new_n / n)
    return y.astype(np.float32, copy=False)


def _parse_participants(data_root: str) -> dict[int, str]:
    """Parse participants.tsv and return {subject_num: group} mapping."""
    path = os.path.join(data_root, "participants.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"participants.tsv not found at {path}")

    sub_group = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = row["participant_id"]  # e.g., "sub-001"
            group = row.get("group", "").strip().lower()
            if group not in ("expert", "novice"):
                print(f"[WARN] Unknown group '{group}' for {pid}, skipping")
                continue
            sub_num = int(pid.split("-")[1])
            sub_group[sub_num] = group
    return sub_group


def _select_biosemi_channels(raw_ch_names: list[str], all_data: np.ndarray):
    """Select COMMON_19 channels from BioSemi data.

    Tries two strategies:
    1. BioSemi internal names (A1, B2, ...) via BIOSEMI_TO_1020 mapping
    2. Direct 10-10 names (Fp1, T7, ...) via normalize_channel_name

    Returns (data_19ch, ch_names_19) or raises ValueError.
    """
    # Strategy 1: BioSemi A/B naming
    ch_idx = {}
    for i, ch in enumerate(raw_ch_names):
        name_1020 = BIOSEMI_TO_1020.get(ch)
        if name_1020 is None:
            continue
        up = name_1020.upper()
        mapped = _MAP_1010_TO_1020.get(up, up)
        norm = normalize_channel_name(mapped)
        if norm not in ch_idx:
            ch_idx[norm] = i

    if len(ch_idx) >= 19:
        picks, ch_names = [], []
        for tgt in COMMON_19:
            tn = normalize_channel_name(tgt)
            if tn in ch_idx:
                picks.append(ch_idx[tn])
                ch_names.append(tgt)
        if len(picks) == 19:
            return all_data[picks, :], ch_names

    # Strategy 2: Direct 10-10 naming
    ch_idx = {}
    for i, ch in enumerate(raw_ch_names):
        up = ch.upper()
        mapped = _MAP_1010_TO_1020.get(up, up)
        norm = normalize_channel_name(mapped)
        if norm not in ch_idx:
            ch_idx[norm] = i

    picks, ch_names = [], []
    missing = []
    for tgt in COMMON_19:
        tn = normalize_channel_name(tgt)
        if tn in ch_idx:
            picks.append(ch_idx[tn])
            ch_names.append(tgt)
        else:
            missing.append(tgt)

    if missing:
        raise ValueError(
            f"Missing COMMON_19 channels: {missing}. "
            f"Available (first 10): {raw_ch_names[:10]}"
        )
    return all_data[picks, :], ch_names


class MeditationDataset(Dataset):
    """PyTorch Dataset for Meditation EEG (OpenNeuro ds001787).

    Each item returns:
        epochs: (M, 19, T) float32 tensor — EEG epochs
        label: int — 0 (novice) or 1 (expert)
        n_epochs: int — actual number of valid epochs
        subject_id: int — unique subject identifier (1..24)

    Notes:
    - Between-subject: each subject's label is fixed (expert or novice).
    - Multiple sessions per subject carry the same label.
    - Recordings cropped to middle crop_sec (default 300s) before epoching.
    """

    def __init__(
        self,
        data_root: str,
        target_sfreq: float = 200.0,
        window_sec: float = 5.0,
        stride_sec: float | None = None,
        norm: str = "zscore",
        cache_dir: str = "data/cache_meditation",
        crop_sec: float = 300.0,
    ):
        self.data_root = data_root
        self.target_sfreq = target_sfreq
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.norm = norm
        self.cache_dir = cache_dir
        self.crop_sec = crop_sec

        os.makedirs(cache_dir, exist_ok=True)

        # Parse subject metadata
        sub_group = _parse_participants(data_root)

        # Find all .bdf files
        bdf_pattern = os.path.join(data_root, "sub-*", "ses-*", "eeg", "*.bdf")
        bdf_files = sorted(glob.glob(bdf_pattern))

        self.records: List[Dict] = []
        for bdf_path in bdf_files:
            # Parse subject and session from BIDS directory structure
            # Path: .../sub-XXX/ses-YY/eeg/filename.bdf
            eeg_dir = os.path.dirname(bdf_path)
            ses_dir = os.path.dirname(eeg_dir)
            sub_dir = os.path.dirname(ses_dir)
            sub_id = os.path.basename(sub_dir)  # e.g., "sub-001"
            ses_id = os.path.basename(ses_dir)  # e.g., "ses-01"
            if not sub_id.startswith("sub-") or not ses_id.startswith("ses-"):
                continue
            sub_num = int(sub_id.split("-")[1])

            if sub_num not in sub_group:
                print(f"[WARN] {sub_id} not in participants.tsv, skipping")
                continue

            group = sub_group[sub_num]
            label = 1 if group == "expert" else 0

            stride_tag = "" if stride_sec is None else f"_s{stride_sec}"
            norm_tag = "" if norm == "zscore" else f"_n{norm}"
            crop_tag = f"_c{crop_sec}" if crop_sec else ""
            cache_name = (
                f"meditation_{sub_id}_{ses_id}_w{window_sec}{stride_tag}"
                f"_sr{target_sfreq}{norm_tag}{crop_tag}.pt"
            )
            self.records.append({
                "subject_id": sub_num,
                "patient_id": sub_num,
                "label": label,
                "baseline_label": label,
                "stress_score": 0.0,
                "eeg_path": bdf_path,
                "cache_name": cache_name,
                "group": group,
                "session": ses_id,
                "recording_kind": group,
            })

        n_unique = len(set(r["subject_id"] for r in self.records))
        n_expert = sum(1 for r in self.records if r["label"] == 1)
        n_novice = sum(1 for r in self.records if r["label"] == 0)
        print(
            f"Meditation: {len(self.records)} recordings from {n_unique} subjects "
            f"(expert={n_expert}, novice={n_novice})"
        )

    def _preprocess(self, record: Dict) -> torch.Tensor:
        """Load .bdf, select COMMON_19, crop, resample, epoch, normalize, cache."""
        cache_path = os.path.join(self.cache_dir, record["cache_name"])
        if os.path.exists(cache_path):
            return torch.load(cache_path, weights_only=True)

        import mne

        raw = mne.io.read_raw_bdf(record["eeg_path"], preload=False, verbose=False)
        sfreq = raw.info["sfreq"]

        # Crop to middle crop_sec
        if self.crop_sec:
            total_s = raw.n_times / sfreq
            crop_s = min(self.crop_sec, total_s)
            t_start = (total_s - crop_s) / 2.0
            raw.crop(tmin=t_start, tmax=t_start + crop_s)

        raw.load_data()
        all_data = raw.get_data() * 1e6  # V → µV
        raw_ch = list(raw.ch_names)
        del raw

        # Select COMMON_19 channels (handles BioSemi or 10-10 naming)
        data, _ = _select_biosemi_channels(raw_ch, all_data)
        del all_data

        # Resample
        if sfreq != self.target_sfreq:
            data = _fft_resample(data, sfreq, self.target_sfreq)

        # Epoch into windows
        target_sfreq = self.target_sfreq
        samples_per_window = int(target_sfreq * self.window_sec)
        stride_samples = int(target_sfreq * (self.stride_sec or self.window_sec))
        total = data.shape[1]
        starts = list(range(0, total - samples_per_window + 1, stride_samples))
        if not starts:
            raise ValueError(
                f"{record['eeg_path']}: too short ({total / target_sfreq:.1f}s) "
                f"for {self.window_sec}s windows"
            )
        all_epochs = np.stack([
            data[:, s: s + samples_per_window] for s in starts
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
        return epochs, rec["label"], rec["stress_score"], n_epochs, rec["subject_id"]

    def get_labels(self) -> np.ndarray:
        return np.array([r["label"] for r in self.records])

    def get_patient_ids(self) -> np.ndarray:
        return np.array([r["subject_id"] for r in self.records])
