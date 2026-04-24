"""SAM40 Dataset Loader (Stress Assessment using Multimodal — 40 subjects).

40 subjects × 4 tasks (Relax, Arithmetic, Mirror_image, Stroop) × 3 trials = 480 .mat files.
Each trial is 25 seconds of 32-channel EEG at 128 Hz.

Binary stress classification: Relax=0 (control), {Arithmetic, Mirror_image, Stroop}=1 (stress).

Within-subject design: every subject has both Relax and Stress recordings.
Subject-level CV must keep all recordings from the same subject in the same fold.

Channels: 32-channel montage → COMMON_19 (19 standard 10-20) via 10-10→10-20 mapping.

Usage:
    ds = SAM40Dataset("data/sam40/Data/filtered_data")
    epochs, label, n_epochs, subject_id = ds[0]
"""
import os
import re
import warnings
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .common_channels import COMMON_19, normalize_channel_name

warnings.filterwarnings("ignore", message=".*boundary.*", category=RuntimeWarning)

# SAM40 32-channel order from Coordinates.locs
SAM40_CH = [
    'Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7',
    'CP5', 'CP1', 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10',
    'P8', 'P4', 'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4', 'FC2', 'F4',
    'F8', 'Fp2',
]

# 10-10 → 10-20 aliases for channel matching
_MAP_1010_TO_1020 = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}

# File naming regex: {Task}_sub_{SubjectID}_trial{TrialNum}.mat
_FNAME_RE = re.compile(
    r"^(Relax|Arithmetic|Mirror_image|Stroop)_sub_(\d+)_trial(\d+)\.mat$"
)


def _fft_resample(data: np.ndarray, src_sfreq: float, dst_sfreq: float) -> np.ndarray:
    """Bandlimited resample along the last axis using rfft/irfft.

    Handles both upsampling (128→200) and downsampling via zero-padding
    or truncation in the frequency domain.
    """
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
        # Upsampling: zero-pad in frequency domain
        pad_shape = list(X.shape)
        pad_shape[-1] = new_keep - X.shape[-1]
        X_new = np.concatenate(
            [X, np.zeros(pad_shape, dtype=X.dtype)], axis=-1
        )
    y = irfft(X_new, n=new_n, axis=-1)
    y = y * (new_n / n)
    return y.astype(np.float32, copy=False)


def _build_channel_picks() -> list[int]:
    """Pre-compute COMMON_19 channel indices from the SAM40 32-channel montage."""
    ch_idx = {}
    for i, ch in enumerate(SAM40_CH):
        up = ch.upper()
        mapped = _MAP_1010_TO_1020.get(up, up)
        norm = normalize_channel_name(mapped)
        if norm not in ch_idx:
            ch_idx[norm] = i

    picks = []
    for tgt in COMMON_19:
        tn = normalize_channel_name(tgt)
        if tn in ch_idx:
            picks.append(ch_idx[tn])
        else:
            raise ValueError(f"COMMON_19 channel {tgt} not found in SAM40 montage")
    return picks


# Pre-computed at module load — constant across all recordings
_CHANNEL_PICKS = _build_channel_picks()


class SAM40Dataset(Dataset):
    """PyTorch Dataset for SAM40 (Stress Assessment Multimodal, 40 subjects).

    Each item returns:
        epochs: (M, 19, T) float32 tensor — EEG epochs
        label: int — 0 (relax) or 1 (stress)
        n_epochs: int — actual number of valid epochs
        subject_id: int — unique subject identifier (1..40)

    Notes:
    - Within-subject: every subject has both relax and stress recordings.
    - Class ratio is 3:1 (360 stress vs 120 relax recordings).
    - Default stride=2.5s gives 9 overlapping windows per 25s trial.
    """

    def __init__(
        self,
        data_root: str,
        target_sfreq: float = 200.0,
        window_sec: float = 5.0,
        stride_sec: float | None = 2.5,
        norm: str = "zscore",
        cache_dir: str = "data/cache_sam40",
    ):
        self.data_root = data_root
        self.target_sfreq = target_sfreq
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.norm = norm
        self.cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)

        self.records: List[Dict] = []
        mat_files = sorted(os.listdir(data_root))
        for fname in mat_files:
            m = _FNAME_RE.match(fname)
            if m is None:
                continue
            task, sub_str, trial_str = m.groups()
            sub_num = int(sub_str)
            trial_num = int(trial_str)
            label = 0 if task == "Relax" else 1

            eeg_path = os.path.join(data_root, fname)
            if not os.path.isfile(eeg_path):
                print(f"[WARN] missing {eeg_path}")
                continue

            stride_tag = "" if stride_sec is None else f"_s{stride_sec}"
            norm_tag = "" if norm == "zscore" else f"_n{norm}"
            stem = f"{task}_sub_{sub_str}_trial{trial_str}"
            cache_name = (
                f"sam40_{stem}_w{window_sec}{stride_tag}"
                f"_sr{target_sfreq}{norm_tag}.pt"
            )
            self.records.append({
                "subject_id": sub_num,
                "patient_id": sub_num,
                "label": label,
                "baseline_label": label,
                "stress_score": 0.0,
                "eeg_path": eeg_path,
                "cache_name": cache_name,
                "task": task,
                "trial": trial_num,
                "recording_kind": "relax" if label == 0 else "stress",
            })

        n_unique = len(set(r["subject_id"] for r in self.records))
        n_stress = sum(1 for r in self.records if r["label"] == 1)
        n_relax = sum(1 for r in self.records if r["label"] == 0)
        print(
            f"SAM40: {len(self.records)} recordings from {n_unique} subjects "
            f"(stress={n_stress}, relax={n_relax})"
        )

    def _preprocess(self, record: Dict) -> torch.Tensor:
        """Load .mat, select COMMON_19 channels, resample, epoch, normalize, cache."""
        cache_path = os.path.join(self.cache_dir, record["cache_name"])
        if os.path.exists(cache_path):
            return torch.load(cache_path, weights_only=True)

        import scipy.io as sio

        mat = sio.loadmat(record["eeg_path"])
        all_data = mat["Clean_data"].astype(np.float64)  # (32, 3200)

        # Select COMMON_19 channels
        data = all_data[_CHANNEL_PICKS, :]  # (19, 3200)

        # Resample 128 → target_sfreq
        if 128.0 != self.target_sfreq:
            data = _fft_resample(data, 128.0, self.target_sfreq)
        sfreq = self.target_sfreq

        # Epoch into fixed-length windows with stride
        samples_per_window = int(sfreq * self.window_sec)
        stride_samples = int(sfreq * (self.stride_sec or self.window_sec))
        total = data.shape[1]
        starts = list(range(0, total - samples_per_window + 1, stride_samples))
        if not starts:
            raise ValueError(
                f"{record['eeg_path']}: too short ({total / sfreq:.1f}s) "
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
