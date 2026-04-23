"""Sleep Deprivation Dataset Loader (OpenNeuro ds004902).

71 subjects × 2 sessions (Normal Sleep vs Sleep Deprivation) × 2 tasks (eyes-open, eyes-closed).
61-channel EEG @ 500 Hz, 300s per task recording. Counterbalanced session order.

Binary classification: Normal Sleep (NS)=0, Sleep Deprivation (SD)=1.
Session-to-condition mapping uses SessionOrder from participants.tsv:
    "NS->SD": ses-1=NS(0), ses-2=SD(1)
    "SD->NS": ses-1=SD(1), ses-2=NS(0)

Within-subject design: same subject in both NS and SD conditions.
Only eyes-closed task is used (consistent with resting-state paradigm).

Channels: 61ch 10-10 → COMMON_19 via normalize_channel_name() (handles T7→T3 etc.).

Usage:
    ds = SleepDepDataset("data/sleep_deprivation")
    epochs, label, n_epochs, subject_id = ds[0]
"""
import csv
import glob
import os
import warnings
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .common_channels import COMMON_19, normalize_channel_name

warnings.filterwarnings("ignore", message=".*boundary.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*annotation.*", category=RuntimeWarning)

# 10-10 → 10-20 aliases for direct channel name mapping
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


def _parse_session_order(data_root: str) -> dict[int, str]:
    """Parse participants.tsv → {subject_num: session_order}.

    session_order is "NS->SD" or "SD->NS".
    """
    path = os.path.join(data_root, "participants.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"participants.tsv not found at {path}")

    sub_order = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = row["participant_id"]  # e.g., "sub-01"
            order = row.get("SessionOrder", "").strip()
            if order not in ("NS->SD", "SD->NS"):
                print(f"[WARN] Unknown SessionOrder '{order}' for {pid}, skipping")
                continue
            sub_num = int(pid.split("-")[1])
            sub_order[sub_num] = order
    return sub_order


def _session_to_label(ses_id: str, session_order: str) -> int:
    """Map session ID to NS(0) or SD(1) given the subject's session order.

    Args:
        ses_id: "ses-1" or "ses-2"
        session_order: "NS->SD" or "SD->NS"
    """
    ses_num = int(ses_id.split("-")[1])
    if session_order == "NS->SD":
        return 0 if ses_num == 1 else 1
    else:  # "SD->NS"
        return 1 if ses_num == 1 else 0


def _select_channels_manual(raw_ch_names: list[str], all_data: np.ndarray):
    """Select COMMON_19 channels from 10-10 named data.

    Returns (data_19ch, ch_names_19) or raises ValueError.
    """
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


class SleepDepDataset(Dataset):
    """PyTorch Dataset for Sleep Deprivation EEG (OpenNeuro ds004902).

    Each item returns:
        epochs: (M, 19, T) float32 tensor — EEG epochs
        label: int — 0 (normal sleep) or 1 (sleep deprivation)
        n_epochs: int — actual number of valid epochs
        subject_id: int — unique subject identifier

    Notes:
    - Within-subject: every included subject has both NS and SD recordings.
    - Subjects missing either session are skipped.
    - Only eyes-closed task is used.
    """

    def __init__(
        self,
        data_root: str,
        target_sfreq: float = 200.0,
        window_sec: float = 5.0,
        stride_sec: float | None = None,
        norm: str = "zscore",
        cache_dir: str = "data/cache_sleepdep",
    ):
        self.data_root = data_root
        self.target_sfreq = target_sfreq
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.norm = norm
        self.cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)

        # Parse session order metadata
        sub_order = _parse_session_order(data_root)

        # Find all eyes-closed .set files
        set_pattern = os.path.join(
            data_root, "sub-*", "ses-*", "eeg", "*task-eyesclosed*eeg.set"
        )
        set_files = sorted(glob.glob(set_pattern))

        # Group by subject to check for complete pairs
        from collections import defaultdict
        sub_sessions: dict[int, dict[str, str]] = defaultdict(dict)
        for set_path in set_files:
            # Parse from BIDS directory structure: .../sub-XX/ses-Y/eeg/filename.set
            eeg_dir = os.path.dirname(set_path)
            ses_dir_path = os.path.dirname(eeg_dir)
            sub_dir_path = os.path.dirname(ses_dir_path)
            sub_id = os.path.basename(sub_dir_path)
            ses_id = os.path.basename(ses_dir_path)
            if not sub_id.startswith("sub-") or not ses_id.startswith("ses-"):
                continue
            sub_num = int(sub_id.split("-")[1])
            sub_sessions[sub_num][ses_id] = set_path

        # Build records — require both sessions per subject
        self.records: List[Dict] = []
        skipped_incomplete = 0
        skipped_no_order = 0
        skipped_failed_cache = 0
        for sub_num in sorted(sub_sessions.keys()):
            sessions = sub_sessions[sub_num]
            if sub_num not in sub_order:
                skipped_no_order += 1
                continue
            if len(sessions) < 2:
                skipped_incomplete += 1
                continue

            order = sub_order[sub_num]
            sub_id = f"sub-{sub_num:02d}"

            # Check for failed cache markers — if any session of this subject
            # has a known-broken raw file, drop the entire subject (within-subject design).
            stride_tag = "" if stride_sec is None else f"_s{stride_sec}"
            norm_tag = "" if norm == "zscore" else f"_n{norm}"
            session_cache_names = {
                ses_id: (
                    f"sleepdep_{sub_id}_{ses_id}_ec_w{window_sec}{stride_tag}"
                    f"_sr{target_sfreq}{norm_tag}.pt"
                )
                for ses_id in sessions.keys()
            }
            has_failed = any(
                os.path.exists(os.path.join(cache_dir, cn + ".failed"))
                for cn in session_cache_names.values()
            )
            if has_failed:
                skipped_failed_cache += 1
                continue

            for ses_id, set_path in sorted(sessions.items()):
                label = _session_to_label(ses_id, order)
                cache_name = session_cache_names[ses_id]
                self.records.append({
                    "subject_id": sub_num,
                    "patient_id": sub_num,
                    "label": label,
                    "baseline_label": label,
                    "stress_score": 0.0,
                    "eeg_path": set_path,
                    "cache_name": cache_name,
                    "session": ses_id,
                    "condition": "NS" if label == 0 else "SD",
                    "recording_kind": "normal_sleep" if label == 0 else "sleep_deprived",
                })

        n_unique = len(set(r["subject_id"] for r in self.records))
        n_sd = sum(1 for r in self.records if r["label"] == 1)
        n_ns = sum(1 for r in self.records if r["label"] == 0)
        if skipped_incomplete or skipped_no_order or skipped_failed_cache:
            print(
                f"[INFO] Skipped {skipped_incomplete} subjects (incomplete sessions), "
                f"{skipped_no_order} subjects (no SessionOrder), "
                f"{skipped_failed_cache} subjects (failed cache markers)"
            )
        print(
            f"SleepDep: {len(self.records)} recordings from {n_unique} subjects "
            f"(SD={n_sd}, NS={n_ns})"
        )

    def _preprocess(self, record: Dict) -> torch.Tensor:
        """Load .set, select COMMON_19, resample, epoch, normalize, cache."""
        cache_path = os.path.join(self.cache_dir, record["cache_name"])
        if os.path.exists(cache_path):
            return torch.load(cache_path, weights_only=True)

        # Check for failure marker (so we don't retry known-bad files)
        fail_path = cache_path + ".failed"
        if os.path.exists(fail_path):
            raise RuntimeError(
                f"SleepDep record {record['eeg_path']} has a .failed marker; "
                f"__init__ should have filtered it out. Re-run the job so the "
                f"record is dropped at construction time."
            )

        import mne

        # Some .set files have boundary events or sample count mismatches
        try:
            raw = mne.io.read_raw_eeglab(record["eeg_path"], preload=True, verbose=False)
        except (RuntimeError, ValueError):
            try:
                raw = mne.io.read_raw_eeglab(record["eeg_path"], preload=False, verbose=False)
                raw.load_data()
            except (RuntimeError, ValueError) as e:
                print(f"[WARN] Cannot load {record['eeg_path']}: {e}")
                # Save failure marker so a re-run drops this record at __init__
                with open(fail_path, "w") as f:
                    f.write(str(e))
                raise RuntimeError(
                    f"SleepDep preprocess failed for {record['eeg_path']}: {e}. "
                    f"Marker written; please re-run so the record is dropped."
                ) from e

        sfreq = raw.info["sfreq"]
        all_data = raw.get_data() * 1e6  # V → µV
        raw_ch = list(raw.ch_names)
        del raw

        # Select COMMON_19 channels
        data, _ = _select_channels_manual(raw_ch, all_data)
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
