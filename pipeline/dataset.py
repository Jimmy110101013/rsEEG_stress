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
        max_duration: float | None = None,
    ):
        self.data_root = data_root
        self.target_sfreq = target_sfreq
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.norm = norm
        self.cache_dir = cache_dir

        df = pd.read_csv(csv_path)
        self.records: List[Dict] = []
        n_skipped_duration = 0
        for _, row in df.iterrows():
            raw_path = row["File_Path"]
            rel = raw_path.lstrip("./").lstrip("/")
            if rel.startswith("data/"):
                rel = rel[len("data/"):]
            file_path = os.path.join(data_root, rel)

            if not os.path.isfile(file_path):
                print(f"[WARN] File not found, skipping: {file_path}")
                continue

            # Duration filter using pre-computed Duration column
            if max_duration is not None and "Duration" in df.columns:
                if row["Duration"] > max_duration:
                    n_skipped_duration += 1
                    continue

            group = row["Group"]
            patient_id = int(row["Patient_ID"])
            trial_name = os.path.splitext(os.path.basename(file_path))[0]

            # Handle missing Stress_Score (NaN) — set to 0.0 sentinel
            stress_raw = row.get("Stress_Score", float("nan"))
            stress_score = float(stress_raw) / 100.0 if pd.notna(stress_raw) else 0.0

            stride_tag = "" if stride_sec is None else f"_s{stride_sec}"
            norm_tag = "" if norm == "zscore" else f"_n{norm}"
            win_tag = "" if window_sec == 10.0 else f"_w{window_sec}"
            self.records.append(
                {
                    "file_path": file_path,
                    "baseline_label": 1 if group == "increase" else 0,
                    "stress_score": stress_score,
                    "patient_id": patient_id,
                    "cache_name": f"{group}_p{patient_id:02d}_{trial_name}{win_tag}{stride_tag}{norm_tag}.pt",
                }
            )

        if n_skipped_duration > 0:
            print(f"Duration filter: skipped {n_skipped_duration} recordings > {max_duration}s")

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float, int, int]:
        rec = self.records[idx]
        cache_path = os.path.join(self.cache_dir, rec["cache_name"])
        epochs_tensor = torch.load(cache_path, weights_only=True)

        return (
            epochs_tensor,
            rec["baseline_label"],
            rec["stress_score"],
            epochs_tensor.shape[0],
            rec["patient_id"],
        )


class WindowDataset(Dataset):
    """Flat window-level dataset for end-to-end fine-tuning.

    Each item is a single (C, T) EEG window with its label.
    Supports per-class overlap augmentation: increase-class recordings
    can be re-sliced with overlapping windows to balance class counts.
    """

    def __init__(
        self,
        base_dataset: "StressEEGDataset",
        indices: np.ndarray,
        aug_overlap: float | None = None,
        label_mode: str = "dass",
        threshold_norm: float = 0.6,
        label_override: np.ndarray | None = None,
    ):
        """
        Args:
            base_dataset: StressEEGDataset with cached recordings.
            indices: Recording indices to include.
            aug_overlap: Overlap fraction for increase class (e.g. 0.75).
                         Normal class always uses non-overlapping windows.
            label_mode: "dass" (baseline_label), "subject-dass", or "dss" (score threshold).
            threshold_norm: Normalized threshold for DSS labels.
            label_override: Pre-computed labels array (indexed by dataset position).
                           When provided, uses these labels directly.
        """
        self.windows: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.patient_ids: List[int] = []
        self.rec_indices: List[int] = []  # original recording index (for aggregation)

        for idx in indices:
            rec = base_dataset.records[int(idx)]
            cache_path = os.path.join(base_dataset.cache_dir, rec["cache_name"])
            epochs = torch.load(cache_path, weights_only=True)  # (M, C, T)

            if label_override is not None:
                label = int(label_override[int(idx)])
            elif label_mode == "dass" or label_mode == "subject-dass":
                label = rec["baseline_label"]
            else:
                label = 1 if rec["stress_score"] >= threshold_norm else 0

            if aug_overlap is not None and label == 1 and epochs.shape[0] > 1:
                # Reconstruct full signal from non-overlapping windows, re-slice with overlap
                C, T = epochs.shape[1], epochs.shape[2]
                full_signal = epochs.permute(1, 0, 2).reshape(C, -1)  # (C, M*T)
                stride_samples = max(1, int(T * (1.0 - aug_overlap)))
                starts = range(0, full_signal.shape[1] - T + 1, stride_samples)
                for s in starts:
                    self.windows.append(full_signal[:, s : s + T].clone())
                    self.labels.append(label)
                    self.patient_ids.append(rec["patient_id"])
                    self.rec_indices.append(int(idx))
            else:
                for w in range(epochs.shape[0]):
                    self.windows.append(epochs[w])
                    self.labels.append(label)
                    self.patient_ids.append(rec["patient_id"])
                    self.rec_indices.append(int(idx))

        n0 = sum(1 for l in self.labels if l == 0)
        n1 = sum(1 for l in self.labels if l == 1)
        print(f"    WindowDataset: {len(self.windows)} windows (class_0={n0}, class_1={n1})")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, int]:
        return self.windows[idx], self.labels[idx], self.patient_ids[idx], self.rec_indices[idx]


class RecordingGroupSampler(torch.utils.data.Sampler):
    """Sampler ensuring each batch has multiple windows per recording
    while maintaining class diversity (LEAD-style index group-shuffling).

    Algorithm:
        1. Group window indices by recording, shuffle within each recording.
        2. Split each recording's windows into chunks of `group_size`.
        3. Shuffle all chunks globally, then concatenate.

    Result: consecutive windows in the iteration come from the same recording
    (in groups of `group_size`), but chunks from different recordings are
    interleaved. With batch_size=32 and group_size=8, each batch typically
    contains ~4 recordings with ~8 windows each — enough for meaningful
    recording-level aggregation without class-homogeneous batches.
    """

    def __init__(self, dataset: "WindowDataset", batch_size: int,
                 group_size: int = 8, drop_last: bool = False, seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_size = group_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Group window indices by recording
        from collections import defaultdict
        rec_to_windows = defaultdict(list)
        for i, rid in enumerate(self.dataset.rec_indices):
            rec_to_windows[rid].append(i)

        # For each recording: shuffle windows, split into chunks of group_size
        all_chunks = []
        for rid, wins in rec_to_windows.items():
            perm = torch.randperm(len(wins), generator=g).tolist()
            shuffled = [wins[p] for p in perm]
            for start in range(0, len(shuffled), self.group_size):
                all_chunks.append(shuffled[start:start + self.group_size])

        # Shuffle chunks globally — interleaves recordings
        chunk_order = torch.randperm(len(all_chunks), generator=g).tolist()
        indices = []
        for ci in chunk_order:
            indices.extend(all_chunks[ci])

        # Drop remainder if needed
        if self.drop_last and len(indices) % self.batch_size != 0:
            indices = indices[:len(indices) - len(indices) % self.batch_size]

        return iter(indices)

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n - n % self.batch_size
        return n

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def window_collate_fn(
    batch: List[Tuple[torch.Tensor, int, int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate for WindowDataset (all windows same shape)."""
    windows, labels, pids, rec_idxs = zip(*batch)
    return (
        torch.stack(windows),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(pids, dtype=torch.long),
        torch.tensor(rec_idxs, dtype=torch.long),
    )


def stress_collate_fn(
    batch: List[Tuple[torch.Tensor, int, float, int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length epoch sequences by padding to the max M in the batch.

    Returns:
        epochs: (B, M_max, C, T)
        baseline_labels: (B,) long
        stress_scores: (B,) float32
        mask: (B, M_max) bool — True for valid epochs, False for padding
        patient_ids: (B,) long
    """
    epochs_list, labels, scores, n_epochs_list, pids = zip(*batch)

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
    patient_ids = torch.tensor(pids, dtype=torch.long)

    return padded, baseline_labels, stress_scores, mask, patient_ids
