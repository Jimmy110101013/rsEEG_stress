"""Extract per-window (non-pooled) frozen features for Stress.

Unlike `extract_frozen_stress.py` which mean-pools across windows within each
recording to produce (n_rec, embed_dim), this variant saves per-window features
so downstream LP can match FT's per-window training + prediction-pooling
protocol exactly.

Output npz keys:
    features       : (N_windows, embed_dim)  float32
    window_rec_idx : (N_windows,)            int32   — index into recording list
    window_labels  : (N_windows,)            int32   — per-recording label broadcast
    window_pids    : (N_windows,)            int64   — per-recording subject id broadcast
    rec_labels     : (n_rec,)                int32
    rec_pids       : (n_rec,)                int64
    rec_n_epochs   : (n_rec,)                int32

Usage:
    PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
    $PY scripts/features/extract_frozen_stress_perwindow.py --extractor labram --device cuda:6
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pipeline.dataset import StressEEGDataset
import baseline.labram   # noqa: F401 — register extractors
import baseline.cbramod  # noqa: F401
import baseline.reve     # noqa: F401
from baseline.abstract.factory import create_extractor

# labram: 2026-04-26 changed zscore → none (extractor does /100 internally)
MODEL_NORM = {"labram": "none", "cbramod": "none", "reve": "none"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extractor", required=True, choices=list(MODEL_NORM.keys()))
    p.add_argument("--device", default="cuda:6")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out-dir", default="results/features_cache")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = args.extractor
    norm = MODEL_NORM[model_name]
    cache_dir = "data/cache" if norm == "zscore" else f"data/cache_n{norm}"

    ds = StressEEGDataset(
        csv_path="data/comprehensive_labels.csv",
        data_root="data",
        target_sfreq=200.0,
        window_sec=5.0,
        norm=norm,
        cache_dir=cache_dir,
    )
    rec_labels_all = ds.get_labels()
    rec_pids_all = ds.get_patient_ids()
    print(f"Stress: {len(ds)} recordings, {len(np.unique(rec_pids_all))} subjects")

    # Stress is 30ch — use LaBraM's default OUR_CHANNELS mapping (no override).
    # This matches train_ft.py convention (which only overrides COMMON_19 for
    # 19ch datasets like ADFTD/TDBRAIN/EEGMAT).
    extractor = create_extractor(model_name)
    extractor.eval().to(args.device)
    print(f"{model_name} loaded: embed_dim={extractor.embed_dim}, norm={norm}")

    all_feats = []
    all_rec_idx = []
    all_labels = []
    all_pids = []
    rec_n_epochs = []

    with torch.no_grad():
        for i in range(len(ds)):
            epochs_tensor, label, stress, n_epochs, pid = ds[i]
            M = epochs_tensor.shape[0]
            epoch_feats = []
            for start in range(0, M, args.batch_size):
                batch = epochs_tensor[start: start + args.batch_size].to(args.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    feats = extractor(batch)
                epoch_feats.append(feats.float().cpu())
            epoch_feats = torch.cat(epoch_feats, dim=0).numpy().astype(np.float32)  # (M, D)

            all_feats.append(epoch_feats)
            all_rec_idx.append(np.full(M, i, dtype=np.int32))
            all_labels.append(np.full(M, int(label), dtype=np.int32))
            all_pids.append(np.full(M, int(pid), dtype=np.int64))
            rec_n_epochs.append(M)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(ds)}] M={M}")

    features = np.concatenate(all_feats, axis=0)
    window_rec_idx = np.concatenate(all_rec_idx)
    window_labels = np.concatenate(all_labels)
    window_pids = np.concatenate(all_pids)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"frozen_{model_name}_stress_perwindow.npz")
    np.savez_compressed(
        out_path,
        features=features,
        window_rec_idx=window_rec_idx,
        window_labels=window_labels,
        window_pids=window_pids,
        rec_labels=rec_labels_all.astype(np.int32),
        rec_pids=rec_pids_all.astype(np.int64),
        rec_n_epochs=np.array(rec_n_epochs, dtype=np.int32),
    )
    print(f"\nSaved: {out_path}")
    print(f"  features={features.shape}, n_rec={len(ds)}, total_windows={features.shape[0]}")


if __name__ == "__main__":
    main()
