"""Extract per-window (non-pooled) frozen features for any dataset × any FM.

Generic per-window version of extract_frozen_all.py. Saves per-window features
with recording/subject metadata so downstream LP can match FT's per-window
training + prediction-pooling protocol.

Output npz:
    features       : (N_windows, embed_dim)  float32
    window_rec_idx : (N_windows,)            int32
    window_labels  : (N_windows,)            int32
    window_pids    : (N_windows,)            int64
    rec_labels     : (n_rec,)                int32
    rec_pids       : (n_rec,)                int64
    rec_n_epochs   : (n_rec,)                int32

Usage:
    PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
    $PY scripts/features/extract_frozen_all_perwindow.py --extractor labram --dataset eegmat --device cuda:5
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import baseline.labram   # noqa: F401
import baseline.cbramod  # noqa: F401
import baseline.reve     # noqa: F401
from baseline.abstract.factory import create_extractor
from pipeline.common_channels import COMMON_19

from scripts.features.extract_frozen_all import load_dataset, setup_extractor  # type: ignore

# labram: 2026-04-26 changed zscore → none (extractor does /100 internally)
MODEL_NORM = {"labram": "none", "cbramod": "none", "reve": "none"}


def extract_perwindow(extractor, dataset, device, batch_size=16, dataset_type="stress"):
    """Extract per-window frozen features, preserving recording grouping."""
    all_feats, all_rec_idx, all_labels, all_pids = [], [], [], []
    rec_labels_list, rec_pids_list, rec_n_epochs = [], [], []

    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            if dataset_type == "stress":
                epochs, label, _score, _n, pid = item
            else:
                epochs, label, _n, pid = item

            M = epochs.shape[0]
            epoch_feats = []
            for start in range(0, M, batch_size):
                batch = epochs[start:start + batch_size].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    feats = extractor(batch)
                epoch_feats.append(feats.float().cpu())

            epoch_feats = torch.cat(epoch_feats, dim=0).numpy().astype(np.float32)

            all_feats.append(epoch_feats)
            all_rec_idx.append(np.full(M, i, dtype=np.int32))
            all_labels.append(np.full(M, int(label), dtype=np.int32))
            all_pids.append(np.full(M, int(pid), dtype=np.int64))
            rec_labels_list.append(int(label))
            rec_pids_list.append(int(pid))
            rec_n_epochs.append(M)

            if (i + 1) % max(len(dataset) // 10, 1) == 0 or i == 0:
                print(f"  [{i+1}/{len(dataset)}] M={M}")

    features = np.concatenate(all_feats, axis=0)
    window_rec_idx = np.concatenate(all_rec_idx)
    window_labels = np.concatenate(all_labels)
    window_pids = np.concatenate(all_pids)
    return {
        "features": features,
        "window_rec_idx": window_rec_idx,
        "window_labels": window_labels,
        "window_pids": window_pids,
        "rec_labels": np.array(rec_labels_list, dtype=np.int32),
        "rec_pids": np.array(rec_pids_list, dtype=np.int64),
        "rec_n_epochs": np.array(rec_n_epochs, dtype=np.int32),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extractor", required=True, choices=list(MODEL_NORM.keys()))
    p.add_argument("--dataset", required=True,
                   choices=["stress", "adftd", "tdbrain", "eegmat",
                            "meditation", "sleepdep"])
    p.add_argument("--device", default="cuda:5")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--window-sec", type=float, default=5.0)
    p.add_argument("--adftd-n-splits", type=int, default=1,
                   help="ADFTD: pseudo-recordings per subject (default: 1). "
                        "Match the pooled extractor for consistent n_rec.")
    p.add_argument("--out-suffix", default="")
    args = p.parse_args()

    model_name = args.extractor
    norm = MODEL_NORM[model_name]
    print(f"[Per-window] Extracting frozen {model_name} × {args.dataset}")
    print(f"  norm={norm}, device={args.device}, window_sec={args.window_sec}")

    ds, pids, labels, n_ch = load_dataset(
        args.dataset, norm, window_sec=args.window_sec,
        adftd_n_splits=args.adftd_n_splits,
    )
    print(f"  {args.dataset}: {len(ds)} recordings, {len(np.unique(pids))} subjects, {n_ch}ch")

    extractor = setup_extractor(model_name, n_ch, args.device)
    print(f"  {model_name} loaded: embed_dim={extractor.embed_dim}")

    ds_type = "stress" if args.dataset in ("stress", "meditation", "sleepdep") else "other"
    out = extract_perwindow(extractor, ds, args.device, args.batch_size, ds_type)

    save_dir = "results/features_cache"
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(
        save_dir,
        f"frozen_{model_name}_{args.dataset}_perwindow{args.out_suffix}.npz"
    )
    np.savez_compressed(out_path, **out)
    print(f"\nSaved: {out_path}")
    print(f"  features: {out['features'].shape}, "
          f"n_rec={len(out['rec_labels'])}, total_windows={out['features'].shape[0]}")


if __name__ == "__main__":
    main()
