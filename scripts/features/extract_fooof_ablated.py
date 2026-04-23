"""Extract frozen FM features from FOOOF-ablated epochs.

Takes an npz produced by `scripts/analysis/fooof_ablation.py` and runs each of
the three ablation variants (aperiodic_removed, periodic_removed, both_removed)
through a frozen FM, producing per-window features analogous to
`extract_frozen_all_perwindow.py` output.

The ablation npz stores signals in RAW voltage domain. This script applies the
per-FM norm at extraction time, so a single ablation cache serves all 3 FMs.

Usage:
    PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
    $PY scripts/features/extract_fooof_ablated.py --extractor labram --dataset stress --device cuda:5
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

MODEL_NORM = {"labram": "zscore", "cbramod": "none", "reve": "none"}
MODEL_WINDOW = {"labram": 5.0, "cbramod": 5.0, "reve": 10.0}


def apply_norm(x: np.ndarray, norm: str) -> np.ndarray:
    """Apply per-FM norm to (M, C, T) raw-voltage epochs."""
    if norm == "none":
        return x
    if norm == "zscore":
        # Per-channel z-score within each recording would require grouping
        # across all windows of that recording. The cached dataset applies
        # z-score per-window per-channel. Match that here.
        # x: (M, C, T) → normalize per (m, c)
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        std = np.maximum(std, 1e-8)
        return (x - mean) / std
    raise ValueError(f"Unknown norm: {norm}")


def setup_extractor(model_name: str, n_channels: int, device: str):
    """Match extract_frozen_all.py channel mapping convention."""
    extractor = create_extractor(model_name)
    if model_name == "labram" and n_channels == 19:
        from baseline.labram.channel_map import get_input_chans
        extractor.input_chans = get_input_chans(COMMON_19)
    if model_name == "reve" and n_channels == 19:
        extractor.set_channels(COMMON_19)
    extractor.eval().to(device)
    return extractor


def extract_one_condition(
    extractor, epochs_all: np.ndarray, norm: str, device: str,
    batch_size: int = 16,
) -> np.ndarray:
    """Run FM on (N, C, T) epochs → (N, D) features. Applies per-FM norm."""
    N = epochs_all.shape[0]
    x_normed = apply_norm(epochs_all, norm).astype(np.float32)
    feats_all = []
    with torch.no_grad():
        for start in range(0, N, batch_size):
            batch = torch.from_numpy(x_normed[start:start + batch_size]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats = extractor(batch)
            feats_all.append(feats.float().cpu().numpy())
    return np.concatenate(feats_all, axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extractor", required=True, choices=list(MODEL_NORM.keys()))
    p.add_argument("--dataset", required=True, choices=["stress", "eegmat", "sleepdep", "adftd"])
    p.add_argument("--ablation-npz", default=None)
    p.add_argument("--device", default="cuda:5")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out-dir", default="results/features_cache/fooof_ablation")
    args = p.parse_args()

    model = args.extractor
    norm = MODEL_NORM[model]
    # ADFTD keeps per-FM window (refresh 2026-04-23); other datasets at legacy w=5.
    if args.ablation_npz:
        ablation_npz = args.ablation_npz
    elif args.dataset == "adftd":
        win = int(MODEL_WINDOW[model])
        ablation_npz = f"results/features_cache/fooof_ablation/adftd_norm_none_w{win}.npz"
    else:
        ablation_npz = f"results/features_cache/fooof_ablation/{args.dataset}_norm_none.npz"

    print(f"Extracting {model} × {args.dataset} FOOOF-ablated features")
    print(f"  norm={norm}, device={args.device}")
    data = np.load(ablation_npz)
    n_ch = 30 if args.dataset == "stress" else 19  # eegmat/sleepdep = 19ch

    extractor = setup_extractor(model, n_ch, args.device)
    print(f"  {model} loaded: embed_dim={extractor.embed_dim}")

    # Shared metadata across conditions
    meta = {
        "window_rec_idx": data["window_rec_idx"],
        "rec_labels": data["rec_labels"],
        "rec_pids": data["rec_pids"],
        "rec_n_epochs": data["rec_n_epochs"],
    }

    # Also need ORIGINAL (un-ablated) features to baseline against — load raw
    # from cached per-window file if available, else re-extract from dataset
    # (but dataset load for norm=='zscore' LaBraM would differ from norm=='none'
    # path, so we re-extract here consistently from the dataset cache).
    # Strategy: skip original and rely on existing per-window cache (matches norm).
    print(f"  (Original features: use existing {model}_{args.dataset}_perwindow.npz)")

    for cond in ("aperiodic_removed", "periodic_removed", "both_removed"):
        print(f"  === Condition: {cond} ===")
        epochs_ab = data[cond]  # (N, C, T) float32 raw voltage
        feats = extract_one_condition(
            extractor, epochs_ab, norm, args.device, args.batch_size)
        print(f"    features shape: {feats.shape}")
        out_path = os.path.join(
            args.out_dir,
            f"feat_{model}_{args.dataset}_{cond}.npz"
        )
        np.savez_compressed(out_path, features=feats, **meta)
        print(f"    → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
