"""Extract frozen features for any FM on Stress dataset.

Saves mean-pooled encoder features (no fine-tuning) to
``results/features_cache/frozen_{model}_stress_19ch.npz``.

Usage:
    PY=.conda/envs/stress/bin/python
    $PY scripts/extract_frozen_stress.py --extractor cbramod --device cuda:6
    $PY scripts/extract_frozen_stress.py --extractor reve    --device cuda:6
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.dataset import StressEEGDataset
from pipeline.common_channels import COMMON_19
import baseline.labram   # noqa: F401 — register extractors
import baseline.cbramod  # noqa: F401
import baseline.reve     # noqa: F401
from baseline.abstract.factory import create_extractor

# Per-model norm (must match CLAUDE.md / train_ft.py conventions)
# labram: 2026-04-26 changed zscore → none (extractor does /100 internally)
MODEL_NORM = {
    "labram": "none",
    "cbramod": "none",
    "reve": "none",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extractor", required=True, choices=list(MODEL_NORM.keys()))
    p.add_argument("--device", default="cuda:6")
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    model_name = args.extractor
    norm = MODEL_NORM[model_name]

    # Dataset — cache dir includes norm tag so different norms don't collide
    # No max_duration filter: matches LaBraM frozen extraction (70 recs, all subjects).
    # The LP comparison is frozen-vs-frozen across models, not frozen-vs-FT,
    # so including all 70 recordings is fine and keeps parity with the existing
    # frozen_labram_stress_19ch.npz (70, 200).
    cache_dir = "data/cache" if norm == "zscore" else f"data/cache_n{norm}"
    ds = StressEEGDataset(
        csv_path="data/comprehensive_labels.csv",
        data_root="data",
        target_sfreq=200.0,
        window_sec=5.0,
        norm=norm,
        cache_dir=cache_dir,
    )
    labels = ds.get_labels()
    pids = ds.get_patient_ids()
    print(f"Stress: {len(ds)} recordings, {len(np.unique(pids))} subjects")
    print(f"  Labels: {(labels == 0).sum()} normal, {(labels == 1).sum()} increase")

    # Extractor
    extractor = create_extractor(model_name)
    if model_name == "labram":
        from baseline.labram.channel_map import get_input_chans
        extractor.input_chans = get_input_chans(COMMON_19)
    extractor.eval().to(args.device)
    print(f"{model_name} loaded: embed_dim={extractor.embed_dim}")

    # Extract
    all_feats = []
    with torch.no_grad():
        for i in range(len(ds)):
            epochs_tensor, label, stress, n_epochs, pid = ds[i]
            M = epochs_tensor.shape[0]
            epoch_feats = []
            for start in range(0, M, args.batch_size):
                batch = epochs_tensor[start : start + args.batch_size].to(args.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    feats = extractor(batch)
                epoch_feats.append(feats.float().cpu())
            epoch_feats = torch.cat(epoch_feats, dim=0)
            pooled = epoch_feats.mean(dim=0).numpy()
            all_feats.append(pooled)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(ds)}] epochs={M}, pooled={pooled.shape}")

    features = np.stack(all_feats)
    save_dir = "results/features_cache"
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"frozen_{model_name}_stress_19ch.npz")
    np.savez_compressed(out_path, features=features)
    print(f"\nSaved: {out_path}  shape={features.shape}")


if __name__ == "__main__":
    main()
