"""Extract frozen features for any FM × any dataset combination.

Saves mean-pooled encoder features to
``results/features_cache/frozen_{model}_{dataset}_19ch.npz``.

Usage:
    PY=.conda/envs/stress/bin/python
    $PY scripts/extract_frozen_all.py --extractor cbramod --dataset adftd --device cuda:6
    $PY scripts/extract_frozen_all.py --extractor reve --dataset tdbrain --device cuda:7
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import baseline.labram   # noqa: F401
import baseline.cbramod  # noqa: F401
import baseline.reve     # noqa: F401
from baseline.abstract.factory import create_extractor
from pipeline.common_channels import COMMON_19

# Per-model norm (must match CLAUDE.md conventions)
MODEL_NORM = {"labram": "zscore", "cbramod": "none", "reve": "none"}


def load_dataset(name: str, norm: str, window_sec: float = 5.0):
    """Load dataset by name, return (dataset, patient_ids, labels)."""
    if name == "stress":
        from pipeline.dataset import StressEEGDataset
        cache_dir = "data/cache" if norm == "zscore" else f"data/cache_n{norm}"
        ds = StressEEGDataset(
            "data/comprehensive_labels.csv", "data",
            target_sfreq=200.0, window_sec=window_sec, norm=norm,
            cache_dir=cache_dir,
        )
        return ds, ds.get_patient_ids(), ds.get_labels(), 30
    elif name == "adftd":
        from pipeline.adftd_dataset import ADFTDDataset
        cache_suffix = "" if norm == "zscore" else f"_n{norm}"
        ds = ADFTDDataset(
            "data/adftd", binary=True, window_sec=window_sec,
            cache_dir=f"data/cache_adftd_split3{cache_suffix}", n_splits=3, norm=norm,
        )
        return ds, ds.get_patient_ids(), ds.get_labels(), 19
    elif name == "tdbrain":
        from pipeline.tdbrain_dataset import TDBRAINDataset
        cache_suffix = "" if norm == "zscore" else f"_n{norm}"
        ds = TDBRAINDataset(
            "data/tdbrain", target_sfreq=200.0, window_sec=window_sec,
            norm=norm, condition="both", target_dx="MDD",
            cache_dir=f"data/cache_tdbrain{cache_suffix}", n_splits=1,
        )
        return ds, ds.get_patient_ids(), ds.get_labels(), 19
    elif name == "eegmat":
        from pipeline.eegmat_dataset import EEGMATDataset
        cache_suffix = "" if norm == "zscore" else f"_n{norm}"
        ds = EEGMATDataset(
            "data/eegmat", target_sfreq=200.0, window_sec=window_sec,
            norm=norm, cache_dir=f"data/cache_eegmat{cache_suffix}",
        )
        return ds, ds.get_patient_ids(), ds.get_labels(), 19
    else:
        raise ValueError(f"Unknown dataset: {name}")


def setup_extractor(model_name: str, n_channels: int, device: str):
    """Create extractor with proper channel mapping."""
    extractor = create_extractor(model_name)

    if model_name == "labram" and n_channels == 19:
        from baseline.labram.channel_map import get_input_chans
        extractor.input_chans = get_input_chans(COMMON_19)

    if model_name == "reve" and n_channels == 19:
        extractor.set_channels(COMMON_19)

    # CBraMod handles any channel count natively (no mapping needed)

    extractor.eval().to(device)
    return extractor


def extract_features(extractor, dataset, device, batch_size=16, dataset_type="stress"):
    """Extract mean-pooled frozen features for all recordings."""
    all_feats, all_pids, all_labels = [], [], []

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

            epoch_feats = torch.cat(epoch_feats, dim=0)
            pooled = epoch_feats.mean(dim=0).numpy()
            all_feats.append(pooled)
            all_pids.append(pid)
            all_labels.append(label)

            if (i + 1) % max(len(dataset) // 10, 1) == 0 or i == 0:
                print(f"  [{i+1}/{len(dataset)}] epochs={M}, pooled={pooled.shape}")

    return np.stack(all_feats), np.array(all_pids), np.array(all_labels)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extractor", required=True, choices=list(MODEL_NORM.keys()))
    p.add_argument("--dataset", required=True,
                   choices=["stress", "adftd", "tdbrain", "eegmat"])
    p.add_argument("--device", default="cuda:6")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--window-sec", type=float, default=5.0,
                   help="Window size in seconds (default: 5.0)")
    p.add_argument("--out-suffix", default="",
                   help="Suffix for output filename (e.g. '_w10' → frozen_labram_adftd_19ch_w10.npz)")
    args = p.parse_args()

    model_name = args.extractor
    norm = MODEL_NORM[model_name]

    print(f"Extracting frozen {model_name} features for {args.dataset}")
    print(f"  norm={norm}, device={args.device}, window_sec={args.window_sec}")

    ds, pids, labels, n_ch = load_dataset(args.dataset, norm, window_sec=args.window_sec)
    print(f"  {args.dataset}: {len(ds)} recordings, {len(np.unique(pids))} subjects, {n_ch}ch")

    extractor = setup_extractor(model_name, n_ch, args.device)
    print(f"  {model_name} loaded: embed_dim={extractor.embed_dim}")

    ds_type = "stress" if args.dataset == "stress" else "other"
    features, pids_arr, labels_arr = extract_features(
        extractor, ds, args.device, args.batch_size, ds_type)

    save_dir = "results/features_cache"
    os.makedirs(save_dir, exist_ok=True)
    ch_tag = f"{n_ch}ch"
    out_path = os.path.join(save_dir, f"frozen_{model_name}_{args.dataset}_{ch_tag}{args.out_suffix}.npz")
    np.savez_compressed(
        out_path, features=features, patient_ids=pids_arr, labels=labels_arr)
    print(f"\nSaved: {out_path}")
    print(f"  features: {features.shape}, pids: {len(np.unique(pids_arr))}, "
          f"labels: {dict(zip(*np.unique(labels_arr, return_counts=True)))}")


if __name__ == "__main__":
    main()
