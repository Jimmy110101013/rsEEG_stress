"""Extract frozen LaBraM features for EEGMAT → results/features_cache/frozen_labram_eegmat_19ch.npz.

Mirrors scripts/extract_frozen_tdbrain.py for the EEGMAT positive-control
dataset. EEGMAT has within-subject labels (rest=0, task=1, every subject
contributes both), so the resulting frozen features can be compared to
the fine-tuned EEGMAT features in scripts/run_variance_analysis.py to
test the hypothesis that on within-subject task labels, fine-tuning
cleanly rewrites the LaBraM representation even on small data —
in contrast to the projection-only failure mode on Stress.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.eegmat_dataset import EEGMATDataset
from pipeline.common_channels import COMMON_19
from baseline.labram.channel_map import get_input_chans
from baseline.abstract.factory import create_extractor

DEVICE = os.environ.get("EEGMAT_DEVICE", "cuda:5")
SAVE_DIR = "results/features_cache"
os.makedirs(SAVE_DIR, exist_ok=True)


def main():
    ds = EEGMATDataset(
        "data/eegmat",
        target_sfreq=200.0,
        window_sec=5.0,
        norm="none",  # 2026-04-26: was "zscore"; LaBraM extractor now does /100 internally
        cache_dir="data/cache_eegmat_nnone",
    )
    pids = ds.get_patient_ids()
    labels = ds.get_labels()
    print(f"EEGMAT: {len(ds)} recordings, {len(np.unique(pids))} subjects")
    print(f"  Labels: {(labels==0).sum()} rest, {(labels==1).sum()} task")

    extractor = create_extractor("labram")
    extractor.input_chans = get_input_chans(COMMON_19)
    extractor.eval().to(DEVICE)
    print(f"LaBraM loaded: {len(extractor.input_chans)} positions")

    cache_path = os.path.join(SAVE_DIR, "frozen_labram_eegmat_19ch.npz")
    all_feats, all_pids, all_labels = [], [], []

    with torch.no_grad():
        for i in range(len(ds)):
            epochs, label, n_epochs, sub_id = ds[i]
            M = epochs.shape[0]
            epoch_feats = []
            for start in range(0, M, 16):
                batch = epochs[start:start + 16].to(DEVICE)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    feats = extractor(batch)
                epoch_feats.append(feats.float().cpu())
            epoch_feats = torch.cat(epoch_feats, dim=0)
            pooled = epoch_feats.mean(dim=0).numpy()
            all_feats.append(pooled)
            all_pids.append(sub_id)
            all_labels.append(label)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(ds)}] epochs={M}, pooled={pooled.shape}")

    features = np.stack(all_feats)
    all_pids = np.array(all_pids)
    all_labels = np.array(all_labels)

    np.savez_compressed(
        cache_path,
        features=features,
        patient_ids=all_pids,
        labels=all_labels,
    )
    print(f"\nSaved: {cache_path}")
    print(f"  features: {features.shape}")
    print(f"  pids:     {all_pids.shape}  unique={len(np.unique(all_pids))}")
    print(f"  labels:   {all_labels.shape}  rest={int((all_labels==0).sum())} task={int((all_labels==1).sum())}")


if __name__ == "__main__":
    main()
