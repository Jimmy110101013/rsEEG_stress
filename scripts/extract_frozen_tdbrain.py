"""Extract frozen LaBraM features for TDBRAIN → cache_dataset/features_tdbrain_19ch.npz.

Mirrors notebook cells 3, 5, 6 of Cross_Dataset_Signal_Strength.ipynb.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.tdbrain_dataset import TDBRAINDataset
from pipeline.common_channels import COMMON_19
from baseline.labram.channel_map import get_input_chans
from baseline.abstract.factory import create_extractor

DEVICE = 'cuda:5'
SAVE_DIR = 'results/cross_dataset'
os.makedirs(SAVE_DIR, exist_ok=True)


def main():
    # 1. Load TDBRAIN dataset
    ds = TDBRAINDataset(
        'data/tdbrain',
        target_sfreq=200.0,
        window_sec=5.0,       # LaBraM window
        norm='zscore',
        condition='both',
        target_dx='MDD',
        cache_dir='data/cache_tdbrain',
    )
    pids = ds.get_patient_ids()
    labels = ds.get_labels()
    print(f'TDBRAIN: {len(ds)} recordings, {len(np.unique(pids))} subjects')
    print(f'  Labels: {(labels==0).sum()} HC, {(labels==1).sum()} MDD')

    # 2. Load LaBraM with 19-channel mapping
    extractor = create_extractor('labram')
    extractor.input_chans = get_input_chans(COMMON_19)
    extractor.eval().to(DEVICE)
    print(f'LaBraM loaded: {len(extractor.input_chans)} positions')

    # 3. Extract and pool features (mean across epochs)
    cache_path = os.path.join(SAVE_DIR, 'features_tdbrain_19ch.npz')
    all_feats, all_pids, all_labels = [], [], []

    with torch.no_grad():
        for i in range(len(ds)):
            epochs, label, n_epochs, sub_id = ds[i]
            # Sub-batch to avoid OOM
            M = epochs.shape[0]
            epoch_feats = []
            for start in range(0, M, 16):
                batch = epochs[start:start + 16].to(DEVICE)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    feats = extractor(batch)
                epoch_feats.append(feats.float().cpu())
            epoch_feats = torch.cat(epoch_feats, dim=0)   # (M, 200)
            pooled = epoch_feats.mean(dim=0).numpy()      # (200,)
            all_feats.append(pooled)
            all_pids.append(sub_id)
            all_labels.append(label)

            if (i + 1) % 50 == 0 or i == 0:
                print(f'  [{i+1}/{len(ds)}] epochs={M}, pooled={pooled.shape}')

    features = np.stack(all_feats)
    all_pids = np.array(all_pids)
    all_labels = np.array(all_labels)

    np.savez_compressed(
        cache_path,
        features=features,
        patient_ids=all_pids,
        labels=all_labels,
    )
    print(f'\nSaved: {cache_path}')
    print(f'  features: {features.shape}')
    print(f'  pids:     {all_pids.shape}  unique={len(np.unique(all_pids))}')
    print(f'  labels:   {all_labels.shape}  hc={int((all_labels==0).sum())} mdd={int((all_labels==1).sum())}')


if __name__ == '__main__':
    main()
