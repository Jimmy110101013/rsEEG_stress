"""Channel ablation importance for frozen FM representations.

For each channel, zero it out in raw EEG, re-extract frozen features,
measure cosine similarity drop vs original. Higher drop = more important.

Outputs:
  - JSON with per-channel importance scores
  - Topomap figures (30ch for Stress, 19ch for others)

Usage:
    PY=.conda/envs/stress/bin/python
    $PY scripts/channel_ablation_importance.py --device cuda:3
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import matplotlib
import mne

matplotlib.rcParams.update({"font.size": 10, "font.family": "sans-serif"})
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import baseline.labram   # noqa: F401
import baseline.cbramod  # noqa: F401
import baseline.reve     # noqa: F401
from baseline.abstract.factory import create_extractor
from pipeline.common_channels import COMMON_19

OUT_DIR = "results/studies/exp14_channel_importance"

# Per-model norm
MODEL_NORM = {"labram": "zscore", "cbramod": "none", "reve": "none"}

# Stress 30ch names (from raw .set files)
STRESS_30CH = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "FT7", "FC3", "FCZ", "FC4", "FT8",
    "T3", "C3", "CZ", "C4", "T4",
    "TP7", "CP3", "CPZ", "CP4", "TP8",
    "T5", "P3", "PZ", "P4", "T6",
    "O1", "OZ", "O2",
]

# MNE standard name mapping (our names → MNE 10-20 names)
MNE_NAME_MAP = {
    "FP1": "Fp1", "FP2": "Fp2", "F7": "F7", "F3": "F3", "FZ": "Fz",
    "F4": "F4", "F8": "F8", "FT7": "FT7", "FC3": "FC3", "FCZ": "FCz",
    "FC4": "FC4", "FT8": "FT8", "T3": "T3", "C3": "C3", "CZ": "Cz",
    "C4": "C4", "T4": "T4", "TP7": "TP7", "CP3": "CP3", "CPZ": "CPz",
    "CP4": "CP4", "TP8": "TP8", "T5": "T5", "P3": "P3", "PZ": "Pz",
    "P4": "P4", "T6": "T6", "O1": "O1", "OZ": "Oz", "O2": "O2",
}


def setup_extractor(model_name, n_channels, channel_names, device):
    extractor = create_extractor(model_name)
    if model_name == "labram":
        from baseline.labram.channel_map import get_input_chans
        extractor.input_chans = get_input_chans(channel_names)
    if model_name == "reve":
        extractor.set_channels(channel_names)
    extractor.eval().to(device)
    return extractor


def extract_recording_features(extractor, epochs_tensor, device, batch_size=16):
    """Extract mean-pooled features for one recording."""
    M = epochs_tensor.shape[0]
    feats = []
    for start in range(0, M, batch_size):
        batch = epochs_tensor[start:start + batch_size].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = extractor(batch)
        feats.append(f.float().cpu())
    return torch.cat(feats, dim=0).mean(dim=0).numpy()


def run_ablation(model_name, device, batch_size=16):
    """Run channel ablation for one FM on Stress dataset."""
    norm = MODEL_NORM[model_name]
    channel_names = STRESS_30CH
    n_ch = len(channel_names)

    # Load dataset
    from pipeline.dataset import StressEEGDataset
    cache_dir = "data/cache" if norm == "zscore" else f"data/cache_n{norm}"
    ds = StressEEGDataset(
        "data/comprehensive_labels.csv", "data",
        target_sfreq=200.0, window_sec=10.0, norm=norm,
        cache_dir=cache_dir,
    )
    print(f"\n{'='*60}")
    print(f"{model_name.upper()}: {len(ds)} recordings, {n_ch} channels")
    print(f"{'='*60}")

    # Setup extractor
    extractor = setup_extractor(model_name, n_ch, channel_names, device)

    # Step 1: Extract original (non-ablated) features for all recordings
    print("Extracting original features...")
    orig_feats = []
    with torch.no_grad():
        for i in range(len(ds)):
            epochs, label, score, n_ep, pid = ds[i]
            feat = extract_recording_features(extractor, epochs, device, batch_size)
            orig_feats.append(feat)
    orig_feats = np.stack(orig_feats)
    print(f"  Original features: {orig_feats.shape}")

    # Step 2: For each channel, zero it out and re-extract
    importance = np.zeros((len(ds), n_ch))

    with torch.no_grad():
        for ch_idx in range(n_ch):
            ablated_feats = []
            for i in range(len(ds)):
                epochs, label, score, n_ep, pid = ds[i]
                # Zero out channel ch_idx
                epochs_ablated = epochs.clone()
                epochs_ablated[:, ch_idx, :] = 0.0
                feat = extract_recording_features(
                    extractor, epochs_ablated, device, batch_size)
                ablated_feats.append(feat)

                # Importance = 1 - cosine_similarity
                sim = 1.0 - cosine(orig_feats[i], feat)
                importance[i, ch_idx] = 1.0 - sim

            mean_imp = importance[:, ch_idx].mean()
            print(f"  Ch {ch_idx:2d} ({channel_names[ch_idx]:>4s}): "
                  f"importance={mean_imp:.4f}")

    # Average across recordings
    mean_importance = importance.mean(axis=0)
    std_importance = importance.std(axis=0)

    return {
        "channel_names": channel_names,
        "mean_importance": mean_importance.tolist(),
        "std_importance": std_importance.tolist(),
        "per_recording": importance.tolist(),
    }


def create_mne_montage(channel_names):
    """Create MNE montage for topomap plotting."""
    mne_names = [MNE_NAME_MAP.get(ch, ch) for ch in channel_names]

    # Use standard 10-20 extended montage
    standard = mne.channels.make_standard_montage("standard_1020")
    available = {ch.upper(): ch for ch in standard.ch_names}

    # Map our names to standard positions
    pos_dict = {}
    for our_name, mne_name in zip(channel_names, mne_names):
        # Try direct match
        if mne_name in standard.ch_names:
            idx = standard.ch_names.index(mne_name)
            pos_dict[our_name] = standard.dig[idx + 3]["r"][:2]  # skip fiducials
        elif mne_name.upper() in available:
            matched = available[mne_name.upper()]
            idx = standard.ch_names.index(matched)
            pos_dict[our_name] = standard.dig[idx + 3]["r"][:2]

    # Build info object
    found = [ch for ch in channel_names if ch in pos_dict]
    positions = np.array([pos_dict[ch] for ch in found])

    info = mne.create_info(found, sfreq=200, ch_types="eeg")
    montage = mne.channels.make_dig_montage(
        ch_pos={ch: np.append(pos_dict[ch], 0) for ch in found},
        coord_frame="head",
    )
    info.set_montage(montage)
    return info, found


def plot_topomaps(results, out_dir):
    """Plot topomap for each FM."""
    models = list(results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    model_titles = {"labram": "LaBraM", "cbramod": "CBraMod", "reve": "REVE"}

    for ax, model in zip(axes, models):
        data = results[model]
        ch_names = data["channel_names"]
        importance = np.array(data["mean_importance"])

        info, found_chs = create_mne_montage(ch_names)

        # Match importance to found channels
        ch_to_imp = dict(zip(ch_names, importance))
        imp_matched = np.array([ch_to_imp[ch] for ch in found_chs])

        # Normalize to [0, 1] for each model
        imp_norm = (imp_matched - imp_matched.min()) / (imp_matched.max() - imp_matched.min() + 1e-10)

        im, _ = mne.viz.plot_topomap(
            imp_norm, info, axes=ax, show=False,
            cmap="YlOrRd", contours=0, sensors=True,
        )
        # Add channel labels manually
        pos_2d = np.array([info.get_montage().get_positions()["ch_pos"][ch][:2]
                           for ch in found_chs])
        for ch, p in zip(found_chs, pos_2d):
            ax.text(p[0], p[1], ch, fontsize=5, ha="center", va="center",
                    color="black", fontweight="bold")
        ax.set_title(f"{model_titles.get(model, model)}\n(max={importance.max():.4f})",
                     fontsize=12, fontweight="bold")

    plt.suptitle("Channel Importance (frozen, Stress 30ch)\nHigher = FM relies more on this channel",
                 fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/channel_importance_topomap.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{out_dir}/channel_importance_topomap.png", bbox_inches="tight", dpi=150)
    print(f"\nSaved → {out_dir}/channel_importance_topomap.{{pdf,png}}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--models", nargs="+", default=["labram", "cbramod", "reve"])
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    for model in args.models:
        results[model] = run_ablation(model, args.device, args.batch_size)

    # Save JSON
    with open(f"{OUT_DIR}/channel_importance.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_DIR}/channel_importance.json")

    # Plot
    plot_topomaps(results, OUT_DIR)


if __name__ == "__main__":
    main()
