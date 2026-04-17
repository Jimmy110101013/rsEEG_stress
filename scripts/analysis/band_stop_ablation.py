"""Band-stop ablation: remove frequency bands from raw EEG, measure FM feature change.

For each band (delta/theta/alpha/beta), apply a band-stop (notch) filter
to raw EEG, re-extract frozen FM features, compute cosine distance from
original. Higher distance = FM relies more on that frequency band.

This provides causal evidence (vs RSA which is correlational).

Usage:
    PY=.conda/envs/stress/bin/python
    $PY scripts/band_stop_ablation.py --device cuda:3
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 10, "font.family": "sans-serif"})

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import baseline.labram   # noqa: F401
import baseline.cbramod  # noqa: F401
import baseline.reve     # noqa: F401
from baseline.abstract.factory import create_extractor
from pipeline.common_channels import COMMON_19

OUT_DIR = "results/studies/exp14_channel_importance"
MODEL_NORM = {"labram": "zscore", "cbramod": "none", "reve": "none"}

STRESS_30CH = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "FT7", "FC3", "FCZ", "FC4", "FT8",
    "T3", "C3", "CZ", "C4", "T4",
    "TP7", "CP3", "CPZ", "CP4", "TP8",
    "T5", "P3", "PZ", "P4", "T6",
    "O1", "OZ", "O2",
]

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}

SFREQ = 200.0


def bandstop_filter(data, flo, fhi, fs, order=4):
    """Apply band-stop (notch) filter. data: (C, T) or (M, C, T)."""
    # Clamp to Nyquist
    nyq = fs / 2.0
    if fhi >= nyq:
        fhi = nyq - 1.0
    if flo <= 0:
        flo = 0.5
    sos = butter(order, [flo, fhi], btype="bandstop", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=-1).astype(np.float32)


def setup_extractor(model_name, channel_names, device):
    extractor = create_extractor(model_name)
    if model_name == "labram":
        from baseline.labram.channel_map import get_input_chans
        extractor.input_chans = get_input_chans(channel_names)
    if model_name == "reve":
        extractor.set_channels(channel_names)
    extractor.eval().to(device)
    return extractor


def extract_features(extractor, epochs_tensor, device, batch_size=16):
    """Mean-pooled features for one recording."""
    M = epochs_tensor.shape[0]
    feats = []
    for start in range(0, M, batch_size):
        batch = epochs_tensor[start:start + batch_size].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = extractor(batch)
        feats.append(f.float().cpu())
    return torch.cat(feats, dim=0).mean(dim=0).numpy()


def run_bandstop(model_name, device, batch_size=16, dataset="stress"):
    """Run band-stop ablation for one FM on a dataset."""
    norm = MODEL_NORM[model_name]

    if dataset == "stress":
        from pipeline.dataset import StressEEGDataset
        cache_dir = "data/cache" if norm == "zscore" else f"data/cache_n{norm}"
        ds = StressEEGDataset(
            "data/comprehensive_labels.csv", "data",
            target_sfreq=SFREQ, window_sec=10.0, norm=norm,
            cache_dir=cache_dir,
        )
        channel_names = STRESS_30CH
        ds_type = "stress"
    elif dataset == "eegmat":
        from pipeline.eegmat_dataset import EEGMATDataset
        ds = EEGMATDataset(
            "data/eegmat", target_sfreq=SFREQ, window_sec=5.0,
            norm=norm, cache_dir="data/cache_eegmat",
        )
        channel_names = COMMON_19
        ds_type = "eegmat"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"\n{'='*60}")
    print(f"{model_name.upper()} × {dataset}: {len(ds)} recordings, band-stop ablation")
    print(f"{'='*60}")

    extractor = setup_extractor(model_name, channel_names, device)

    # Extract original features
    print("Extracting original features...")
    orig_feats = []
    raw_epochs = []  # cache raw for filtering
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            if ds_type == "stress":
                epochs, label, score, n_ep, pid = item
            else:
                epochs, label, n_ep, pid = item
            feat = extract_features(extractor, epochs, device, batch_size)
            orig_feats.append(feat)
            raw_epochs.append(epochs.numpy())
    orig_feats = np.stack(orig_feats)
    print(f"  Original features: {orig_feats.shape}")

    # Per-band ablation
    results = {}
    with torch.no_grad():
        for band_name, (flo, fhi) in BANDS.items():
            distances = []
            for i in range(len(ds)):
                # Filter raw epochs: (M, C, T)
                filtered = bandstop_filter(raw_epochs[i], flo, fhi, SFREQ)
                filtered_tensor = torch.from_numpy(filtered)

                feat = extract_features(extractor, filtered_tensor, device, batch_size)
                dist = cosine(orig_feats[i], feat)
                distances.append(dist)

            distances = np.array(distances)
            results[band_name] = {
                "flo": flo,
                "fhi": fhi,
                "mean_distance": round(float(np.mean(distances)), 6),
                "std_distance": round(float(np.std(distances)), 6),
                "per_recording": distances.tolist(),
            }
            print(f"  {band_name} ({flo}-{fhi} Hz): "
                  f"cosine_dist={np.mean(distances):.6f} ± {np.std(distances):.6f}")

    return results


def plot_results(all_ds_results, out_dir):
    """Plot band-stop ablation results for multiple datasets."""
    datasets = list(all_ds_results.keys())
    n_ds = len(datasets)
    ds_labels = {"stress": "Stress", "eegmat": "EEGMAT"}
    model_labels_map = {"labram": "LaBraM", "cbramod": "CBraMod", "reve": "REVE"}
    model_colors_map = {"labram": "#4C72B0", "cbramod": "#DD8452", "reve": "#55A868"}
    band_names = list(BANDS.keys())

    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5), sharey=True)
    if n_ds == 1:
        axes = [axes]

    for ax_idx, ds_name in enumerate(datasets):
        ax = axes[ax_idx]
        ds_results = all_ds_results[ds_name]
        models_in = list(ds_results.keys())

        x = np.arange(len(band_names))
        width = 0.22

        for i, model in enumerate(models_in):
            vals = [ds_results[model][b]["mean_distance"] for b in band_names]
            errs = [ds_results[model][b]["std_distance"] for b in band_names]
            ax.bar(x + i * width, vals, width, yerr=errs, capsize=3,
                   color=model_colors_map[model], alpha=0.85,
                   edgecolor="white", linewidth=0.5,
                   label=model_labels_map[model] if ax_idx == 0 else None)
            for j, (v, e) in enumerate(zip(vals, errs)):
                ax.text(x[j] + i * width, v + e + 0.001,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6)

        ax.set_xticks(x + width)
        ax.set_xticklabels([f"{b.capitalize()}\n({BANDS[b][0]}-{BANDS[b][1]} Hz)"
                            for b in band_names])
        ax.set_title(ds_labels.get(ds_name, ds_name), fontsize=12, fontweight="bold")
        if ax_idx == 0:
            ax.set_ylabel("Cosine distance\n(higher = FM relies more on this band)")
            ax.legend(fontsize=9)
        ax.set_ylim(bottom=0)

    plt.suptitle("Band-Stop Ablation: Which frequency band removal hurts FM representations most?",
                 fontweight="bold", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/band_stop_ablation.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{out_dir}/band_stop_ablation.png", bbox_inches="tight", dpi=150)
    print(f"\nSaved → {out_dir}/band_stop_ablation.{{pdf,png}}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--models", nargs="+", default=["labram", "cbramod", "reve"])
    p.add_argument("--datasets", nargs="+", default=["stress", "eegmat"])
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    all_ds_results = {}
    for ds_name in args.datasets:
        all_ds_results[ds_name] = {}
        for model in args.models:
            all_ds_results[ds_name][model] = run_bandstop(
                model, args.device, args.batch_size, dataset=ds_name)

    with open(f"{OUT_DIR}/band_stop_ablation.json", "w") as f:
        json.dump(all_ds_results, f, indent=2)
    print(f"\nSaved → {OUT_DIR}/band_stop_ablation.json")

    plot_results(all_ds_results, OUT_DIR)


if __name__ == "__main__":
    main()
