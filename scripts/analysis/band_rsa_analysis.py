"""Band-specific RSA: correlate FM representations with theta/alpha/beta band power.

For each FM × dataset, compute:
1. Band power RDM (theta 4-8Hz, alpha 8-13Hz, beta 13-30Hz)
2. FM feature RDM (frozen)
3. RSA correlation between FM RDM and each band RDM

Shows which frequency band each FM's representation aligns with most.

Usage:
    PY=.conda/envs/stress/bin/python
    $PY scripts/band_rsa_analysis.py
"""
import json
import os
import sys

import numpy as np
from scipy.signal import welch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 10, "font.family": "sans-serif"})

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FEAT_DIR = "results/features_cache"
OUT_DIR = "results/studies/exp14_channel_importance"

models = ["labram", "cbramod", "reve"]
model_labels = ["LaBraM", "CBraMod", "REVE"]
model_colors = {"labram": "#4C72B0", "cbramod": "#DD8452", "reve": "#55A868"}

bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}
band_colors = {
    "delta": "#9467BD",
    "theta": "#2CA02C",
    "alpha": "#D62728",
    "beta": "#FF7F0E",
}

SFREQ = 200.0


def compute_band_power(cache_dir, records, norm, sfreq=200.0):
    """Compute per-recording band power vectors from cached EEG."""
    import torch

    all_powers = []
    for rec in records:
        cache_path = os.path.join(cache_dir, rec["cache_name"])
        epochs = torch.load(cache_path, weights_only=True).numpy()  # (M, C, T)
        M, C, T = epochs.shape

        # Average PSD across all windows
        band_feats = []
        for band_name, (flo, fhi) in bands.items():
            # Welch PSD per channel, averaged across windows
            powers = []
            for ch in range(C):
                signal = epochs[:, ch, :].reshape(-1)  # concatenate windows
                freqs, psd = welch(signal, fs=sfreq, nperseg=min(512, len(signal)))
                band_mask = (freqs >= flo) & (freqs < fhi)
                powers.append(np.mean(psd[band_mask]))
            band_feats.extend(powers)  # C values per band

        all_powers.append(band_feats)

    return np.array(all_powers)  # (N_rec, n_bands * C)


def compute_rdm(features):
    """Compute representational dissimilarity matrix (1 - Pearson r)."""
    return squareform(pdist(features, metric="correlation"))


def rsa_correlation(rdm1, rdm2):
    """Spearman correlation between upper triangles of two RDMs."""
    triu = np.triu_indices_from(rdm1, k=1)
    r, p = spearmanr(rdm1[triu], rdm2[triu])
    return r, p


def run_stress_analysis():
    """Run band RSA on Stress dataset."""
    from pipeline.dataset import StressEEGDataset

    results = {}

    for model in models:
        norm = {"labram": "zscore", "cbramod": "none", "reve": "none"}[model]
        cache_dir = "data/cache" if norm == "zscore" else f"data/cache_n{norm}"

        # Load dataset for cache paths
        ds = StressEEGDataset(
            "data/comprehensive_labels.csv", "data",
            target_sfreq=SFREQ, window_sec=10.0, norm=norm,
            cache_dir=cache_dir,
        )

        # Load frozen features
        feat = np.load(f"{FEAT_DIR}/frozen_{model}_stress_30ch.npz")
        X_fm = feat["features"]  # (70, embed_dim)

        # FM RDM
        rdm_fm = compute_rdm(X_fm)

        # Band power
        print(f"\n{model}: computing band power from {cache_dir}...")
        band_power_full = compute_band_power(cache_dir, ds.records, norm, SFREQ)
        # (70, n_bands * 30ch)

        n_ch = 30
        band_names = list(bands.keys())
        model_results = {"fm_rdm_shape": rdm_fm.shape[0]}

        # Per-band RSA
        for i, band_name in enumerate(band_names):
            # Extract this band's power across channels
            bp = band_power_full[:, i * n_ch:(i + 1) * n_ch]  # (70, 30)
            rdm_band = compute_rdm(bp)
            r, p = rsa_correlation(rdm_fm, rdm_band)
            model_results[band_name] = {"rsa_r": round(r, 4), "rsa_p": round(p, 6)}
            print(f"  {band_name} (RSA r={r:.4f}, p={p:.6f})")

        # Also: all-band combined
        rdm_all = compute_rdm(band_power_full)
        r, p = rsa_correlation(rdm_fm, rdm_all)
        model_results["all_bands"] = {"rsa_r": round(r, 4), "rsa_p": round(p, 6)}
        print(f"  all_bands (RSA r={r:.4f}, p={p:.6f})")

        results[model] = model_results

    return results


def run_other_dataset(dataset_name):
    """Run band RSA on ADFTD/TDBRAIN/EEGMAT."""
    results = {}

    for model in models:
        norm = {"labram": "zscore", "cbramod": "none", "reve": "none"}[model]

        # Load frozen features
        feat = np.load(f"{FEAT_DIR}/frozen_{model}_{dataset_name}_19ch.npz")
        X_fm = feat["features"]

        # FM RDM
        rdm_fm = compute_rdm(X_fm)

        # Load dataset for cache paths
        if dataset_name == "adftd":
            from pipeline.adftd_dataset import ADFTDDataset
            cache_suffix = "" if norm == "zscore" else f"_n{norm}"
            window_sec = 10.0 if model == "reve" else 5.0
            ds = ADFTDDataset(
                "data/adftd", binary=True, window_sec=window_sec,
                cache_dir=f"data/cache_adftd_split1{cache_suffix}",
                n_splits=1, norm=norm,
            )
        elif dataset_name == "eegmat":
            from pipeline.eegmat_dataset import EEGMATDataset
            ds = EEGMATDataset(
                "data/eegmat", target_sfreq=200.0, window_sec=5.0,
                norm=norm, cache_dir="data/cache_eegmat",
            )
        else:
            print(f"  Skipping {dataset_name} (not implemented)")
            continue

        print(f"\n{model} × {dataset_name}: computing band power...")
        band_power_full = compute_band_power(ds.cache_dir, ds.records, norm, SFREQ)
        n_ch = 19
        band_names = list(bands.keys())
        model_results = {"fm_rdm_shape": rdm_fm.shape[0]}

        for i, band_name in enumerate(band_names):
            bp = band_power_full[:, i * n_ch:(i + 1) * n_ch]
            rdm_band = compute_rdm(bp)
            r, p = rsa_correlation(rdm_fm, rdm_band)
            model_results[band_name] = {"rsa_r": round(r, 4), "rsa_p": round(p, 6)}
            print(f"  {band_name} (RSA r={r:.4f}, p={p:.6f})")

        rdm_all = compute_rdm(band_power_full)
        r, p = rsa_correlation(rdm_fm, rdm_all)
        model_results["all_bands"] = {"rsa_r": round(r, 4), "rsa_p": round(p, 6)}
        print(f"  all_bands (RSA r={r:.4f}, p={p:.6f})")

        results[model] = model_results

    return results


def plot_results(all_results, out_dir):
    """Plot band RSA results."""
    datasets = list(all_results.keys())
    n_ds = len(datasets)
    dataset_labels_map = {"stress": "Stress", "adftd": "ADFTD", "eegmat": "EEGMAT"}
    band_names = list(bands.keys())

    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5),
                             sharey=True)
    if n_ds == 1:
        axes = [axes]

    for j, ds in enumerate(datasets):
        ax = axes[j]
        x = np.arange(len(band_names))
        width = 0.22

        for i, (model, mlabel) in enumerate(zip(models, model_labels)):
            if model not in all_results[ds]:
                continue
            data = all_results[ds][model]
            vals = [data[b]["rsa_r"] for b in band_names]
            pvals = [data[b]["rsa_p"] for b in band_names]

            bars = ax.bar(x + i * width, vals, width,
                          color=model_colors[model], alpha=0.85,
                          edgecolor="white", linewidth=0.5,
                          label=mlabel if j == 0 else None)

            # Significance markers
            for k, (v, p) in enumerate(zip(vals, pvals)):
                if p < 0.001:
                    marker = "***"
                elif p < 0.01:
                    marker = "**"
                elif p < 0.05:
                    marker = "*"
                else:
                    marker = "ns"
                y_pos = v + 0.01 if v >= 0 else v - 0.03
                ax.text(x[k] + i * width, y_pos, marker,
                        ha="center", va="bottom", fontsize=7)

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticks(x + width)
        ax.set_xticklabels([b.capitalize() for b in band_names])
        ax.set_title(dataset_labels_map.get(ds, ds), fontsize=12, fontweight="bold")
        if j == 0:
            ax.set_ylabel("RSA (Spearman r)\nFM features ↔ band power")
            ax.legend(fontsize=8, loc="upper left")

    plt.suptitle("Band-Specific RSA: Which frequency bands do FM representations align with?",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/band_rsa.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{out_dir}/band_rsa.png", bbox_inches="tight", dpi=150)
    print(f"\nSaved → {out_dir}/band_rsa.{{pdf,png}}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = {}

    # Stress (30ch)
    all_results["stress"] = run_stress_analysis()

    # EEGMAT (19ch) — contrast with Stress
    all_results["eegmat"] = run_other_dataset("eegmat")

    # ADFTD (19ch, binary, split1) — subject-label × strong-aligned cell (2026-04-23)
    all_results["adftd"] = run_other_dataset("adftd")

    # Save JSON
    with open(f"{OUT_DIR}/band_rsa.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {OUT_DIR}/band_rsa.json")

    # Plot
    plot_results(all_results, OUT_DIR)


if __name__ == "__main__":
    main()
