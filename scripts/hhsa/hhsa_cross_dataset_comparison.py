"""Cross-dataset HHSA comparison — 4 datasets.

Generates side-by-side comparisons of:
1. Grand mean holospectra
2. Condition contrast t-maps
3. AM coherence distributions
4. Summary statistics table

Usage:
    python scripts/hhsa_cross_dataset_comparison.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pipeline.hhsa as hhsa

BASE = "results/hhsa"
OUT_DIR = f"{BASE}/cross_dataset_comparison"

DATASETS = [
    {
        "name": "eegmat",
        "label": "EEGMAT",
        "contrast": "rest/task",
        "color": "#2196F3",
    },
    {
        "name": "stress",
        "label": "Stress",
        "contrast": "normal/increase",
        "color": "#FF5722",
    },
    {
        "name": "meditation",
        "label": "Meditation",
        "contrast": "ses1/ses2",
        "color": "#4CAF50",
    },
    {
        "name": "sleep_deprivation",
        "label": "Sleep Dep",
        "contrast": "normal/deprived",
        "color": "#9C27B0",
    },
]


def load_grand_mean(ds_name):
    path = f"{BASE}/07_full_holospectrum/{ds_name}/grand_mean.npz"
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d["grand_mean"], d["fc"], d["fa"]


def load_condition_contrast(ds_name):
    path = f"{BASE}/07_full_holospectrum/{ds_name}/condition_contrast.npz"
    if not os.path.exists(path):
        return None
    return dict(np.load(path))


def load_am_coherence(ds_name):
    path = f"{BASE}/08_am_coherence/{ds_name}/am_coherence.npz"
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d["coherence"], d["rec_ids"]


def plot_grand_means():
    """Side-by-side grand mean holospectra."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Grand Mean Holospectra — Cross-Dataset Comparison", fontsize=14, y=1.02)

    vmin, vmax = None, None
    data_list = []
    for ds in DATASETS:
        result = load_grand_mean(ds["name"])
        if result is None:
            data_list.append(None)
            continue
        gm, fc, fa = result
        log_gm = np.log10(gm.T + 1e-12)
        data_list.append((log_gm, fc, fa))
        if vmin is None:
            vmin, vmax = log_gm.min(), log_gm.max()
        else:
            vmin = min(vmin, log_gm.min())
            vmax = max(vmax, log_gm.max())

    for i, (ds, data) in enumerate(zip(DATASETS, data_list)):
        ax = axes[i]
        if data is None:
            ax.set_title(f"{ds['label']} — N/A")
            ax.axis("off")
            continue
        log_gm, fc, fa = data
        im = ax.pcolormesh(fc, fa, log_gm, cmap="hot", shading="auto",
                           vmin=vmin, vmax=vmax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Carrier freq (Hz)")
        if i == 0:
            ax.set_ylabel("AM freq (Hz)")
        ax.set_title(f"{ds['label']} ({ds['contrast']})")
        diag = np.linspace(fa.min(), min(fc.max(), fa.max()), 100)
        ax.plot(diag, diag, "w--", lw=0.5, alpha=0.5)

    fig.colorbar(im, ax=axes, label="log₁₀ power", shrink=0.8)
    fig.savefig(os.path.join(OUT_DIR, "grand_mean_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved grand_mean_comparison.png")


def plot_condition_contrasts():
    """Side-by-side condition contrast t-maps."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Condition Contrast t-maps — Cross-Dataset", fontsize=14, y=1.02)

    contrast_labels = {
        "eegmat": ("Task vs Rest", "Rest", "Task"),
        "stress": ("Stress-increase vs Normal", "Normal", "Increase"),
        "meditation": ("Session 2 vs Session 1", "Session 1", "Session 2"),
        "sleep_deprivation": ("Sleep Deprived vs Normal", "Normal Sleep", "Sleep Deprived"),
    }

    for i, ds in enumerate(DATASETS):
        ax = axes[i]
        data = load_condition_contrast(ds["name"])
        if data is None:
            ax.set_title(f"{ds['label']} — N/A")
            ax.axis("off")
            continue
        t_map = data["t_map"]
        fc, fa = data["fc"], data["fa"]
        n0, n1 = int(data["n0"]), int(data["n1"])
        title, _, _ = contrast_labels[ds["name"]]

        vmax = np.percentile(np.abs(t_map), 95)
        im = ax.pcolormesh(fc, fa, t_map.T, cmap="RdBu_r", shading="auto",
                           vmin=-vmax, vmax=vmax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Carrier freq (Hz)")
        if i == 0:
            ax.set_ylabel("AM freq (Hz)")
        ax.set_title(f"{title}\n(n={n0} vs {n1})")
        diag = np.linspace(fa.min(), min(fc.max(), fa.max()), 100)
        ax.plot(diag, diag, "gray", ls="--", lw=0.5, alpha=0.5)
        plt.colorbar(im, ax=ax, label="t-stat")

    fig.savefig(os.path.join(OUT_DIR, "condition_contrast_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved condition_contrast_comparison.png")


def plot_am_coherence():
    """AM coherence distribution comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("AM Coherence — Cross-Dataset Comparison", fontsize=14)

    positions = []
    for i, ds in enumerate(DATASETS):
        result = load_am_coherence(ds["name"])
        if result is None:
            continue
        coh, _ = result
        pos = i
        positions.append(pos)
        bp = ax.boxplot(coh, positions=[pos], widths=0.5,
                        boxprops=dict(color=ds["color"]),
                        medianprops=dict(color=ds["color"]),
                        whiskerprops=dict(color=ds["color"]),
                        capprops=dict(color=ds["color"]),
                        flierprops=dict(markeredgecolor=ds["color"]))
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(coh))
        ax.scatter(np.full(len(coh), pos) + jitter, coh,
                   alpha=0.5, color=ds["color"], s=20, zorder=3)

    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([ds["label"] for ds in DATASETS])
    ax.set_ylabel("Inter-window holospectrum correlation")
    fig.savefig(os.path.join(OUT_DIR, "am_coherence_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved am_coherence_comparison.png")


def summary_stats():
    """Print and save summary statistics table."""
    print(f"\n{'='*80}")
    print("Cross-Dataset HHSA Summary")
    print(f"{'='*80}")
    print(f"{'Dataset':<18} {'n_rec':>6} {'max|t|':>8} {'mean|t|':>9} "
          f"{'Bonf sig':>9} {'Coherence':>10} {'Coh SD':>8}")
    print("-" * 80)

    rows = []
    for ds in DATASETS:
        contrast = load_condition_contrast(ds["name"])
        coh_result = load_am_coherence(ds["name"])
        gm = load_grand_mean(ds["name"])

        n_rec = 0
        max_t = mean_t = bonf_sig = 0
        coh_mean = coh_sd = 0.0

        if contrast is not None:
            t_map = contrast["t_map"]
            n_rec = int(contrast["n0"]) + int(contrast["n1"])
            max_t = float(np.nanmax(np.abs(t_map)))
            mean_t = float(np.nanmean(np.abs(t_map)))
            n_bins = t_map.size
            # Bonferroni: two-sided t-test, df ~ n0+n1-2
            df = int(contrast["n0"]) + int(contrast["n1"]) - 2
            p_vals = 2 * (1 - stats.t.cdf(np.abs(t_map), df=df))
            bonf_sig = int(np.sum(p_vals < 0.05 / n_bins))

        if coh_result is not None:
            coh, _ = coh_result
            coh_mean = float(np.mean(coh))
            coh_sd = float(np.std(coh))

        row = {
            "dataset": ds["label"],
            "contrast": ds["contrast"],
            "n_rec": n_rec,
            "max_t": max_t,
            "mean_t": mean_t,
            "bonf_sig": bonf_sig,
            "coh_mean": coh_mean,
            "coh_sd": coh_sd,
        }
        rows.append(row)
        print(f"{ds['label']+' ('+ds['contrast']+')':.<18} {n_rec:>6} {max_t:>8.2f} "
              f"{mean_t:>9.2f} {bonf_sig:>9} {coh_mean:>10.3f} {coh_sd:>8.3f}")

    print("-" * 80)

    # Save as npz
    np.savez_compressed(
        os.path.join(OUT_DIR, "summary_stats.npz"),
        datasets=np.array([r["dataset"] for r in rows]),
        contrasts=np.array([r["contrast"] for r in rows]),
        n_rec=np.array([r["n_rec"] for r in rows]),
        max_t=np.array([r["max_t"] for r in rows]),
        mean_t=np.array([r["mean_t"] for r in rows]),
        bonf_sig=np.array([r["bonf_sig"] for r in rows]),
        coh_mean=np.array([r["coh_mean"] for r in rows]),
        coh_sd=np.array([r["coh_sd"] for r in rows]),
    )
    print(f"  Saved summary_stats.npz")
    return rows


def plot_summary_bar(rows):
    """Bar chart summary of key metrics across datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Cross-Dataset HHSA Metrics", fontsize=14, y=1.02)

    labels = [r["dataset"] for r in rows]
    colors = [ds["color"] for ds in DATASETS]

    # Panel 1: max |t|
    ax = axes[0]
    vals = [r["max_t"] for r in rows]
    ax.bar(range(len(rows)), vals, color=colors, alpha=0.8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("max |t-statistic|")
    ax.set_title("Peak Condition Contrast")

    # Panel 2: mean |t|
    ax = axes[1]
    vals = [r["mean_t"] for r in rows]
    ax.bar(range(len(rows)), vals, color=colors, alpha=0.8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("mean |t-statistic|")
    ax.set_title("Average Condition Contrast")

    # Panel 3: AM coherence
    ax = axes[2]
    vals = [r["coh_mean"] for r in rows]
    errs = [r["coh_sd"] for r in rows]
    ax.bar(range(len(rows)), vals, color=colors, alpha=0.8, yerr=errs, capsize=4)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean coherence")
    ax.set_title("AM Coherence (inter-window)")
    ax.set_ylim(0.85, 1.0)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "summary_bar_chart.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved summary_bar_chart.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Cross-Dataset HHSA Comparison (4 datasets)")
    print(f"Output: {OUT_DIR}\n")

    plot_grand_means()
    plot_condition_contrasts()
    plot_am_coherence()
    rows = summary_stats()
    plot_summary_bar(rows)

    print(f"\nDone. All figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
