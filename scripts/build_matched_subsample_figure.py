"""Matched subsample curves (exp09): label fraction vs N for 3 FMs × 2 datasets.

Shows that FT mode (injection/erosion/neutral) is N-invariant.
6 subplots: 3 models × 2 datasets (ADFTD, TDBRAIN).
Each subplot: frozen vs FT label fraction curves across N rungs,
with permutation null as reference.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 10, "font.family": "sans-serif"})

OUT_DIR = "results/studies/exp09_multimodel_matched"
data = json.load(open(f"{OUT_DIR}/matched_subsample_multimodel.json"))

models = ["labram", "cbramod", "reve"]
model_labels = ["LaBraM", "CBraMod", "REVE"]
datasets = ["adftd", "tdbrain"]
dataset_labels = ["ADFTD", "TDBRAIN"]
model_colors = {"labram": "#4C72B0", "cbramod": "#DD8452", "reve": "#55A868"}

# Stress reference (N=17, from exp06)
stress_ref = {
    "labram": 2.917,
    "cbramod": 1.140,
    "reve": 2.163,
}

fig, axes = plt.subplots(len(models), len(datasets), figsize=(12, 10),
                         gridspec_kw={"hspace": 0.35, "wspace": 0.3})

for i, (model, mlabel) in enumerate(zip(models, model_labels)):
    for j, (ds, dlabel) in enumerate(zip(datasets, dataset_labels)):
        ax = axes[i, j]
        key = f"{model}_{ds}"

        if key not in data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        entry = data[key]
        rungs = entry["rungs"]
        full_n = entry["full_n"]

        # Sort by N
        ns = sorted(rungs.keys(), key=lambda k: int(k[1:]))
        x_vals = [int(k[1:]) for k in ns]

        # Observed
        frz_means = [rungs[k]["observed"]["frozen_mean"] * 100 for k in ns]
        frz_stds = [rungs[k]["observed"]["frozen_std"] * 100 for k in ns]
        ft_means = [rungs[k]["observed"]["ft_mean"] * 100 for k in ns]
        ft_stds = [rungs[k]["observed"]["ft_std"] * 100 for k in ns]

        # Null
        null_frz = [rungs[k]["null"]["frozen_mean"] * 100 for k in ns]
        null_ft = [rungs[k]["null"]["ft_mean"] * 100 for k in ns]

        # Plot curves
        ax.fill_between(x_vals,
                        [m - s for m, s in zip(frz_means, frz_stds)],
                        [m + s for m, s in zip(frz_means, frz_stds)],
                        alpha=0.15, color="#1f77b4")
        ax.plot(x_vals, frz_means, "o-", color="#1f77b4", linewidth=2,
                markersize=5, label="Frozen", zorder=3)

        ax.fill_between(x_vals,
                        [m - s for m, s in zip(ft_means, ft_stds)],
                        [m + s for m, s in zip(ft_means, ft_stds)],
                        alpha=0.15, color="#d62728")
        ax.plot(x_vals, ft_means, "s-", color="#d62728", linewidth=2,
                markersize=5, label="FT", zorder=3)

        # Null reference
        ax.plot(x_vals, null_frz, "--", color="#1f77b4", alpha=0.4,
                linewidth=1, label="Null frozen")
        ax.plot(x_vals, null_ft, "--", color="#d62728", alpha=0.4,
                linewidth=1, label="Null FT")

        # Full-N markers
        ax.plot(full_n["n_subj"], full_n["frozen_label_frac"], "^",
                color="#1f77b4", markersize=10, markeredgecolor="black",
                markeredgewidth=0.8, zorder=5)
        ax.plot(full_n["n_subj"], full_n["ft_label_frac"], "v",
                color="#d62728", markersize=10, markeredgecolor="black",
                markeredgewidth=0.8, zorder=5)

        # Stress reference line (N=17)
        ax.axhline(stress_ref[model], color="gray", linestyle=":",
                   linewidth=1, alpha=0.7)
        ax.text(x_vals[-1], stress_ref[model] + 0.15,
                f"Stress={stress_ref[model]:.1f}%", fontsize=7,
                color="gray", ha="right", style="italic")

        # Delta annotation
        delta = full_n["delta"]
        sign = "+" if delta > 0 else ""
        mode = "injection" if delta > 0.5 else ("erosion" if delta < -0.5 else "neutral")
        ax.text(0.97, 0.97, f"Δ={sign}{delta:.2f}pp\n({mode})",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8))

        ax.set_xlabel("N (subjects)")
        ax.set_ylabel("Label fraction (%)")

        if i == 0:
            ax.set_title(dlabel, fontsize=12, fontweight="bold")
        if j == 0:
            ax.text(-0.25, 0.5, mlabel, transform=ax.transAxes,
                    fontsize=12, fontweight="bold", va="center", ha="right",
                    rotation=90, color=model_colors[model])

        if i == 0 and j == 0:
            ax.legend(fontsize=7, loc="upper left", ncol=2)

        ax.set_ylim(bottom=-0.5)

# Panel labels
for idx, (r, c) in enumerate([(i, j) for i in range(len(models))
                               for j in range(len(datasets))]):
    label = chr(65 + idx)
    axes[r, c].text(-0.08, 1.08, label, transform=axes[r, c].transAxes,
                    fontsize=14, fontweight="bold", va="top")

plt.suptitle("Matched Subsample: Label Fraction vs N (3 FMs × 2 Datasets)",
             fontsize=13, fontweight="bold", y=1.01)
plt.savefig(f"{OUT_DIR}/matched_subsample_curves.pdf", bbox_inches="tight", dpi=300)
plt.savefig(f"{OUT_DIR}/matched_subsample_curves.png", bbox_inches="tight", dpi=150)
print(f"Saved → {OUT_DIR}/matched_subsample_curves.{{pdf,png}}")
