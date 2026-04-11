"""Build the core erosion-evidence figure: Frozen LP vs FT BA across datasets.

Two-panel figure:
  Panel A: Grouped bar chart of Frozen LP BA vs FT BA per dataset.
           Error bars show multi-seed std where available.
  Panel B: Representation-level pooled label fraction (ω²_label)
           Frozen vs FT per dataset (Stress/ADFTD/TDBRAIN).
           EEGMAT omitted (within-subject design uses crossed decomposition).

Run from project root:
    /raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python \
        scripts/build_erosion_figure.py

Outputs:
    paper/figures/erosion_evidence.pdf
    paper/figures/erosion_evidence.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------

# Behavioral-level: Frozen LP BA vs FT BA (multi-seed where available)
BEHAVIORAL = {
    "Stress": {
        "frozen_mean": 0.6049, "frozen_std": 0.0299, "frozen_n": 8,
        "ft_mean": 0.4434, "ft_std": 0.0677, "ft_n": 3,
        "mode": "erosion",
    },
    "ADFTD": {
        "frozen_mean": 0.6694, "frozen_std": 0.0248, "frozen_n": 3,
        "ft_mean": 0.7521, "ft_std": 0.0, "ft_n": 1,
        "mode": "injection",
    },
    "TDBRAIN": {
        "frozen_mean": 0.6794, "frozen_std": 0.0073, "frozen_n": 3,
        "ft_mean": 0.6812, "ft_std": 0.0, "ft_n": 1,
        "mode": "silent erosion",
    },
    "EEGMAT": {
        "frozen_mean": 0.6713, "frozen_std": 0.0458, "frozen_n": 3,
        "ft_mean": 0.7361, "ft_std": 0.0, "ft_n": 1,
        "mode": "mild injection",
    },
}

# Try to load from analysis JSON if available
EROSION_JSON = Path("results/studies/2026-04-10_stress_erosion/analysis.json")
VA_JSON = Path("paper/figures/variance_analysis.json")


def load_representation_data():
    """Load ω²_label from variance_analysis.json for Stress/ADFTD/TDBRAIN."""
    if not VA_JSON.is_file():
        return None
    va = json.load(open(VA_JSON))
    rep = {}
    for name in ["Stress", "ADFTD", "TDBRAIN"]:
        ds = va.get("datasets", {}).get(name)
        if ds is None:
            continue
        fz = ds["analysis"]["frozen"]["nested_omega2"]["label"]
        ft = ds["analysis"]["ft_pooled"]["nested_omega2"]["label"]
        rep[name] = {"frozen": fz, "ft": ft}
    return rep


def build_figure():
    rep_data = load_representation_data()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 2]})

    # --- Panel A: Behavioral BA ---
    ax = axes[0]
    datasets = list(BEHAVIORAL.keys())
    x = np.arange(len(datasets))
    w = 0.35

    frozen_means = [BEHAVIORAL[d]["frozen_mean"] for d in datasets]
    frozen_errs = [BEHAVIORAL[d]["frozen_std"] for d in datasets]
    ft_means = [BEHAVIORAL[d]["ft_mean"] for d in datasets]
    ft_errs = [BEHAVIORAL[d]["ft_std"] for d in datasets]

    bars_fz = ax.bar(x - w/2, frozen_means, w, yerr=frozen_errs, label="Frozen LP",
                     color="#4a90d9", capsize=4, edgecolor="black", linewidth=0.5)
    bars_ft = ax.bar(x + w/2, ft_means, w, yerr=ft_errs, label="Fine-tuned",
                     color="#e74c3c", capsize=4, edgecolor="black", linewidth=0.5)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="Chance")
    ax.set_ylabel("Subject-level Balanced Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0.3, 0.85)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("A. Behavioral: Frozen LP vs Fine-Tuned BA")

    # Annotate deltas
    for i, d in enumerate(datasets):
        delta = BEHAVIORAL[d]["ft_mean"] - BEHAVIORAL[d]["frozen_mean"]
        ypos = max(BEHAVIORAL[d]["frozen_mean"], BEHAVIORAL[d]["ft_mean"]) + \
               max(BEHAVIORAL[d]["frozen_std"], BEHAVIORAL[d]["ft_std"]) + 0.02
        sign = "+" if delta > 0 else ""
        color = "#2ecc71" if delta > 0 else "#e74c3c"
        ax.text(i, ypos, f"{sign}{delta:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=color)

    # --- Panel B: Representation ω²_label ---
    ax2 = axes[1]
    if rep_data:
        rep_datasets = [d for d in ["Stress", "ADFTD", "TDBRAIN"] if d in rep_data]
        x2 = np.arange(len(rep_datasets))

        fz_vals = [rep_data[d]["frozen"] * 100 for d in rep_datasets]
        ft_vals = [rep_data[d]["ft"] * 100 for d in rep_datasets]

        ax2.bar(x2 - w/2, fz_vals, w, label="Frozen", color="#4a90d9",
                edgecolor="black", linewidth=0.5)
        ax2.bar(x2 + w/2, ft_vals, w, label="Fine-tuned", color="#e74c3c",
                edgecolor="black", linewidth=0.5)

        ax2.set_ylabel(r"Pooled label fraction $\omega^2_{\mathrm{label}}$ (%)")
        ax2.set_xticks(x2)
        ax2.set_xticklabels(rep_datasets)
        ax2.legend(loc="upper left", fontsize=9)
        ax2.set_title(r"B. Representation: $\omega^2_{\mathrm{label}}$")

        for i, d in enumerate(rep_datasets):
            delta = rep_data[d]["ft"] - rep_data[d]["frozen"]
            ypos = max(fz_vals[i], ft_vals[i]) + 0.3
            sign = "+" if delta > 0 else ""
            color = "#2ecc71" if delta > 0 else "#e74c3c"
            ax2.text(i, ypos, f"{sign}{delta*100:.2f}pp", ha="center", va="bottom",
                     fontsize=8, fontweight="bold", color=color)
    else:
        ax2.text(0.5, 0.5, "variance_analysis.json\nnot found",
                 ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()

    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "erosion_evidence.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "erosion_evidence.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"→ {out_dir / 'erosion_evidence.pdf'}")
    print(f"→ {out_dir / 'erosion_evidence.png'}")


if __name__ == "__main__":
    build_figure()
