"""Generate the cross-dataset signal-strength figure for paper §4.3.

Reads precomputed variance-decomposition results from
`paper/figures/variance_analysis.json` (produced by
`scripts/run_variance_analysis.py` under stats_env) and produces a 2-panel
figure: (a) recording-level BA per dataset; (b) per-fold ω²_subject|label
ratio for both frozen and fine-tuned features, with error bars from per-fold
spread.

This script no longer recomputes any statistics — it is figure rendering
only. All math lives in `src/variance_analysis.py`.

Run from project root:
    conda run -n timm_eeg python scripts/build_cross_dataset_figure.py

If `variance_analysis.json` is missing, run:
    conda run -n stats_env python scripts/run_variance_analysis.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

JSON_PATH = "paper/figures/variance_analysis.json"
OUT_PDF = "paper/figures/cross_dataset_signal_strength.pdf"
OUT_PNG = "paper/figures/cross_dataset_signal_strength.png"
DATASET_ORDER = ["Stress", "TDBRAIN", "ADFTD"]


def main():
    if not os.path.isfile(JSON_PATH):
        sys.exit(
            f"ERROR: {JSON_PATH} not found.\n"
            f"Generate it first:\n"
            f"  conda run -n stats_env python scripts/run_variance_analysis.py"
        )

    with open(JSON_PATH) as f:
        data = json.load(f)

    rows = []
    for name in DATASET_ORDER:
        if name not in data["datasets"]:
            print(f"  [skip] {name} not in JSON")
            continue
        d = data["datasets"][name]
        analysis = d["analysis"]
        fz = analysis["frozen"]
        ft_pooled = analysis["ft_pooled"]

        # Pooled label fraction: sum(SS_label) / sum(SS_total) over dims.
        # This is the reviewer-defensible primary metric (see §10 of
        # docs/eta_squared_pipeline_explanation.md). Per-fold ω² is
        # degenerate for Stress (1 subj per positive class per fold).
        fz_frac = fz["pooled_fractions"]["label"]
        ft_frac = ft_pooled["pooled_fractions"]["label"]

        rows.append({
            "dataset": name,
            "ba": d["ba"],
            "n_subj": fz["n_subjects"],
            "n_rec": fz["n_recordings"],
            "frozen_label_frac": fz_frac,
            "ft_pooled_label_frac": ft_frac,
            "nested_identifiable_frozen": fz.get("nested_identifiable", True),
            "nested_identifiable_ft_pooled": ft_pooled.get("nested_identifiable", True),
        })

    print("Loaded analysis for:", [r["dataset"] for r in rows])
    for r in rows:
        direction = "↑" if r["ft_pooled_label_frac"] > 1.05 * r["frozen_label_frac"] else (
            "↓" if r["ft_pooled_label_frac"] < 0.95 * r["frozen_label_frac"] else "→"
        )
        print(
            f"  {r['dataset']:8s} | BA={r['ba']:.3f} | "
            f"label SS/total: {r['frozen_label_frac']*100:5.2f}% → "
            f"{r['ft_pooled_label_frac']*100:5.2f}%  {direction}"
        )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 120,
    })

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    names = [r["dataset"] for r in rows]
    colors = ["#5B9BD5", "#ED7D31", "#70AD47"]
    x = np.arange(len(names))

    # Panel A: BA bars
    ax = axes[0]
    bas = [r["ba"] for r in rows]
    bars = ax.bar(x, bas, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    for b, v in zip(bars, bas):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Subject-level Balanced Accuracy")
    ax.set_ylim(0.4, 0.85)
    ax.set_title("(a) Fine-tuned LaBraM, subject-level CV")
    ax.legend(loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: pooled label fraction (primary reviewer-ready metric).
    # Replaces the earlier per-fold ω² ratio panel, which was degenerate
    # for Stress (1 subject per positive class per fold).
    ax = axes[1]
    width = 0.35
    fz_fracs = [r["frozen_label_frac"] * 100 for r in rows]
    ft_fracs = [r["ft_pooled_label_frac"] * 100 for r in rows]

    b1 = ax.bar(x - width / 2, fz_fracs, width,
                label="Frozen", color="#A6A6A6", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + width / 2, ft_fracs, width,
                label="Fine-tuned (pooled)", color="#404040",
                edgecolor="black", linewidth=0.6)
    for bars_ in (b1, b2):
        for b in bars_:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.2,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    # Annotate fold-change per dataset
    for i, r in enumerate(rows):
        if r["frozen_label_frac"] > 0:
            mult = r["ft_pooled_label_frac"] / r["frozen_label_frac"]
            sym = "↑" if mult > 1.05 else ("↓" if mult < 0.95 else "→")
            color = {"↑": "#388E3C", "↓": "#D32F2F", "→": "#555555"}[sym]
            ax.text(i, max(fz_fracs[i], ft_fracs[i]) + 1.2,
                    f"{sym} {mult:.2f}×", ha="center", fontsize=9,
                    color=color, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(r"$SS_{\mathrm{label}}\ /\ SS_{\mathrm{total}}$  (%)")
    ax.set_ylim(0, max(max(fz_fracs), max(ft_fracs)) * 1.6)
    ax.set_title("(b) Pooled label variance fraction")
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Cross-dataset signal strength: LaBraM on resting-state EEG",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
    print(f"Saved → {OUT_PDF}")
    print(f"Saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
