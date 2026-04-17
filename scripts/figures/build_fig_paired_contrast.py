"""Build the core paired-contrast figure for the EEG FM paper.

Left panel : EEGMAT rest vs arithmetic (within-subject, strong alpha contrast).
             LaBraM / CBraMod / REVE FT (3-seed subject-level BA, bars = mean ±
             s.d., dots = per-seed BAs).
Right panel: Stress longitudinal DSS (within-subject, no known neural contrast).
             Three classifiers (Centroid, 1-NN, Linear) × three FMs clustered
             by FM.  Data from findings.md F-D.2 / exp11_longitudinal_dss.

Common y-axis (Balanced Accuracy, 0.0 – 0.8) and a dashed chance line make the
vertical comparison immediate.  The annotations tell the thesis: strong contrast
-> FT succeeds; no contrast -> FT fails.

Usage
-----
/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python \
    scripts/build_fig_paired_contrast.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
EXP04 = REPO / "results/studies/exp04_eegmat_feat_multiseed"
EXP17 = REPO / "results/studies/exp17_eegmat_cbramod_reve_ft"
OUT_DIR = REPO / "paper/figures/main"
OUT_STEM = "fig_paired_contrast"

# ---------------------------------------------------------------------------
# Colours: one per FM, consistent across panels (colour-blind safe)
# ---------------------------------------------------------------------------
CB = sns.color_palette("colorblind")
FM_COLORS = {
    "LaBraM": CB[0],   # blue
    "CBraMod": CB[1],  # orange
    "REVE": CB[2],     # green
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_seed_bas(*paths: Path) -> list[float]:
    vals: list[float] = []
    for p in paths:
        with open(p / "summary.json") as f:
            vals.append(float(json.load(f)["subject_bal_acc"]))
    return vals


def load_eegmat() -> dict[str, list[float]]:
    """Per-seed LOO-subject BA for each FM (FT, 3 seeds)."""
    return {
        "LaBraM": _load_seed_bas(
            EXP04 / "s42_llrd1.0",
            EXP04 / "s123_llrd1.0",
            EXP04 / "s2024_llrd1.0",
        ),
        "CBraMod": _load_seed_bas(
            EXP17 / "cbramod_s42",
            EXP17 / "cbramod_s123",
            EXP17 / "cbramod_s2024",
        ),
        "REVE": _load_seed_bas(
            EXP17 / "reve_s42",
            EXP17 / "reve_s123",
            EXP17 / "reve_s2024",
        ),
    }


# Stress longitudinal DSS numbers (findings.md F-D.2, n=54 recordings, 13 subjects)
STRESS_LONG: dict[str, dict[str, float]] = {
    "LaBraM":  {"Centroid": 0.296, "1-NN": 0.296, "Linear": 0.000},
    "CBraMod": {"Centroid": 0.241, "1-NN": 0.167, "Linear": 0.000},
    "REVE":    {"Centroid": 0.333, "1-NN": 0.426, "Linear": 0.000},
}
CLF_ORDER = ["Centroid", "1-NN", "Linear"]
FM_ORDER = ["LaBraM", "CBraMod", "REVE"]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_left(ax, data: dict[str, list[float]]) -> None:
    x = np.arange(len(FM_ORDER))
    means = [float(np.mean(data[k])) for k in FM_ORDER]
    sds = [float(np.std(data[k], ddof=1)) for k in FM_ORDER]

    for i, fm in enumerate(FM_ORDER):
        ax.bar(
            x[i], means[i], yerr=sds[i],
            color=FM_COLORS[fm], edgecolor="black", linewidth=0.8,
            width=0.62, capsize=5, alpha=0.9,
            error_kw=dict(elinewidth=1.2, ecolor="black"),
        )
        # Per-seed dots for honest uncertainty display
        jitter = np.linspace(-0.09, 0.09, len(data[fm]))
        ax.scatter(
            x[i] + jitter, data[fm],
            color="white", edgecolor="black", s=22, zorder=5, linewidth=0.8,
        )

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.text(
        -0.42, 0.505, "chance",
        color="grey", fontsize=8, ha="left", va="bottom",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(FM_ORDER)
    ax.set_ylabel("Balanced accuracy  (subject LOO)")
    ax.set_ylim(0.0, 0.8)
    ax.set_title("EEGMAT (rest vs arithmetic task)", fontsize=11, pad=10)

    # Thesis annotation — arrow points UP into the bar cluster; placed in the
    # gap between bars so it does not collide with error whiskers.
    ax.annotate(
        "contrast:\nalpha desynchronization",
        xy=(0.5, 0.70), xytext=(0.5, 0.74),
        xycoords="data", textcoords="data",
        ha="center", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="#e8f3e8",
                  ec="#2a7a2a", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="#2a7a2a", lw=1.1),
    )


def draw_right(ax) -> None:
    n_clf = len(CLF_ORDER)
    n_fm = len(FM_ORDER)
    group_width = 0.78
    bar_w = group_width / n_clf
    x = np.arange(n_fm)

    # Hatching distinguishes the three classifiers within an FM cluster;
    # FM colour is preserved for visual linkage to the left panel.
    hatches = ["", "///", "xxx"]

    for j, clf in enumerate(CLF_ORDER):
        for i, fm in enumerate(FM_ORDER):
            val = STRESS_LONG[fm][clf]
            xi = x[i] - group_width / 2 + (j + 0.5) * bar_w
            ax.bar(
                xi, val, width=bar_w * 0.95,
                color=FM_COLORS[fm], edgecolor="black", linewidth=0.8,
                hatch=hatches[j], alpha=0.9,
            )
            # Linear classifier is 0.0 BA (perfectly anti-correlated predictions
            # given class-balance LOO); mark this with an explicit "0.00" label
            # and a short marker so the reader does not mistake it for missing
            # data.
            if val < 0.01:
                ax.plot(
                    [xi - bar_w * 0.4, xi + bar_w * 0.4], [0.005, 0.005],
                    color="black", linewidth=1.2, solid_capstyle="butt",
                )
                ax.text(
                    xi, 0.025, "0.00",
                    ha="center", va="bottom", fontsize=6.5, color="#8a2a2a",
                    rotation=90,
                )

    # One legend entry per classifier (hatch only, neutral fill)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                      hatch=hatches[j], label=clf)
        for j, clf in enumerate(CLF_ORDER)
    ]
    ax.legend(
        handles=legend_handles, loc="upper left",
        fontsize=8, frameon=False, handlelength=1.6, handleheight=1.1,
        title="classifier", title_fontsize=8,
        bbox_to_anchor=(0.0, 1.0),
    )

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.text(
        -0.42, 0.505, "chance",
        color="grey", fontsize=8, ha="left", va="bottom",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(FM_ORDER)
    ax.set_ylim(0.0, 0.8)
    ax.set_title("Stress (within-subject DSS trajectory)", fontsize=11, pad=10)

    # Thesis annotation — arrow points DOWN at the tallest observed failure
    # bar (REVE 1-NN = 0.43) so the reader sees "the best of the worst still
    # fails to clear chance".
    ax.annotate(
        "contrast:\nnone published",
        xy=(2.0, 0.43), xytext=(1.4, 0.74),
        xycoords="data", textcoords="data",
        ha="center", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="#f7e6e6",
                  ec="#8a2a2a", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="#8a2a2a", lw=1.1),
    )


def build_figure() -> Path:
    sns.set_style("white")
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    eegmat = load_eegmat()

    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 4.0), sharey=True,
        gridspec_kw=dict(wspace=0.12, left=0.09, right=0.98,
                         top=0.82, bottom=0.13),
    )
    draw_left(axes[0], eegmat)
    draw_right(axes[1])

    fig.suptitle(
        "Paired within-subject contrast experiment",
        fontsize=12.5, y=0.965, fontweight="semibold",
    )
    # Small caption tying the two panels to the thesis
    fig.text(
        0.5, 0.02,
        "Same FM family, same within-subject framework (LOO), "
        "opposite outcomes  \u2013  contrast strength governs FM rescue.",
        ha="center", fontsize=8.5, color="#444444", style="italic",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{OUT_STEM}.pdf"
    png = OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Echo the numbers used (makes the script self-documenting in logs)
    print("EEGMAT per-seed BAs:")
    for fm in FM_ORDER:
        vals = eegmat[fm]
        print(f"  {fm:8s} = {np.mean(vals):.3f} +/- {np.std(vals, ddof=1):.3f}"
              f"   seeds: {['%.3f' % v for v in vals]}")
    print("Stress longitudinal (findings.md F-D.2):")
    for fm in FM_ORDER:
        print(f"  {fm:8s} = {STRESS_LONG[fm]}")
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    return pdf


if __name__ == "__main__":
    build_figure()
