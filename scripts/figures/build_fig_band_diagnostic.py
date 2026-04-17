"""Build fig_band_diagnostic: cross-FM band-stop ablation on EEGMAT vs Stress.

EEGMAT shows cross-model convergence on alpha (real neural signature).
Stress shows divergence across FMs (no shared neural target for DASS labels).

Data: results/studies/exp14_channel_importance/band_stop_ablation.json
Output: paper/figures/main/fig_band_diagnostic.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_JSON = ROOT / "results/studies/exp14_channel_importance/band_stop_ablation.json"
OUT_DIR = ROOT / "paper/figures/main"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS = ["delta", "theta", "alpha", "beta"]
BAND_LABELS = [r"$\delta$", r"$\theta$", r"$\alpha$", r"$\beta$"]
FMS = ["labram", "cbramod", "reve"]
FM_LABELS = ["LaBraM", "CBraMod", "REVE"]
# Project palette
FM_COLORS = {
    "labram": "#1f77b4",   # blue
    "cbramod": "#ff7f0e",  # orange
    "reve": "#2ca02c",     # green
}


def load_means(data: dict, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (3 FMs x 4 bands) means and stds."""
    means = np.zeros((len(FMS), len(BANDS)))
    stds = np.zeros_like(means)
    for i, fm in enumerate(FMS):
        for j, band in enumerate(BANDS):
            entry = data[dataset][fm][band]
            means[i, j] = entry["mean_distance"]
            stds[i, j] = entry["std_distance"]
    return means, stds


def draw_panel(ax, means, stds, *, title, ymax, highlight_band=None, divergence=False):
    n_fms, n_bands = means.shape
    x = np.arange(n_bands)
    width = 0.26

    # Highlight rectangle (draw under bars)
    if highlight_band is not None:
        bj = BANDS.index(highlight_band)
        ax.axvspan(bj - 0.5, bj + 0.5, color="#FFD966", alpha=0.25, zorder=0)

    for i, fm in enumerate(FMS):
        offset = (i - 1) * width
        ax.bar(
            x + offset,
            means[i],
            width,
            yerr=stds[i] / np.sqrt(max(1, 1)),  # std shown directly
            color=FM_COLORS[fm],
            edgecolor="black",
            linewidth=0.6,
            label=FM_LABELS[i],
            error_kw=dict(lw=0.8, capsize=2, capthick=0.6, ecolor="#333"),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(BAND_LABELS, fontsize=11)
    ax.set_xlabel("Frequency band", fontsize=10)
    ax.set_title(title, fontsize=11, pad=20)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9)

    # Annotation banner above plot
    if divergence:
        banner = "cross-model peak: diverges"
        banner_color = "#B22222"  # dark red
    else:
        banner = f"cross-model peak: {highlight_band}"
        banner_color = "#1a6b1a"  # dark green

    ax.text(
        0.5,
        1.02,
        banner,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=banner_color,
    )

    # Per-FM peak-band arrows on divergence panel
    if divergence:
        # Annotate each FM's max band with a small marker above its tallest bar
        for i, fm in enumerate(FMS):
            peak_j = int(np.argmax(means[i]))
            peak_val = means[i, peak_j]
            peak_std = stds[i, peak_j]
            xpos = peak_j + (i - 1) * width
            ypos = peak_val + peak_std + 0.010
            ax.annotate(
                "",
                xy=(xpos, peak_val + peak_std + 0.002),
                xytext=(xpos, ypos + 0.012),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=FM_COLORS[fm],
                    lw=1.2,
                    mutation_scale=8,
                ),
            )
            ax.text(
                xpos,
                ypos + 0.016,
                FM_LABELS[i],
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=FM_COLORS[fm],
                fontweight="bold",
            )


def main():
    with open(DATA_JSON) as f:
        data = json.load(f)

    eeg_m, eeg_s = load_means(data, "eegmat")
    str_m, str_s = load_means(data, "stress")

    ymax = max(
        (eeg_m + eeg_s).max(),
        (str_m + str_s).max(),
    ) * 1.35  # headroom for annotations

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.0, 4.0), sharey=True)

    draw_panel(
        ax_l,
        eeg_m,
        eeg_s,
        title="EEGMAT (rest vs arithmetic)",
        ymax=ymax,
        highlight_band="alpha",
        divergence=False,
    )
    draw_panel(
        ax_r,
        str_m,
        str_s,
        title="Stress (DASS state)",
        ymax=ymax,
        highlight_band=None,
        divergence=True,
    )

    ax_l.set_ylabel("Causal band importance\n(cosine distance)", fontsize=10)

    # Shared legend
    handles = [
        mpatches.Patch(facecolor=FM_COLORS[fm], edgecolor="black", label=FM_LABELS[i])
        for i, fm in enumerate(FMS)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    pdf_path = OUT_DIR / "fig_band_diagnostic.pdf"
    png_path = OUT_DIR / "fig_band_diagnostic.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    # Print summary table
    print("\nEEGMAT means (rows: FMs, cols: δ θ α β):")
    print(np.array_str(eeg_m, precision=4, suppress_small=True))
    print("Stress means:")
    print(np.array_str(str_m, precision=4, suppress_small=True))


if __name__ == "__main__":
    main()
