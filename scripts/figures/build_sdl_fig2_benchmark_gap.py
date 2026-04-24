"""Fig 2 — Benchmark literature gap (Intro §1.1 motivator).

Visualises Table II: Frozen → FT balanced accuracy across 3 FMs on 12
benchmarks under explicit subject-level CV. Two panels:
    Left : between-subject design (subject ≡ label)
    Right: within-subject design (subject ⊥ label)

Each row (dataset × FM) shows a Frozen→FT arrow. Colour by FM.
Source attribution (literature vs ours) shown by row-label suffix.

Output: paper/figures/main/sdl_critique/fig2_benchmark_gap.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,
})

ROOT = Path("/raid/jupyter-linjimmy1003.md10/UCSD_stress")
OUT_DIR = ROOT / "paper/figures/main/sdl_critique"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FM_COLOR = {
    "LaBraM":  "#4c78a8",
    "CBraMod": "#f58518",
    "REVE":    "#54a24b",
}

# Literature rows: EEG-FM-Bench arXiv 2508.17742v2, Appendix 6
#   Frozen = Table 4 (multi-task, avg-pool head)
#   FT     = Table 3 (multi-task FT, avg-pool head)
#   This pairing is apples-to-apples (both multi-task).
# Column order in both T3/T4: BENDR, BIOT, LaBraM, EEGPT, CBraMod, CSBrain, REVE.
# (dataset_label, source, LaBraM(F,FT), CBraMod(F,FT), REVE(F,FT))
BETWEEN = [
    ("TUAB",       "lit",  (75.87, 79.36), (73.15, 80.49), (63.80, 80.32)),
    ("Siena",      "lit",  (50.00, 71.99), (64.17, 82.75), (68.60, 70.65)),
    ("ADFTD",      "ours", (69.5,  70.9),  (55.8,  53.7),  (69.2,  65.8)),
    ("TDBRAIN",    "ours", (67.9,  66.5),  (56.4,  48.9),  (54.4,  48.8)),
    ("Meditation", "ours", (47.3,  51.5),  (71.0,  68.3),  (53.8,  43.3)),
]

WITHIN = [
    ("HMC",          "lit",  (59.80, 71.63), (51.81, 71.08), (63.80, 71.43)),
    ("PhysioMI",     "lit",  (29.63, 43.19), (26.90, 31.15), (27.37, 30.63)),
    ("BCIC-IV-2a",   "lit",  (28.40, 34.58), (29.17, 35.50), (28.63, 36.89)),
    ("SEED-VII",     "lit",  (23.23, 26.13), (19.43, 26.05), (20.57, 20.76)),
    ("Things-EEG-2", "lit",  (50.00, 50.90), (50.00, 50.70), (50.00, 59.43)),
    ("EEGMAT",       "ours", (67.1,  73.1),  (73.1,  62.0),  (67.1,  72.7)),
    ("SleepDep",     "ours", (50.0,  53.2),  (55.7,  55.6),  (54.4,  54.2)),
]

FM_ORDER = ["LaBraM", "CBraMod", "REVE"]


def draw_panel(ax, rows, title, panel_color):
    """Each dataset gets one row; within the row, 3 FM pairs (F→FT) stacked horizontally."""
    y_labels = []
    y_positions = []
    fm_offset = {"LaBraM": -0.28, "CBraMod": 0.0, "REVE": +0.28}

    n = len(rows)
    for i, row in enumerate(rows):
        ds_label, source, labram, cbramod, reve = row
        y = n - 1 - i   # top-to-bottom listing
        y_positions.append(y)
        suffix = "†" if source == "ours" else ""
        y_labels.append(f"{ds_label}{suffix}")

        for fm, (f, ft) in zip(FM_ORDER, [labram, cbramod, reve]):
            color = FM_COLOR[fm]
            yo = y + fm_offset[fm]
            delta = ft - f
            # Frozen marker (open circle, clearly hollow), FT marker (filled diamond)
            ax.plot(f, yo, marker="o", markersize=10,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=2.0, zorder=4)
            ax.plot(ft, yo, marker="D", markersize=10,
                    markerfacecolor=color, markeredgecolor="black",
                    markeredgewidth=0.8, zorder=5)
            # arrow — thicker and unambiguous direction
            ax.annotate(
                "", xy=(ft, yo), xytext=(f, yo),
                arrowprops=dict(arrowstyle="-|>,head_width=0.45,head_length=0.7",
                                color=color, lw=2.0,
                                shrinkA=7, shrinkB=7),
                zorder=3,
            )
            # ΔBA annotation next to the FT marker
            sign = "+" if delta >= 0 else ""
            ax.text(
                max(f, ft) + 1.2, yo,
                f"{sign}{delta:.1f}",
                fontsize=7.8, color=color, fontweight="bold",
                va="center", ha="left", zorder=6,
            )

    # y_positions and y_labels are appended in the same iteration order (i=0..n-1),
    # so they pair element-wise. set_yticklabels must receive them WITHOUT reversing
    # — reversing caused TUAB's data (at top y=4) to display label "Meditation".
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9.5)
    ax.set_xlim(15, 95)
    ax.set_ylim(-0.5, n - 0.1)
    ax.set_xlabel("Balanced accuracy (%)")
    ax.set_title(title, fontsize=11, fontweight="bold", color=panel_color)
    ax.grid(True, axis="x", alpha=0.3)
    ax.axvline(50, color="#888", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_axisbelow(True)

    # mean FT marker per panel, per FM (placed below x-axis label)
    means = {fm: np.mean([r[2 + i][1] for r in rows]) for i, fm in enumerate(FM_ORDER)}
    mean_str = "Mean FT BA:   " + "   ".join(
        f"$\\bf{{{fm}}}$ {means[fm]:.1f}" for fm in FM_ORDER
    )
    ax.text(0.02, -0.16, mean_str, transform=ax.transAxes,
            fontsize=9, color="#333", va="top", ha="left")


fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.5), sharex=False)
draw_panel(axes[0], BETWEEN, "Between-subject benchmarks  (subject ≡ label)",  "#e45756")
draw_panel(axes[1], WITHIN,  "Within-subject benchmarks   (subject ⊥ label)",  "#4c78a8")

# shared legend
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker="o", linestyle="", markersize=9,
           markerfacecolor="white", markeredgecolor="black",
           markeredgewidth=1.4, label="Frozen (linear probe)"),
    Line2D([0], [0], marker="D", linestyle="", markersize=9,
           markerfacecolor="#444", markeredgecolor="black",
           label="Fine-tuned"),
    *[
        Line2D([0], [0], marker="s", linestyle="", markersize=10,
               markerfacecolor=c, markeredgecolor="black", label=fm)
        for fm, c in FM_COLOR.items()
    ],
    Line2D([0], [0], marker="", linestyle="", label=" "),
    Line2D([0], [0], marker="", linestyle="", label="†  = our exp_30 runs"),
    Line2D([0], [0], marker="", linestyle="", label="others = EEG-FM-Bench T3/T4"),
]
fig.legend(handles=legend_elems, loc="lower center",
           ncol=7, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.02))

# source attribution (placed below legend, at figure bottom)
fig.text(0.5, -0.04,
         "Literature rows: Frozen from EEG-FM-Bench Table 4, FT from Table 3 "
         "(both multi-task, avg-pool head).",
         fontsize=8.4, color="#444", ha="center", fontstyle="italic")

fig.suptitle(
    "Foundation-model Frozen → FT gains across EEG benchmarks with subject-level CV",
    fontsize=12.5, fontweight="bold", y=0.995,
)

plt.tight_layout(rect=(0, 0.05, 1, 0.97))

out_pdf = OUT_DIR / "fig2_benchmark_gap.pdf"
out_png = OUT_DIR / "fig2_benchmark_gap.png"
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"wrote {out_pdf}")
print(f"wrote {out_png}")
