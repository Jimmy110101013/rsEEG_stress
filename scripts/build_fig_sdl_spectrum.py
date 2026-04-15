"""Fig 1 SDL Spectrum — visual statement of the SDL thesis.

Scatter of subject-level FT BA across datasets ordered by within-subject
neural-contrast anchor strength. Replaces the originally-considered
radar chart (per Gemini-style critique: a radar conflates within-subject
and cross-subject anchors, undermining the SDL definition; a 1D
spectrum makes the operative axis explicit).

X axis (categorical): anchor strength
    Absent  | Absent (within-subj.) | Intermediate | Strong
    (DASS)  | (longitudinal DSS)    | (TDBRAIN)    | (EEGMAT alpha-ERD)
Y axis: subject-level FT BA. Chance = 0.50, classical XGBoost ≈ 0.55.

Outlier: ADFTD — explicitly annotated as cross-subject categorical,
not within-subject paired; outside SDL's primary scope.

Output: paper/figures/main/fig_sdl_spectrum.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parent.parent

# ---------- Data --------------------------------------------------------
# Each entry: x_pos, label, ba_values (one per FM), fm_labels
COLS = [
    {
        "x": 0,
        "label": "UCSD Stress\nper-rec DASS",
        "anchor": "Absent",
        "subline": "(no published\nwithin-subj. correlate)",
        "ba": [0.524, 0.548, 0.577],
        "fms": ["LaBraM", "CBraMod", "REVE"],
    },
    {
        "x": 1,
        "label": "UCSD Stress\nlongitudinal DSS",
        "anchor": "Absent",
        "subline": "(within-subj.,\nno anchor)",
        "ba": [0.296, 0.241, 0.333],
        "fms": ["LaBraM", "CBraMod", "REVE"],
    },
    {
        "x": 2,
        "label": "TDBRAIN\nMDD vs HC",
        "anchor": "Intermediate",
        "subline": "(group-level lit.,\nno within-subj.)",
        "ba": [0.690, 0.498, 0.476],
        "fms": ["LaBraM", "CBraMod", "REVE"],
    },
    {
        "x": 3,
        "label": "EEGMAT\nrest vs arithmetic",
        "anchor": "Strong",
        "subline": "(Klimesch alpha-ERD,\nwithin-subj.)",
        "ba": [0.731, 0.620, 0.727],
        "fms": ["LaBraM", "CBraMod", "REVE"],
    },
]

# Outlier: ADFTD plotted off-axis as out-of-scope
ADFTD = {
    "label": "ADFTD AD vs HC",
    "ba": [0.703, 0.509, 0.662],
    "fms": ["LaBraM", "CBraMod", "REVE"],
}

# ---------- Plot --------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 4.6))

# Background regime shading
ax.axvspan(-0.5, 1.5, color="#FFC107", alpha=0.10, zorder=0)  # bounded
ax.axvspan(1.5, 2.5, color="#A0A0A0", alpha=0.10, zorder=0)   # intermediate
ax.axvspan(2.5, 3.5, color="#228833", alpha=0.10, zorder=0)   # anchored

# Chance line
ax.axhline(0.50, color="#555", lw=0.9, ls="--", alpha=0.8, zorder=1)
ax.text(3.40, 0.510, "chance", fontsize=7, color="#555", ha="right", va="bottom")

# Classical baseline reference
ax.axhline(0.553, color="#A85100", lw=0.9, ls=":", alpha=0.8, zorder=1)
ax.text(-0.45, 0.561, "classical (XGBoost balanced)", fontsize=7, color="#A85100",
        ha="left", va="bottom")

# FM marker styles
FM_COLOR = {"LaBraM": "#CC3311", "CBraMod": "#4477AA", "REVE": "#228833"}
FM_MARKER = {"LaBraM": "o", "CBraMod": "s", "REVE": "^"}

# Plot main columns
for col in COLS:
    x = col["x"]
    # subtle vertical guide
    ax.axvline(x, color="#DDDDDD", lw=0.5, zorder=0)
    # jitter so markers don't overlap when same-y
    n = len(col["ba"])
    if n > 1:
        x_jitter = np.linspace(-0.15, 0.15, n)
    else:
        x_jitter = [0.0]
    for ba, fm, jx in zip(col["ba"], col["fms"], x_jitter):
        ax.scatter(
            x + jx, ba,
            s=85, marker=FM_MARKER[fm], facecolor=FM_COLOR[fm],
            edgecolor="black", lw=0.7, zorder=4,
        )

# Plot ADFTD as outlier — off the spectrum, plotted at x = -1.2
ADFTD_X = -1.4
ax.axvline(ADFTD_X, color="#999999", lw=0.5, zorder=0, ls=":")
n_a = len(ADFTD["ba"])
x_jitter_a = np.linspace(-0.10, 0.10, n_a)
for ba, fm, jx in zip(ADFTD["ba"], ADFTD["fms"], x_jitter_a):
    ax.scatter(
        ADFTD_X + jx, ba,
        s=120, marker="*", facecolor=FM_COLOR[fm],
        edgecolor="black", lw=0.7, zorder=4,
    )

ax.annotate(
    "ADFTD AD vs HC\n(★ cross-subject only —\n outside within-subject\n SDL scope)",
    xy=(ADFTD_X, 0.62), xytext=(ADFTD_X - 0.05, 0.40),
    fontsize=7, color="#222", ha="center", va="top",
    arrowprops=dict(arrowstyle="-", color="#777", lw=0.7),
)

# Separator between ADFTD outlier and main spectrum
ax.axvline(-0.7, color="#888", lw=1.0, ls="-", alpha=0.6, zorder=1)
ax.text(-0.7, 0.90, "out-of-scope $|$ within-subject SDL spectrum",
        fontsize=7, color="#666", ha="center", va="bottom",
        rotation=0)

# X axis
xticks = [c["x"] for c in COLS]
xlabels = [c["label"] for c in COLS]
ax.set_xticks([ADFTD_X] + xticks)
ax.set_xticklabels(["ADFTD\n(outlier)"] + xlabels, fontsize=8)
ax.set_xlim(-2.0, 3.7)

# Anchor strength annotation row below x-axis
for col in COLS:
    ax.text(col["x"], 0.10, col["anchor"],
            fontsize=8, ha="center", va="center", style="italic",
            color="#444",
            transform=ax.get_xaxis_transform())

# Y axis
ax.set_ylabel("Subject-level fine-tuned balanced accuracy", fontsize=9)
ax.set_ylim(0.15, 0.92)
ax.set_yticks(np.arange(0.2, 0.95, 0.1))
ax.grid(True, axis="y", alpha=0.20, zorder=0)

# Top header
ax.set_title("SDL Spectrum: FT performance vs within-subject neural anchor strength",
             fontsize=10, pad=12)

# X axis caption (anchor strength label)
ax.text(0.5, -0.16,
        "→ within-subject neural contrast strength →",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=8, color="#444", style="italic")

# Legend (FMs)
legend_handles = [
    Line2D([0], [0], marker=FM_MARKER[fm], color="w",
           markerfacecolor=FM_COLOR[fm], markeredgecolor="black",
           markersize=8, label=fm)
    for fm in ["LaBraM", "CBraMod", "REVE"]
]
ax.legend(handles=legend_handles, loc="upper left",
          bbox_to_anchor=(0.02, 0.98), fontsize=7.5,
          framealpha=0.95, ncol=3, handlelength=1.0)

# Tight layout + save
plt.tight_layout()
out_pdf = ROOT / "paper/figures/main/fig_sdl_spectrum.pdf"
out_png = ROOT / "paper/figures/main/fig_sdl_spectrum.png"
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"wrote {out_pdf}")
print(f"wrote {out_png}")

print("\nDatapoints:")
for col in COLS:
    print(f"  x={col['x']:5.1f}  {col['label'].replace(chr(10),' '):40s}  "
          f"BA={col['ba']}")
print(f"  ADFTD outlier: BA={ADFTD['ba']}")
