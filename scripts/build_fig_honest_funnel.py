"""Build fig_honest_funnel.pdf/.png — three-step collapse figure.

Wang 2025 trial-level 0.9047  →  subject-level CV (3 FMs × 3 seeds)
  →  LaBraM FT vs permutation null (p=0.70, indistinguishable).

Supports thesis: the 90% was an evaluation artifact, not a weak result.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Data (from docs/findings.md F-B, F-C.2, F-D.4 and paper_strategy.md §1)
# ---------------------------------------------------------------------------
WANG_BA = 0.9047

FM_POINTS = [
    ("LaBraM",   0.524, 0.010),
    ("CBraMod",  0.548, 0.031),
    ("REVE",     0.577, 0.051),
]

LABRAM_REAL_MEAN, LABRAM_REAL_STD = 0.443, 0.083
LABRAM_NULL_MEAN, LABRAM_NULL_STD = 0.497, 0.086
PERM_P = 0.70

# ---------------------------------------------------------------------------
# Color palette — cool → amber → red (temperature encodes "honesty")
# ---------------------------------------------------------------------------
COL_WANG = "#4E79A7"   # cool blue: unwarranted confidence
COL_FM   = "#E8A33D"   # amber: honest uncertainty
COL_NULL = "#C1453B"   # red: indistinguishable from chance
COL_NULLDIST = "#8B8B8B"  # gray for null distribution

BOX_EDGE = "#333333"
GRID = "#BFBFBF"

# ---------------------------------------------------------------------------
# Figure geometry — one axis spanning BA 0.30–1.00
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.8, 3.6))

BA_LO, BA_HI = 0.30, 1.00
ax.set_xlim(BA_LO, BA_HI)
ax.set_ylim(0, 3)

# Row centers (top = Wang, middle = FM, bottom = null)
Y_WANG = 2.55
Y_FM   = 1.55
Y_NULL = 0.55

ROW_HALF_H = 0.28  # half-height of each row's box/band

# ---- Reference vertical lines ---------------------------------------------
ax.axvline(0.50, color=GRID, lw=0.8, ls=(0, (3, 3)), zorder=1)
ax.axvline(0.90, color="#9CBDD9", lw=0.8, ls=(0, (3, 3)), zorder=1)

ax.text(0.50, 3.0, "chance", fontsize=7, color="#666", ha="center", va="top")
ax.text(0.895, 3.0, "ceiling perceived", fontsize=7, color="#4E79A7",
        ha="right", va="top")

# ============================================================================
# Level 1 — Wang 2025 trial-level CV (single point, no CI)
# ============================================================================
# Row label strip (left gutter) — use text outside axis via annotate
ax.text(BA_LO - 0.015, Y_WANG, "Wang 2025\ntrial-level CV",
        fontsize=8.5, color=BOX_EDGE, ha="right", va="center",
        fontweight="bold")

# A filled box at Wang's BA — wide enough to hold the number legibly
w_box_left, w_box_right = 0.845, 0.985
w_box_cx = 0.5 * (w_box_left + w_box_right)
box_w = FancyBboxPatch(
    (w_box_left, Y_WANG - ROW_HALF_H),
    w_box_right - w_box_left, 2 * ROW_HALF_H,
    boxstyle="round,pad=0.0,rounding_size=0.012",
    facecolor=COL_WANG, edgecolor=BOX_EDGE, lw=0.8, alpha=0.9, zorder=3,
)
ax.add_patch(box_w)
ax.text(w_box_cx, Y_WANG + 0.06, f"{WANG_BA:.4f}",
        ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
ax.text(w_box_cx, Y_WANG - 0.14, "n=1, no CI",
        ha="center", va="center", fontsize=6.0, color="white", style="italic")

# ============================================================================
# Level 2 — Subject-level CV, 3 FMs × 3 seeds
# ============================================================================
ax.text(BA_LO - 0.015, Y_FM, "Our subject-\nlevel CV\n(3 FMs x 3 seeds)",
        fontsize=8.5, color=BOX_EDGE, ha="right", va="center",
        fontweight="bold")

# Amber band spanning the FM range
fm_means = np.array([m for _, m, _ in FM_POINTS])
fm_stds  = np.array([s for _, _, s in FM_POINTS])
fm_lo = (fm_means - fm_stds).min() - 0.005
fm_hi = (fm_means + fm_stds).max() + 0.005
band_fm = FancyBboxPatch(
    (fm_lo, Y_FM - ROW_HALF_H),
    fm_hi - fm_lo, 2 * ROW_HALF_H,
    boxstyle="round,pad=0.0,rounding_size=0.012",
    facecolor=COL_FM, edgecolor=BOX_EDGE, lw=0.8, alpha=0.25, zorder=2,
)
ax.add_patch(band_fm)

# Plot three FM points with error bars + labels
for i, (name, mean, std) in enumerate(FM_POINTS):
    y_jit = Y_FM + (i - 1) * 0.11  # stack slightly so labels don't overlap
    ax.errorbar(mean, y_jit, xerr=std, fmt="o", color=COL_FM,
                ecolor=COL_FM, elinewidth=1.2, capsize=2.5,
                markersize=6, markeredgecolor=BOX_EDGE, markeredgewidth=0.6,
                zorder=4)
    ax.text(mean + std + 0.008, y_jit,
            f"{name} {mean:.3f}±{std:.3f}",
            fontsize=7.0, va="center", ha="left", color=BOX_EDGE)

# ============================================================================
# Level 3 — LaBraM FT vs permutation null
# ============================================================================
ax.text(BA_LO - 0.015, Y_NULL, "LaBraM FT vs\nperm null\n(p=0.70)",
        fontsize=8.5, color=BOX_EDGE, ha="right", va="center",
        fontweight="bold")

# Two overlapping Gaussian bell shapes on the bottom row
xs = np.linspace(BA_LO, BA_HI, 400)

def gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2)

bell_real = gauss(xs, LABRAM_REAL_MEAN, LABRAM_REAL_STD)
bell_null = gauss(xs, LABRAM_NULL_MEAN, LABRAM_NULL_STD)

bell_h = 0.42  # visual height of bells
base_y = Y_NULL - ROW_HALF_H + 0.02

ax.fill_between(xs, base_y, base_y + bell_h * bell_null,
                color=COL_NULLDIST, alpha=0.45, zorder=2,
                label=f"Null {LABRAM_NULL_MEAN:.3f}±{LABRAM_NULL_STD:.3f}")
ax.plot(xs, base_y + bell_h * bell_null, color=COL_NULLDIST, lw=0.9, zorder=3)

ax.fill_between(xs, base_y, base_y + bell_h * bell_real,
                color=COL_NULL, alpha=0.55, zorder=3,
                label=f"Real {LABRAM_REAL_MEAN:.3f}±{LABRAM_REAL_STD:.3f}")
ax.plot(xs, base_y + bell_h * bell_real, color=COL_NULL, lw=1.0, zorder=4)

# Mean markers
ax.plot([LABRAM_REAL_MEAN, LABRAM_REAL_MEAN], [base_y, base_y + bell_h],
        color=COL_NULL, lw=1.0, ls="--", zorder=5)
ax.plot([LABRAM_NULL_MEAN, LABRAM_NULL_MEAN], [base_y, base_y + bell_h * 0.95],
        color=COL_NULLDIST, lw=1.0, ls="--", zorder=4)

# Legend text inside the panel (compact) — place in upper-right where bells are flat
ax.text(0.735, Y_NULL + 0.18,
        f"real  {LABRAM_REAL_MEAN:.3f}±{LABRAM_REAL_STD:.3f}",
        fontsize=7, color=COL_NULL, fontweight="bold")
ax.text(0.735, Y_NULL + 0.08,
        f"null  {LABRAM_NULL_MEAN:.3f}±{LABRAM_NULL_STD:.3f}",
        fontsize=7, color=COL_NULLDIST, fontweight="bold")
ax.text(0.735, Y_NULL - 0.02,
        f"p = {PERM_P:.2f}  (indistinguishable)",
        fontsize=7, color=BOX_EDGE, style="italic")

# ============================================================================
# Arrows between levels, labeled with the correction
# ============================================================================
arrow_x = 0.335

def add_correction_arrow(y0, y1, label, sub):
    arr = FancyArrowPatch(
        (arrow_x, y0), (arrow_x, y1),
        arrowstyle="-|>", mutation_scale=14,
        color="#555", lw=1.3, zorder=6,
    )
    ax.add_patch(arr)
    ym = 0.5 * (y0 + y1)
    ax.text(arrow_x + 0.015, ym + 0.05, label,
            fontsize=7.3, color="#333", ha="left", va="center",
            fontweight="bold")
    ax.text(arrow_x + 0.015, ym - 0.08, sub,
            fontsize=6.8, color="#555", ha="left", va="center",
            style="italic")

add_correction_arrow(
    Y_WANG - ROW_HALF_H - 0.02, Y_FM + ROW_HALF_H + 0.02,
    "subject-level CV",
    "-35 pp",
)
add_correction_arrow(
    Y_FM - ROW_HALF_H - 0.02, Y_NULL + ROW_HALF_H + 0.02,
    "label-shuffle null",
    "real ~ null (p=0.70)",
)

# ============================================================================
# Axis cosmetics
# ============================================================================
ax.set_yticks([])
ax.set_xticks([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
ax.set_xticklabels(["0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90", "1.00"],
                   fontsize=7.5)
ax.set_xlabel("Balanced accuracy", fontsize=9)

for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#666")
ax.tick_params(axis="x", colors="#444", length=3)

# Subtitle
fig.suptitle("Honest evaluation collapses the UCSD Stress 90% benchmark",
             fontsize=10.5, fontweight="bold", y=0.995)

plt.tight_layout(rect=[0.13, 0.02, 0.99, 0.95])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
OUT_DIR = Path("/raid/jupyter-linjimmy1003.md10/UCSD_stress/paper/figures/main")
OUT_DIR.mkdir(parents=True, exist_ok=True)

pdf_path = OUT_DIR / "fig_honest_funnel.pdf"
png_path = OUT_DIR / "fig_honest_funnel.png"
fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Wrote {pdf_path}")
print(f"Wrote {png_path}")
