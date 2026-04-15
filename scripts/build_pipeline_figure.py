"""Fig 1: Pipeline and subject-level CV protocol schematic.

Panels:
  (a) Data → epoch → per-model norm → FM backbone → {frozen LP, FT}.
  (b) Subject-level StratifiedGroupKFold(5) vs trial-level StratifiedKFold(5)
      visualization on the 17-subject/70-recording Stress layout.

No computation required — purely a matplotlib diagram.

Output: paper/figures/main/fig1_pipeline.pdf
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

matplotlib.rcParams.update({
    "font.size": 9.5,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,
})

fig = plt.figure(figsize=(10.5, 6.4))

# Two stacked panels: (a) top, (b) bottom
gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0], hspace=0.35)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[1, 0])


# ---------------------------------------------------------------------------
# (a) Pipeline row
# ---------------------------------------------------------------------------
ax_a.set_xlim(0, 100)
ax_a.set_ylim(0, 40)
ax_a.axis("off")
ax_a.text(1.5, 37, "(a) Pipeline",
          fontsize=11.5, fontweight="bold")


def box(ax, x, y, w, h, label, color, edge="black", fontsize=9.5, fontweight="normal"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=1.4",
        linewidth=1.1, facecolor=color, edgecolor=edge,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight)
    return (x, y, w, h)


def arrow(ax, x1, y1, x2, y2, color="#444"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->", mutation_scale=12,
        linewidth=1.2, color=color,
    )
    ax.add_patch(a)


# Row boxes (single horizontal pipeline)
y0 = 17
h = 10
box(ax_a, 2,  y0, 16, h,
    "EEG\n17 subj / 70 rec\n30 ch, 200 Hz",
    "#eef3fa")
box(ax_a, 22, y0, 16, h,
    "Window\n5 s (LaBraM/\nCBraMod)\n10 s (REVE)",
    "#eef3fa")
box(ax_a, 42, y0, 16, h,
    "Per-model norm\nLaBraM: z-score\nCBraMod / REVE:\nnone (µV scale)",
    "#fff3e4",
    fontweight="bold")
box(ax_a, 62, y0, 16, h,
    "FM backbone\n{LaBraM,\nCBraMod, REVE}",
    "#eaf4ea")

# Global pool branch-point
box(ax_a, 82, y0, 14, h,
    "Global\npool\n→ (B, d)",
    "#eaf4ea")

for (x1, x2) in [(18, 22), (38, 42), (58, 62), (78, 82)]:
    arrow(ax_a, x1, y0 + h / 2, x2, y0 + h / 2)

# Two output heads from the pool
box(ax_a, 68, 2, 12, 8,
    "Frozen LP\n(encoder frozen)",
    "#e0e7ef")
box(ax_a, 84, 2, 12, 8,
    "Fine-tune\n(encoder trained)",
    "#fadada")

arrow(ax_a, 87, y0, 74, 10)
arrow(ax_a, 90, y0, 90, 10)

# Annotate call-out on per-model norm
ax_a.annotate(
    "silent failure mode\nif mishandled",
    xy=(50, y0), xytext=(50, y0 - 6.5),
    fontsize=8.3, color="#c55", ha="center",
    arrowprops=dict(arrowstyle="->", color="#c55", lw=0.9),
)

# Annotate CV protocol feed
ax_a.text(50, 31.5,
          "Subject-level StratifiedGroupKFold(5) — see panel (b)",
          fontsize=9.5, style="italic", color="#333", ha="center")


# ---------------------------------------------------------------------------
# (b) Subject-level vs trial-level CV schematic
# ---------------------------------------------------------------------------
ax_b.set_xlim(0, 100)
ax_b.set_ylim(0, 40)
ax_b.axis("off")
ax_b.text(1.5, 37,
          "(b) Cross-validation protocol: subject-level (ours) vs trial-level (literature)",
          fontsize=11.5, fontweight="bold")

# We visualize 12 recordings across 4 subjects for compact readability.
# (Real config: 17 subjects, 70 recordings, 5 folds.)
subjects = [
    ("S1", 3),
    ("S2", 3),
    ("S3", 3),
    ("S4", 3),
]
labels = ["+", "+", "-", "-", "-", "+", "-", "-", "-", "+", "-", "-"]
fold_subj  = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]

# Colors: fold assignment — 3 folds for readability (ours uses 5).
fold_colors = ["#4c78a8", "#f28e2b", "#59a14f"]

# Ours: subject-level — all recs of a subject stay in one fold.
fold_ours  = [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0]
# Literature: trial-level — recs from the same subject can be split.
fold_trial = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

# Geometry
xs = np.linspace(8, 92, 12)
y_ours  = 22
y_trial = 6
w = 5.5
h = 5.5

def draw_row(ax, y, fold_of, title, is_leaky):
    for i, x in enumerate(xs):
        c = fold_colors[fold_of[i]]
        rect = Rectangle((x - w / 2, y - h / 2), w, h,
                         facecolor=c, edgecolor="black", linewidth=0.7)
        ax.add_patch(rect)
        ax.text(x, y, labels[i], ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold")
    # Subject brackets above
    for (subj_name, _), gx in zip(subjects,
                                  [xs[1], xs[4], xs[7], xs[10]]):
        ax.text(gx, y + h / 2 + 1.4, subj_name, ha="center",
                fontsize=8.8)
    # Row title left
    ax.text(-0.5, y,
            title, ha="left", va="center",
            fontsize=9.8, fontweight="bold")
    if is_leaky:
        for gx in [xs[1], xs[4], xs[7], xs[10]]:
            ax.annotate("same subject\nacross folds",
                        xy=(gx, y - h / 2 - 0.3),
                        xytext=(gx, y - h / 2 - 3.6),
                        fontsize=7.2, color="#c33",
                        ha="center",
                        arrowprops=dict(arrowstyle="-", color="#c33",
                                        lw=0.7))

draw_row(ax_b, y_ours,  fold_ours,
         "Subject-level\n(ours)", is_leaky=False)
draw_row(ax_b, y_trial, fold_trial,
         "Trial-level\n(Wang 2025)", is_leaky=True)

# Fold legend
for i, c in enumerate(fold_colors):
    rect = Rectangle((72 + i * 8, 31.5), 4, 2.4,
                     facecolor=c, edgecolor="black", linewidth=0.5)
    ax_b.add_patch(rect)
    ax_b.text(74 + i * 8, 32.7, f"fold {i+1}",
              ha="center", va="center", fontsize=7.8, color="white",
              fontweight="bold")
ax_b.text(71, 32.7, "legend:", ha="right", va="center",
          fontsize=8.5)

# Key-takeaway annotation between the two rows
ax_b.text(50, 14.3,
          "Subject identity leaks across folds → inflated BA (F01: ~30 pp gap)",
          fontsize=9.2, style="italic", color="#c33", ha="center")

plt.tight_layout()
out_pdf = "paper/figures/main/fig1_pipeline.pdf"
out_png = out_pdf.replace(".pdf", ".png")
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"wrote {out_pdf}")
