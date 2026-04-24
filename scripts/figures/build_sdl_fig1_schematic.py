"""Fig 1 — Experimental design schematic.

Pipeline overview:
    6 datasets (3 within + 3 between) → 3 FMs → LP/FT protocol → metrics tree

Output: paper/figures/main/sdl_critique/fig1_pipeline.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,
})

ROOT = Path("/raid/jupyter-linjimmy1003.md10/UCSD_stress")
OUT_DIR = ROOT / "paper/figures/main/sdl_critique"
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(13.0, 7.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.set_aspect("auto")
ax.axis("off")

# -------- helpers --------------------------------------------------------
def box(x, y, w, h, label, face="#eef4fb", edge="#4c78a8", fontsize=9,
        fontweight="normal", text_color="black", multiline=True, text_pad=0):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.4",
        linewidth=1.8, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2 + text_pad, label,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_color,
            wrap=multiline)

def arrow(x0, y0, x1, y1, color="#333", lw=1.4, style="-|>", curve=0):
    if curve == 0:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw))
    else:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                    connectionstyle=f"arc3,rad={curve}"))

# -------- stage 1: datasets --------------------------------------------
ax.text(7, 57, "Datasets (n = 6)", fontsize=11.5, fontweight="bold", ha="center")

within_dsets = [
    ("EEGMAT",  "36 sub / 72 rec",  "task vs rest"),
    ("SleepDep","36 sub / 72 rec",  "NS vs SD"),
    ("Stress",  "17 sub / 70 rec",  "DASS (3 crossing)"),
]
between_dsets = [
    ("ADFTD",     "65 sub / 195 rec", "AD vs CN"),
    ("TDBRAIN",   "359 sub / 734 rec","MDD vs CN"),
    ("Meditation","24 sub / 40 rec",  "Expert vs Novice"),
]

# within arm header
ax.text(7, 53, "Within-subject arm", fontsize=9.5, fontweight="bold",
        color="#4c78a8", ha="center", fontstyle="italic")
for i, (name, n, task) in enumerate(within_dsets):
    y = 46 - i * 5.2
    box(1, y, 12, 4.4, f"{name}\n{n}\n{task}",
        face="#eef4fb", edge="#4c78a8", fontsize=8.5)

# between arm header
ax.text(7, 30.5, "Between-subject arm", fontsize=9.5, fontweight="bold",
        color="#e45756", ha="center", fontstyle="italic")
for i, (name, n, task) in enumerate(between_dsets):
    y = 24 - i * 5.2
    box(1, y, 12, 4.4, f"{name}\n{n}\n{task}",
        face="#fdecec", edge="#e45756", fontsize=8.5)

# -------- stage 2: FMs -------------------------------------------------
ax.text(26, 57, "Foundation models", fontsize=11.5, fontweight="bold", ha="center")
fms = [("LaBraM",  "200-dim"),
       ("CBraMod", "200-dim"),
       ("REVE",    "512-dim")]
fm_colors = ["#4c78a8", "#f58518", "#54a24b"]
for i, ((fm, dim), c) in enumerate(zip(fms, fm_colors)):
    y = 44 - i * 8
    box(20, y, 12, 6, f"{fm}\n({dim})",
        face="white", edge=c, fontsize=10, fontweight="bold")

# datasets → FM arrows (converging)
for di in range(6):
    y_src = 48.2 - di * 5.2 if di < 3 else 48.2 - (di - 3) * 5.2 - 17
    for fi, c in enumerate(fm_colors):
        y_tgt = 47 - fi * 8
        arrow(13, y_src, 20, y_tgt, color="#aaa", lw=0.7, style="-")

# -------- stage 3: protocol --------------------------------------------
ax.text(45, 57, "Evaluation protocol", fontsize=11.5, fontweight="bold", ha="center")
box(38, 42, 14, 6, "Linear probing\n(frozen backbone + MLP head)",
    face="#eef4fb", edge="#4c78a8", fontsize=9)
box(38, 32, 14, 6, "Fine-tuning\n(full-parameter, 3 seeds)",
    face="#fdecec", edge="#e45756", fontsize=9)
box(38, 22, 14, 6, "Subject-level 5-fold CV\n(StratifiedGroupKFold)",
    face="white", edge="black", fontsize=9, fontweight="bold")

for fi in range(3):
    y_src = 47 - fi * 8
    arrow(32, y_src, 38, 45, color="#aaa", lw=0.7, style="-")
    arrow(32, y_src, 38, 35, color="#aaa", lw=0.7, style="-")

arrow(45, 42, 45, 28, color="#333", lw=1.0)
arrow(45, 32, 45, 28, color="#333", lw=1.0)

# -------- stage 4: outputs ΔBA -----------------------------------------
ax.text(67, 57, "Performance metric", fontsize=11.5, fontweight="bold", ha="center")
box(60, 40, 14, 10,
    "LP BA   FT BA\n\n$\\Delta$BA = FT − LP\n\n18 cells\n(6 datasets × 3 FMs)",
    face="#fff9e3", edge="#333", fontsize=9.2)
arrow(52, 25, 60, 45, color="#333", lw=1.3)

# -------- stage 5: analysis tree ---------------------------------------
ax.text(88, 57, "Representation analysis", fontsize=11.5, fontweight="bold", ha="center")
box(80, 48, 17, 5, "Variance decomposition\n(Label / Subject / Residual %)",
    face="white", edge="#333", fontsize=8.5)
box(80, 41.5, 17, 5, "Subject-ID linear decodability\n(LogReg probe on frozen features)",
    face="white", edge="#333", fontsize=8.5)
box(80, 35, 17, 5, "RSA (label / subject)  +  PERMANOVA",
    face="white", edge="#333", fontsize=8.5)
box(80, 28.5, 17, 5, "Subject / label variance ratio",
    face="white", edge="#333", fontsize=8.5)

for y_tgt in [50.5, 44, 37.5, 31]:
    arrow(74, 45, 80, y_tgt, color="#aaa", lw=0.7, style="-")

# -------- stage 6: claims ----------------------------------------------
box(60, 10, 37, 11,
    "C1  Subject dominance\nfrozen_subject_frac > 40 % in all 18 cells\n\n"
    "C2  Subject-leakage signatures predict FT gain\n"
    "     in between-subject arm  (n = 9, $\\rho$ = +0.50)",
    face="#f0f8ef", edge="#2ca02c", fontsize=9.5, fontweight="bold")

arrow(88, 28, 88, 21, color="#2ca02c", lw=1.5)
arrow(67, 40, 67, 21, color="#2ca02c", lw=1.5, style="-|>", curve=-0.2)

# -------- stage 7: bench gap context -----------------------------------
# annotation: split-protocol caveat (subtle, no box)
ax.text(22, 5,
        "All 6 benchmarks run under subject-level CV — Table II compares to\n"
        "literature benchmarks with the same split protocol",
        fontsize=8.5, color="#555", ha="center", fontstyle="italic")

fig.suptitle(
    "Fig 1  —  Experimental design: 6 EEG benchmarks × 3 foundation models × LP/FT protocol",
    fontsize=12.5, fontweight="bold", y=0.985,
)

out_pdf = OUT_DIR / "fig1_pipeline.pdf"
out_png = OUT_DIR / "fig1_pipeline.png"
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"wrote {out_pdf}")
print(f"wrote {out_png}")
