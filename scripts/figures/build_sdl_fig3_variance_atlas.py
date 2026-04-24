"""Fig 3 — Variance atlas + subject-ID decodability composite (C1 evidence).

Panel A (top): 3 FM × 6 dataset grid of stacked bars showing
    Label% / Subject% / Residual% of explainable variance in frozen features.

Panel B (bottom): bar chart of subject_id_ba_lr for the same 18 cells,
    arm-coloured (within = grey/blue, between = red/orange).
    Chance baseline (1/n_subjects) shown as per-cell horizontal tick.

Data sources:
    results/studies/exp_30_sdl_vs_between/tables/variance_decomposition.csv
    results/studies/exp_30_sdl_vs_between/tables/subject_decodability.csv
    results/studies/exp_30_sdl_vs_between/tables/master_table.csv

Output: paper/figures/main/sdl_critique/fig3_variance_atlas.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,
})

ROOT = Path("/raid/jupyter-linjimmy1003.md10/UCSD_stress")
TBL = ROOT / "results/studies/exp_30_sdl_vs_between/tables"
OUT_DIR = ROOT / "paper/figures/main/sdl_critique"
OUT_DIR.mkdir(parents=True, exist_ok=True)

var_df = pd.read_csv(TBL / "variance_decomposition.csv")
dec_df = pd.read_csv(TBL / "subject_decodability.csv")
master = pd.read_csv(TBL / "master_table.csv")

DATASETS_WITHIN = ["eegmat", "sleepdep", "stress"]
DATASETS_BETWEEN = ["adftd", "tdbrain", "meditation"]
DATASETS = DATASETS_WITHIN + DATASETS_BETWEEN
FMS = ["labram", "cbramod", "reve"]

DS_LABEL = {
    "eegmat": "EEGMAT", "sleepdep": "SleepDep", "stress": "Stress",
    "adftd": "ADFTD", "tdbrain": "TDBRAIN", "meditation": "Meditation",
}
FM_LABEL = {"labram": "LaBraM", "cbramod": "CBraMod", "reve": "REVE"}

# Panel A colours (stacked bars)
COL_LABEL    = "#4c78a8"   # label signal
COL_SUBJECT  = "#e45756"   # subject signal
COL_RESIDUAL = "#bab0ab"   # residual / noise

# Panel B colours — within (cool) / between (warm)
ARM_COLOR = {
    "within":  "#4c78a8",
    "between": "#e45756",
}

fig = plt.figure(figsize=(12.6, 8.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.34)

# -------- Panel A: variance decomposition -----------------------------
axA = fig.add_subplot(gs[0])
n_cells = len(DATASETS) * len(FMS)
x = np.arange(n_cells)

label_frac, subj_frac, resid_frac, arm_per_cell, xtick_labels = [], [], [], [], []
for di, ds in enumerate(DATASETS):
    for fi, fm in enumerate(FMS):
        row = var_df[(var_df["dataset"] == ds) & (var_df["fm"] == fm)].iloc[0]
        label_frac.append(row["frozen_label_frac"])
        subj_frac.append(row["frozen_subject_frac"])
        resid_frac.append(row["frozen_residual_frac"])
        arm = "within" if ds in DATASETS_WITHIN else "between"
        arm_per_cell.append(arm)
        xtick_labels.append(FM_LABEL[fm])

label_frac = np.array(label_frac)
subj_frac = np.array(subj_frac)
resid_frac = np.array(resid_frac)

bw = 0.85
axA.bar(x, label_frac, bw, color=COL_LABEL, edgecolor="black", linewidth=0.4, label="Label %")
axA.bar(x, subj_frac, bw, bottom=label_frac, color=COL_SUBJECT, edgecolor="black", linewidth=0.4, label="Subject %")
axA.bar(x, resid_frac, bw, bottom=label_frac + subj_frac,
        color=COL_RESIDUAL, edgecolor="black", linewidth=0.4, label="Residual %")

# annotate subject % inside the bar
for xi, s in zip(x, subj_frac):
    axA.text(xi, label_frac[xi] + s / 2, f"{s:.0f}",
             ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")

# dataset group labels + separators
axA.set_xticks(x)
axA.set_xticklabels(xtick_labels, rotation=0, fontsize=8.5)
for di, ds in enumerate(DATASETS):
    center = di * 3 + 1
    axA.text(center, 105, DS_LABEL[ds], ha="center", va="bottom",
             fontsize=10.5, fontweight="bold")
    if di < len(DATASETS) - 1:
        axA.axvline(di * 3 + 2.5, color="#888", linewidth=0.7, alpha=0.5)

# arm separator (between within + between)
axA.axvline(len(DATASETS_WITHIN) * 3 - 0.5, color="black", linewidth=1.2, linestyle="-", alpha=0.6)
axA.text(4.0, 125, "Within-subject arm", ha="center",
         fontsize=10.5, fontstyle="italic", color="#4c78a8", fontweight="bold")
axA.text(13.0, 125, "Between-subject arm", ha="center",
         fontsize=10.5, fontstyle="italic", color="#e45756", fontweight="bold")

axA.set_ylabel("Explainable variance (%)")
axA.set_ylim(0, 135)
axA.set_title("A. Variance decomposition of frozen features (C1: subject signal dominates across all 18 cells)",
              fontsize=11, fontweight="bold", loc="left")
axA.legend(loc="lower right", ncol=3, fontsize=9, frameon=True, bbox_to_anchor=(1.0, 0.02))
axA.set_axisbelow(True)
axA.grid(True, axis="y", alpha=0.25)

# -------- Panel B: subject-ID decodability ---------------------------
axB = fig.add_subplot(gs[1])

subj_ba = []
chance = []
for di, ds in enumerate(DATASETS):
    for fi, fm in enumerate(FMS):
        row = dec_df[(dec_df["dataset"] == ds) & (dec_df["fm"] == fm)].iloc[0]
        subj_ba.append(row["subject_id_ba_lr"])
        chance.append(1.0 / float(row["n_subjects"]))
subj_ba = np.array(subj_ba)
chance = np.array(chance)

# Chance-adjusted: (BA - chance) / (1 - chance).
# Range [0, 1]; 0 = at chance, 1 = perfect subject decoding.
# Necessary because n_subjects differs 4×–20× across datasets, so raw BA
# is not comparable across arms.
above = (subj_ba - chance) / (1.0 - chance)
above = np.clip(above, 0, 1)

bar_colors = [ARM_COLOR[a] for a in arm_per_cell]
axB.bar(x, above, bw, color=bar_colors, edgecolor="black", linewidth=0.5)

# annotate numeric value above each bar
for xi, v in zip(x, above):
    axB.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5, color="#333")

axB.set_xticks(x)
axB.set_xticklabels(xtick_labels, rotation=0, fontsize=8.5)

for di, ds in enumerate(DATASETS):
    center = di * 3 + 1
    axB.text(center, 0.97, DS_LABEL[ds], ha="center", va="bottom",
             fontsize=10.5, fontweight="bold")
    if di < len(DATASETS) - 1:
        axB.axvline(di * 3 + 2.5, color="#888", linewidth=0.7, alpha=0.5)
axB.axvline(len(DATASETS_WITHIN) * 3 - 0.5, color="black", linewidth=1.2, linestyle="-", alpha=0.6)

axB.set_ylabel("Chance-adjusted subject-ID decodability\n(BA − chance) / (1 − chance)")
axB.set_ylim(0, 1.02)
axB.set_title(
    "B. Subject-ID linear decodability from frozen features — chance-adjusted "
    "(0 = at chance = 1/$n_{\\mathrm{sub}}$; 1 = perfect decoding)",
    fontsize=11, fontweight="bold", loc="left",
)
axB.grid(True, axis="y", alpha=0.25)

# legend for Panel B — top right so it doesn't cover EEGMAT on left
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, fc=ARM_COLOR["within"],  ec="black", label="Within-subject arm"),
    plt.Rectangle((0, 0), 1, 1, fc=ARM_COLOR["between"], ec="black", label="Between-subject arm"),
]
axB.legend(handles=legend_elements, loc="lower right", fontsize=9, frameon=True,
           bbox_to_anchor=(1.0, 0.02))

fig.suptitle(
    "C1 — Frozen foundation-model features are dominated by subject identity across 6 EEG benchmarks",
    fontsize=12.5, fontweight="bold", y=0.995,
)

plt.tight_layout(rect=(0, 0, 1, 0.975))

out_pdf = OUT_DIR / "fig3_variance_atlas.pdf"
out_png = OUT_DIR / "fig3_variance_atlas.png"
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"wrote {out_pdf}")
print(f"wrote {out_png}")
