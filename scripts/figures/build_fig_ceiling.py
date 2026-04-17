"""Fig ceiling: architecture-agnostic BA ceiling on UCSD Stress (70 rec / 17 subj).

Horizontal forest plot of 9 architectures spanning 6 orders of magnitude in
parameter count. All converge to the 0.44-0.58 BA band under subject-level CV,
supporting the paper's central claim that the Stress ceiling is a task
property, not a model property. Frozen LP (LaBraM) at 0.605 is the single
highest number — shown as a reference line to underscore that FT does not
exceed frozen.

Sources
-------
- Classical (5-fold CV, fold-std as error):
    results/studies/exp02_classical_dass/rerun_70rec/summary.json
- From-scratch deep (3-seed mean ± std):
    results/studies/exp15_nonfm_baselines/sweep/{shallowconvnet_lr1e-4,eegnet_lr5e-4}_s*/summary.json
- FM FT (3-seed best-HP from F-C.2 / F05 in findings.md):
    LaBraM 0.524 ± 0.010, CBraMod 0.548 ± 0.031, REVE 0.577 ± 0.051
- LaBraM frozen LP reference: 0.605 (findings.md F-C.1)

Output
------
paper/figures/main/fig_ceiling.pdf (+ .png, 300 dpi)
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

matplotlib.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parent.parent

# ---------- Load classical (5-fold CV, fold-std) --------------------------
with open(ROOT / "results/studies/exp02_classical_dass/rerun_70rec/summary.json") as f:
    classical = json.load(f)["models"]


def fold_mean_std(name: str) -> tuple[float, float]:
    d = classical[name]
    folds = [f["bal_acc"] for f in d["folds"]]
    return float(d["bal_acc"]), float(st.stdev(folds))


rf_m, rf_s = fold_mean_std("RF")
lr_m, lr_s = fold_mean_std("LogReg_L2")
svm_m, svm_s = fold_mean_std("SVM_RBF")

# XGBoost sourced from class-balanced audit (M2, R1 C3 reviewer fix):
# sklearn GradientBoostingClassifier has no class_weight param, so sample_weight
# was added retroactively. Use the 'balanced' variant as the headline number.
with open(ROOT / "results/studies/exp02_classical_dass/rerun_70rec_xgb_balanced/summary.json") as f:
    xgb_bal = json.load(f)["balanced"]
xgb_m = float(xgb_bal["bal_acc_mean"])
xgb_s = float(xgb_bal["bal_acc_std"])

# ---------- Load from-scratch deep (3 seeds) ------------------------------
def agg_seeds(cfg_prefix: str) -> tuple[float, float, int]:
    base = ROOT / "results/studies/exp15_nonfm_baselines/sweep"
    vals: list[float] = []
    for d in base.iterdir():
        if not d.name.startswith(cfg_prefix):
            continue
        if not any(d.name.endswith(s) for s in ["_s42", "_s123", "_s2024"]):
            continue
        with open(d / "summary.json") as fh:
            s = json.load(fh)
        ba = s.get("mean_bal_acc") or s.get("bal_acc")
        if ba is None:
            for k, v in s.items():
                if isinstance(v, (int, float)) and "bal" in k.lower():
                    ba = v
                    break
        if ba is not None:
            vals.append(float(ba))
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


scn_m, scn_s, _ = agg_seeds("shallowconvnet_lr1e-4")
egn_m, egn_s, _ = agg_seeds("eegnet_lr5e-4")

# ---------- FM FT best-HP (F-C.2) ----------------------------------------
cbra_m, cbra_s = 0.548, 0.031
labram_m, labram_s = 0.524, 0.010
reve_m, reve_s = 0.577, 0.051

# ---------- Frozen LP reference (F-C.1) ----------------------------------
labram_lp = 0.605

# ---------- Assemble rows (sorted by parameter count ascending) -----------
# params: 0 for classical on spectral features; published counts for deep nets.
# EEGNet/ShallowConvNet param counts are the published ~2-4k and ~40-50k
# ballpark. Exact numbers come from our build (Lawhern 2018 / Schirrmeister
# 2017 reimplementations) — we report order of magnitude only.
rows = [
    # label,            mean,    std,     params,       group,        marker_filled
    ("Random Forest",   rf_m,    rf_s,    None,         "classical",  False),
    ("LogReg (L2)",     lr_m,    lr_s,    None,         "classical",  False),
    ("SVM (RBF)",       svm_m,   svm_s,   None,         "classical",  False),
    ("XGBoost",         xgb_m,   xgb_s,   None,         "classical",  False),
    ("EEGNet",          egn_m,   egn_s,   3_000,        "deep",       True),
    ("ShallowConvNet*", scn_m,   scn_s,   40_000,       "deep",       True),
    ("CBraMod (FT)",    cbra_m,  cbra_s,  100_000_000,  "fm",         True),
    ("LaBraM (FT)",     labram_m,labram_s,100_000_000,  "fm",         True),
    ("REVE (FT)",       reve_m,  reve_s,  1_400_000_000,"fm",         True),
]

n = len(rows)

# ---------- Color palette (colorblind-safe, Tol / Wong) -------------------
GROUP_COLOR = {
    "classical": "#4477AA",  # blue
    "deep":      "#228833",  # green
    "fm":        "#CC3311",  # red
}
GROUP_LABEL = {
    "classical": "Classical ML (spectral features)",
    "deep":      "Deep CNN (from scratch)",
    "fm":        "Pretrained Foundation Model (FT)",
}

# ---------- Plot ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.5, 5.0))

# Shaded convergence band
ax.axvspan(0.44, 0.58, color="#FFC107", alpha=0.18, zorder=0,
           label="Convergence band (0.44-0.58)")

# Chance line
ax.axvline(0.50, color="#555", linewidth=0.9, linestyle="--", alpha=0.8, zorder=1)

# Frozen LP reference line (vertical)
ax.axvline(labram_lp, color="#777", linewidth=0.9, linestyle=":", alpha=0.9, zorder=1)
ax.text(labram_lp - 0.003, 4.5,
        f"LaBraM frozen LP (no FT) = {labram_lp:.3f}",
        fontsize=6.5, color="#555", va="center", ha="right", rotation=90)

# Alternate group shading (horizontal bands)
y_positions = np.arange(n)[::-1]  # top row = first in `rows`
group_bounds: dict[str, list[int]] = {}
for i, (_, _, _, _, g, _) in enumerate(rows):
    group_bounds.setdefault(g, []).append(i)

# Shade alternate groups
for gi, (g, idxs) in enumerate(group_bounds.items()):
    if gi % 2 == 1:
        y_top = y_positions[idxs[0]] + 0.5
        y_bot = y_positions[idxs[-1]] - 0.5
        ax.axhspan(y_bot, y_top, color="#EEEEEE", alpha=0.5, zorder=0)

# Plot points
for i, (label, m, s, _, g, filled) in enumerate(rows):
    y = y_positions[i]
    c = GROUP_COLOR[g]
    # Error bar (std)
    ax.errorbar(m, y, xerr=s, fmt="none", ecolor=c, elinewidth=1.2,
                capsize=3, zorder=3)
    # Marker: hollow for classical (low trust), solid for DL
    if filled:
        ax.scatter([m], [y], s=55, facecolor=c, edgecolor="black",
                   linewidth=0.7, zorder=4)
    else:
        ax.scatter([m], [y], s=55, facecolor="white", edgecolor=c,
                   linewidth=1.4, zorder=4)
    # Bold ShallowConvNet (rhetorical anchor)
    is_anchor = label.startswith("ShallowConvNet")
    fw = "bold" if is_anchor else "normal"
    # BA annotation to the right (outside the x-axis data area)
    ax.text(0.715, y, f"{m:.3f} ± {s:.3f}", fontsize=7.5, va="center",
            ha="left", color="black", fontweight=fw,
            family="monospace", clip_on=False)

# Y-tick labels: architecture names
ytick_labels = []
for label, _, _, p, _, _ in rows:
    if p is None:
        p_str = "~0 params"
    elif p < 1e6:
        p_str = f"~{p/1e3:.0f}k params"
    elif p < 1e9:
        p_str = f"~{p/1e6:.0f}M params"
    else:
        p_str = f"~{p/1e9:.1f}B params"
    ytick_labels.append(f"{label}\n{p_str}")

ax.set_yticks(y_positions)
ax.set_yticklabels(ytick_labels, fontsize=7.5)

# Bold the anchor ytick
for tick, (label, *_rest) in zip(ax.get_yticklabels(), rows):
    if label.startswith("ShallowConvNet"):
        tick.set_fontweight("bold")

ax.set_xlim(0.35, 0.70)
ax.set_ylim(-0.7, n - 0.3)
ax.set_xlabel("Balanced Accuracy (subject-level 5-fold CV)")
ax.set_xticks(np.arange(0.35, 0.71, 0.05))
ax.grid(True, axis="x", alpha=0.25, zorder=0)
ax.set_axisbelow(True)

# Remove top/right spines for cleaner look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Small chance-line label
ax.text(0.502, n - 0.45, "chance", fontsize=6.5, color="#555", rotation=90,
        va="top", ha="left")

# Legend (groups + marker convention)
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor=GROUP_COLOR["classical"], markeredgewidth=1.4,
           markersize=7, label="Classical ML (hollow)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_COLOR["deep"],
           markeredgecolor="black", markersize=7, label="Deep CNN (from scratch)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_COLOR["fm"],
           markeredgecolor="black", markersize=7, label="Foundation Model (FT)"),
    Patch(facecolor="#FFC107", alpha=0.30, label="Convergence band"),
]
ax.legend(handles=legend_handles, loc="upper left",
          bbox_to_anchor=(0.0, -0.12), ncol=2, fontsize=6.5,
          framealpha=0.95, borderpad=0.4, handlelength=1.4)

# Caption-style text below the plot
caption = (
    "All 9 architectures (~0 to 1.4B params) converge to the 0.44-0.58 BA band.\n"
    "Classical: 5-fold CV fold-std. DL: 3-seed mean ± std. *ShallowConvNet\n"
    "(~40k params, 2017) matches best FM FT (REVE, ~1.4B). FT never exceeds\n"
    "LaBraM frozen LP (0.605, dotted line) - pretraining confers no FT gain."
)
fig.text(0.02, -0.10, caption, fontsize=6.5, color="#333", ha="left", va="top",
         family="sans-serif")

plt.tight_layout()
out_pdf = ROOT / "paper/figures/main/fig_ceiling.pdf"
out_png = ROOT / "paper/figures/main/fig_ceiling.png"
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"wrote {out_pdf}")
print(f"wrote {out_png}")
print()
print("Summary of rows:")
for label, m, s, p, g, _ in rows:
    print(f"  {label:20s}  BA={m:.3f} ± {s:.3f}  group={g}")
print(f"\nFrozen LP reference: LaBraM = {labram_lp:.3f}")
