"""Fig 4 — C2 main claim figure.

Top 2x2 scatter grid:
    rows   : predictor ∈ {subject_id_ba_lr, subject_to_label_ratio}
    columns: arm       ∈ {within_strict (n=6), between (n=9)}
Each panel plots ΔBA vs predictor, colours points by FM, annotates point
Spearman ρ + CI and seed-robust median ρ + CI.

Bottom row — arm-comparison summary (Panel C):
    (i) mean ΔBA (± seed SD)            — answers "are FT gains different?"
    (ii) mean chance-adjusted subject_id — answers "is subject signal different?"
    (iii) Spearman ρ(subj_id, ΔBA)       — answers "does subject signal predict ΔBA?"

The three bottom panels make the mechanism story explicit: means (i, ii)
are comparable across arms, yet the correlation (iii) is arm-specific.

Data sources:
    results/studies/exp_30_sdl_vs_between/tables/master_table.csv
    results/studies/exp_30_sdl_vs_between/tables/subject_decodability.csv
    results/studies/exp_30_sdl_vs_between/tables/correlations.json
    results/studies/exp_30_sdl_vs_between/tables/seed_noise_bootstrap.json
    results/studies/exp_30_sdl_vs_between/tables/fm_performance.csv

Output: paper/figures/main/sdl_critique/fig4_c2_main.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

ROOT = Path("/raid/jupyter-linjimmy1003.md10/UCSD_stress")
TBL = ROOT / "results/studies/exp_30_sdl_vs_between/tables"
OUT_DIR = ROOT / "paper/figures/main/sdl_critique"
OUT_DIR.mkdir(parents=True, exist_ok=True)

master = pd.read_csv(TBL / "master_table.csv")
perf = pd.read_csv(TBL / "fm_performance.csv")
dec_df = pd.read_csv(TBL / "subject_decodability.csv")
with open(TBL / "correlations.json") as fh:
    corrs = json.load(fh)
with open(TBL / "seed_noise_bootstrap.json") as fh:
    seed_boot = json.load(fh)

STRICT_WITHIN = {"eegmat", "sleepdep"}
BETWEEN_DS = {"adftd", "tdbrain", "meditation"}
FMS = ["labram", "cbramod", "reve"]

def subset(predictor: str, arm: str) -> pd.DataFrame:
    if arm == "within_strict":
        df = master[(master["arm"] == "within") & (master["dataset"].isin(STRICT_WITHIN))]
    elif arm == "between":
        df = master[master["arm"] == "between"]
    else:
        raise ValueError(arm)
    return df[["dataset", "fm", predictor, "delta_ba"]].dropna()

def lookup_point(predictor, arm):
    return next(r for r in corrs if r["predictor"] == predictor and r["arm"] == arm)

def lookup_seed(predictor, arm):
    return seed_boot[f"{predictor}_{arm}"]

# ======================================================================
FM_COLOR = {"labram": "#4c78a8", "cbramod": "#f58518", "reve": "#54a24b"}
FM_LABEL = {"labram": "LaBraM", "cbramod": "CBraMod", "reve": "REVE"}
DS_MARKER = {"eegmat": "o", "sleepdep": "s", "adftd": "D",
             "tdbrain": "^", "meditation": "v"}
DS_LABEL = {"eegmat": "EEGMAT", "sleepdep": "SleepDep",
            "adftd": "ADFTD", "tdbrain": "TDBRAIN", "meditation": "Meditation"}

PRED_LABEL = {
    "subject_id_ba_lr":       "Subject-ID linear-probe BA",
    "subject_to_label_ratio": "Subject / label variance ratio",
}
ARM_LABEL = {
    "within_strict": "Within-subject (strict)  n = 6",
    "between":       "Between-subject           n = 9",
}
ARM_BG = {"within_strict": "#ffffff", "between": "#f2f7fd"}
ARM_COLOR_BAR = {"within_strict": "#4c78a8", "between": "#e45756"}
ARM_SHORT = {"within_strict": "Within-strict", "between": "Between"}

# ======================================================================
fig = plt.figure(figsize=(10.5, 12.0))
gs_top    = fig.add_gridspec(2, 2, top=0.93, bottom=0.42, hspace=0.36, wspace=0.28)
gs_bottom = fig.add_gridspec(1, 3, top=0.32, bottom=0.12, wspace=0.40)

axes = np.empty((2, 2), dtype=object)
for i in range(2):
    for j in range(2):
        axes[i, j] = fig.add_subplot(gs_top[i, j])

axC_delta = fig.add_subplot(gs_bottom[0, 0])
axC_subj  = fig.add_subplot(gs_bottom[0, 1])
axC_rho   = fig.add_subplot(gs_bottom[0, 2])

predictors = ["subject_id_ba_lr", "subject_to_label_ratio"]
arms = ["within_strict", "between"]

# ---- Top: 2x2 scatter -------------------------------------------------
for i, pred in enumerate(predictors):
    for j, arm in enumerate(arms):
        ax = axes[i, j]
        ax.set_facecolor(ARM_BG[arm])

        df = subset(pred, arm)
        for _, row in df.iterrows():
            ax.scatter(row[pred], row["delta_ba"], s=110,
                       c=FM_COLOR[row["fm"]], marker=DS_MARKER[row["dataset"]],
                       edgecolor="black", linewidth=0.8, alpha=0.92, zorder=3)

        ax.axhline(0.0, color="#888", linewidth=0.7, linestyle="--", alpha=0.7, zorder=1)

        if len(df) >= 3:
            x = df[pred].to_numpy()
            y = df["delta_ba"].to_numpy()
            rx = pd.Series(x).rank().to_numpy()
            slope, intercept = np.polyfit(rx, y, 1)
            order = np.argsort(x)
            ax.plot(x[order], (slope * rx + intercept)[order],
                    color="#333", linewidth=1.2, alpha=0.55, zorder=2)

        if pred == "subject_to_label_ratio":
            ax.set_xscale("log")

        pt = lookup_point(pred, arm)
        sb = lookup_seed(pred, arm)
        ci_pt = (pt["ci_low"], pt["ci_high"])
        pct_dir = sb["pct_gt_0"] if sb["median"] >= 0 else (1 - sb["pct_gt_0"])
        ann = (
            f"Point ρ = {pt['rho']:+.2f}  "
            f"[{ci_pt[0]:+.2f}, {ci_pt[1]:+.2f}]  p = {pt['p']:.3f}\n"
            f"Seed-robust median ρ = {sb['median']:+.2f}  "
            f"[{sb['ci_low']:+.2f}, {sb['ci_high']:+.2f}]\n"
            f"P(sign matches) = {pct_dir*100:.1f} %"
        )
        ci_excludes_0 = (ci_pt[0] > 0) or (ci_pt[1] < 0)
        box_edge = "#2ca02c" if ci_excludes_0 else "#888"
        ax.text(0.03, 0.97, ann, transform=ax.transAxes,
                fontsize=8.6, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=box_edge, linewidth=1.1))

        if i == 0:
            ax.set_title(ARM_LABEL[arm], fontsize=11, fontweight="bold")
        ax.set_xlabel(PRED_LABEL[pred])
        if j == 0:
            ax.set_ylabel("ΔBA  =  FT − LP")
        ax.grid(True, alpha=0.25)

# ---- Bottom: arm-comparison summary (Panel C) -------------------------
def per_seed_delta(ds, fm):
    sub = perf[(perf["dataset"] == ds) & (perf["fm"] == fm)]
    lp = sub[sub["mode"] == "lp"]["bal_acc"].dropna().to_numpy()
    ft = sub[sub["mode"] == "ft"]["bal_acc"].dropna().to_numpy()
    if len(lp) >= 1 and len(ft) >= 1:
        n = min(len(lp), len(ft))
        return ft[:n] - lp[:n]
    row = master[(master["dataset"] == ds) & (master["fm"] == fm)].iloc[0]
    return np.array([row["delta_ba"]])

def arm_cells(arm):
    dss = STRICT_WITHIN if arm == "within_strict" else BETWEEN_DS
    return [(d, f) for d in dss for f in FMS]

def arm_chance_adjusted_subj(arm):
    vals = []
    for ds, fm in arm_cells(arm):
        row = dec_df[(dec_df["dataset"] == ds) & (dec_df["fm"] == fm)].iloc[0]
        ba = float(row["subject_id_ba_lr"])
        chance = 1.0 / float(row["n_subjects"])
        vals.append(max(0.0, (ba - chance) / (1.0 - chance)))
    return np.array(vals)

def arm_pooled_delta(arm):
    arr = [per_seed_delta(d, f) for d, f in arm_cells(arm)]
    return np.concatenate(arr)

# (i) mean ΔBA bars
means_delta = {a: arm_pooled_delta(a) for a in arms}
x = np.arange(2)
bar_col = [ARM_COLOR_BAR[a] for a in arms]
axC_delta.bar(x,
              [means_delta[a].mean() * 100 for a in arms],
              yerr=[means_delta[a].std(ddof=1) * 100 for a in arms],
              color=bar_col, edgecolor="black", linewidth=0.8,
              capsize=5, error_kw=dict(ecolor="#222", lw=1.0))
for xi, a in zip(x, arms):
    v = means_delta[a].mean() * 100
    axC_delta.text(xi, v + 0.6 if v >= 0 else v - 1.2,
                   f"{v:+.1f} pp", ha="center", fontsize=9, fontweight="bold")
axC_delta.set_xticks(x)
axC_delta.set_xticklabels([ARM_SHORT[a] for a in arms])
axC_delta.axhline(0.0, color="#888", linewidth=0.7, linestyle="--")
axC_delta.set_ylabel("Mean ΔBA  (pp)")
axC_delta.set_title("(i) Mean FT gain", fontsize=10.5, fontweight="bold")
axC_delta.grid(True, axis="y", alpha=0.25)
axC_delta.set_ylim(-7, 7)

# (ii) mean chance-adjusted subject_id_ba
means_subj = {a: arm_chance_adjusted_subj(a) for a in arms}
axC_subj.bar(x,
             [means_subj[a].mean() for a in arms],
             yerr=[means_subj[a].std(ddof=1) for a in arms],
             color=bar_col, edgecolor="black", linewidth=0.8,
             capsize=5, error_kw=dict(ecolor="#222", lw=1.0))
for xi, a in zip(x, arms):
    v = means_subj[a].mean()
    axC_subj.text(xi, v + 0.02, f"{v:.2f}",
                  ha="center", fontsize=9, fontweight="bold")
axC_subj.set_xticks(x)
axC_subj.set_xticklabels([ARM_SHORT[a] for a in arms])
axC_subj.set_ylabel("Chance-adjusted subject-ID BA")
axC_subj.set_title("(ii) Mean subject signal", fontsize=10.5, fontweight="bold")
axC_subj.grid(True, axis="y", alpha=0.25)
axC_subj.set_ylim(0, 0.45)

# (iii) Spearman ρ with 95% CI (seed-robust)
rho_medians = []
ci_lows = []
ci_highs = []
for a in arms:
    sb = seed_boot[f"subject_id_ba_lr_{a}"]
    rho_medians.append(sb["median"])
    ci_lows.append(sb["ci_low"])
    ci_highs.append(sb["ci_high"])
rho_medians = np.array(rho_medians)
ci_lows = np.array(ci_lows)
ci_highs = np.array(ci_highs)

axC_rho.bar(x, rho_medians,
            yerr=[rho_medians - ci_lows, ci_highs - rho_medians],
            color=bar_col, edgecolor="black", linewidth=0.8,
            capsize=7, error_kw=dict(ecolor="#222", lw=1.2))
for xi, a, r, lo, hi in zip(x, arms, rho_medians, ci_lows, ci_highs):
    ci_excludes_0 = (lo > 0) or (hi < 0)
    marker = "★" if ci_excludes_0 else ""
    axC_rho.text(xi, hi + 0.06,
                 f"ρ = {r:+.2f}{marker}",
                 ha="center", fontsize=9, fontweight="bold",
                 color="#2ca02c" if ci_excludes_0 else "#555")
axC_rho.axhline(0.0, color="#888", linewidth=0.7, linestyle="--")
axC_rho.set_xticks(x)
axC_rho.set_xticklabels([ARM_SHORT[a] for a in arms])
axC_rho.set_ylabel("Seed-robust Spearman ρ\n(subject-ID BA → ΔBA)")
axC_rho.set_title("(iii) Correlation with ΔBA ★ = CI excludes 0",
                  fontsize=10.5, fontweight="bold")
axC_rho.grid(True, axis="y", alpha=0.25)
axC_rho.set_ylim(-1.05, 1.15)

# ---- Legend (shared FM × dataset) -------------------------------------
fm_handles = [
    plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
               markerfacecolor=FM_COLOR[k], markeredgecolor="black",
               label=FM_LABEL[k])
    for k in ["labram", "cbramod", "reve"]
]
ds_handles = [
    plt.Line2D([0], [0], marker=DS_MARKER[k], linestyle="", markersize=9,
               markerfacecolor="#cccccc", markeredgecolor="black",
               label=DS_LABEL[k])
    for k in ["eegmat", "sleepdep", "adftd", "tdbrain", "meditation"]
]
fig.legend(handles=fm_handles + ds_handles, loc="lower center",
           ncol=8, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, 0.00))

# ---- Panel labels ----
fig.text(0.06, 0.935, "A", fontsize=16, fontweight="bold")
fig.text(0.06, 0.345, "C", fontsize=16, fontweight="bold")

fig.suptitle(
    "C2 — Subject-leakage signatures predict FT gain in between-subject benchmarks",
    fontsize=12.5, fontweight="bold", y=0.985,
)
# Panel A header
fig.text(0.5, 0.955,
         "Panel A. Per-cell scatter: 2 predictors × 2 arms",
         ha="center", fontsize=10, fontweight="bold", color="#333")
# Panel C header
fig.text(0.5, 0.355,
         "Panel C. Arm-level summary — means (i, ii) are comparable; "
         "only between-arm shows significant ρ (iii)",
         ha="center", fontsize=10, fontweight="bold", color="#333")

out_pdf = OUT_DIR / "fig4_c2_main.pdf"
out_png = OUT_DIR / "fig4_c2_main.png"
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"wrote {out_pdf}")
print(f"wrote {out_png}")
