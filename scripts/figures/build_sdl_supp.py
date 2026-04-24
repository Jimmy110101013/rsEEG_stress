"""Supplementary S1–S4 for the SDL-critique TNSRE paper.

S1: table1_excluded.tex  — excluded benchmarks list (converted from markdown)
S2: fig_s2_seed_noise_hist.{pdf,png} — 10 000-iter bootstrap ρ histograms
S3: fig_s3_stress_dass.{pdf,png} — Stress DASS class-crossing per subject
S4: table_s4_hhsa_raw.tex — HHSA + raw per-seed ΔBA numbers

S2 rebuilds the bootstrap distribution inline (the original script that
produced seed_noise_bootstrap.json did not persist raw samples).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,
})

ROOT = Path("/raid/jupyter-linjimmy1003.md10/UCSD_stress")
TBL = ROOT / "results/studies/exp_30_sdl_vs_between/tables"
OUT_FIG = ROOT / "paper/figures/supplementary/sdl_critique"
OUT_TBL = ROOT / "paper/tables/sdl_critique"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TBL.mkdir(parents=True, exist_ok=True)

STRICT_WITHIN = {"eegmat", "sleepdep"}

# ======================================================================
# S1 — Excluded benchmarks (LaTeX version of existing markdown)
# ======================================================================
EXCLUDED = [
    ("ADFTD (3-class)",      "EEG-FM-Bench",  "Stratified 70/15/15", "Subject leakage"),
    ("TUEV",                 "EEG-FM-Bench",  "Stratified 80/10/10", "Subject leakage"),
    ("Mimul-11",             "EEG-FM-Bench",  "Stratified 76/12/12", "Subject leakage"),
    ("MentalArithmetic",     "EEG-FM-Bench",  "Stratified 72/14/14", "Subject leakage"),
    ("TUSL",                 "EEG-FM-Bench",  "Stratified",          "Subject leakage"),
    ("SEED / SEED-V",        "EEG-FM-Bench",  "Subject-dependent",   "Trial leakage"),
    ("Mumtaz2016 (MDD)",     "CBraMod",       "Unspecified",         "Ambiguous split"),
    ("REVE Table~2/4 rows",  "REVE",          "``same as earlier''", "Ambiguous split"),
]

lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Benchmarks excluded from Table~\ref{tab:benchmark_gap} because the source split protocol is stratified, subject-dependent, or unspecified.}",
    r"\label{tab:excluded}",
    r"\begin{tabular}{lllc}",
    r"\toprule",
    r"Benchmark & Source & Reported split & Reason \\",
    r"\midrule",
]
for bm, src, split, reason in EXCLUDED:
    lines.append(f"{bm} & {src} & {split} & {reason} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
(OUT_TBL / "table_s1_excluded.tex").write_text("\n".join(lines) + "\n")

# ======================================================================
# S2 — Seed-noise bootstrap histograms
# ======================================================================
# rebuild bootstrap inline using per-seed FT/LP BA from fm_performance.csv
perf = pd.read_csv(TBL / "fm_performance.csv")
master = pd.read_csv(TBL / "master_table.csv")


def per_seed_delta_ba(ds: str, fm: str) -> np.ndarray:
    sub = perf[(perf["dataset"] == ds) & (perf["fm"] == fm)]
    lp = sub[sub["mode"] == "lp"]["bal_acc"].dropna().to_numpy()
    ft = sub[sub["mode"] == "ft"]["bal_acc"].dropna().to_numpy()
    if len(lp) >= 1 and len(ft) >= 1:
        n = min(len(lp), len(ft))
        lp, ft = lp[:n], ft[:n]
        return ft - lp
    # fallback: master cell mean only (1 value)
    row = master[(master["dataset"] == ds) & (master["fm"] == fm)].iloc[0]
    return np.array([row["delta_ba"]])


def arm_cells(arm):
    if arm == "within_strict":
        return [(d, f) for d in STRICT_WITHIN for f in ["labram", "cbramod", "reve"]]
    if arm == "between":
        return [(d, f) for d in ["adftd", "tdbrain", "meditation"]
                for f in ["labram", "cbramod", "reve"]]
    raise ValueError(arm)


rng = np.random.default_rng(42)
N_BOOT = 10_000

def bootstrap_rho(predictor: str, arm: str):
    cells = arm_cells(arm)
    x = np.array([master[(master["dataset"] == d) & (master["fm"] == f)][predictor].iloc[0]
                  for d, f in cells])
    delta_samples = [per_seed_delta_ba(d, f) for d, f in cells]

    rhos = np.empty(N_BOOT)
    for b in range(N_BOOT):
        y = np.array([rng.choice(s) for s in delta_samples])
        r, _ = spearmanr(x, y)
        rhos[b] = 0.0 if np.isnan(r) else r
    return rhos


PREDS = ["subject_id_ba_lr", "subject_to_label_ratio"]
ARMS = ["within_strict", "between"]
PRED_LAB = {
    "subject_id_ba_lr": "Subject-ID LP BA",
    "subject_to_label_ratio": "Subject/label variance ratio",
}
ARM_LAB = {"within_strict": "Within-subject (strict)  n=6",
           "between":       "Between-subject           n=9"}

with open(TBL / "seed_noise_bootstrap.json") as fh:
    boot_summary = json.load(fh)

fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
for i, pred in enumerate(PREDS):
    for j, arm in enumerate(ARMS):
        ax = axes[i, j]
        rhos = bootstrap_rho(pred, arm)
        ax.hist(rhos, bins=60, color="#4c78a8" if arm == "within_strict" else "#e45756",
                alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.8)

        summary = boot_summary[f"{pred}_{arm}"]
        med = summary["median"]
        ci_lo = summary["ci_low"]
        ci_hi = summary["ci_high"]
        ax.axvline(med, color="#2ca02c", linewidth=1.8)
        ax.axvspan(ci_lo, ci_hi, color="#2ca02c", alpha=0.12)

        ax.text(0.03, 0.97,
                f"median $\\rho$ = {med:+.2f}\n"
                f"95\\% CI = [{ci_lo:+.2f}, {ci_hi:+.2f}]\n"
                f"P($\\rho$>0) = {summary['pct_gt_0']*100:.1f}\\%",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#888", linewidth=0.9))
        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel("Spearman $\\rho$")
        if j == 0:
            ax.set_ylabel("Bootstrap count")
        if i == 0:
            ax.set_title(ARM_LAB[arm], fontsize=11, fontweight="bold")
        if j == 0:
            ax.text(-1.15, 0.5, PRED_LAB[pred],
                    transform=ax.transAxes, rotation=90,
                    va="center", ha="center", fontsize=10.5, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.2)

fig.suptitle(
    "Fig S2 — Seed-noise bootstrap of Spearman $\\rho$ (10\\,000 iterations)",
    fontsize=12, fontweight="bold", y=0.995,
)
plt.tight_layout(rect=(0.02, 0, 1, 0.97))
plt.savefig(OUT_FIG / "fig_s2_seed_noise_hist.pdf", dpi=300, bbox_inches="tight")
plt.savefig(OUT_FIG / "fig_s2_seed_noise_hist.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT_FIG / 'fig_s2_seed_noise_hist.pdf'}")

# ======================================================================
# S3 — Stress DASS class-crossing per subject
# ======================================================================
STRESS_CSV = ROOT / "data/comprehensive_labels.csv"
if STRESS_CSV.exists():
    df = pd.read_csv(STRESS_CSV)
    # Per subject, list distinct Group values across recordings
    per_sub = (df.groupby("Patient_ID")["Group"]
                 .apply(lambda s: sorted(set(s.astype(str))))
                 .reset_index())
    per_sub["n_classes"] = per_sub["Group"].apply(len)
    per_sub["n_rec"] = df.groupby("Patient_ID").size().values

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    y = np.arange(len(per_sub))
    colors = ["#4c78a8" if n == 1 else "#e45756" for n in per_sub["n_classes"]]
    ax.barh(y, per_sub["n_rec"], color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(per_sub["Patient_ID"], fontsize=8)
    ax.set_xlabel("Number of recordings")
    ax.set_title(
        f"Fig S3 — Stress: {(per_sub['n_classes']>1).sum()}/{len(per_sub)} subjects cross DASS classes (red)",
        fontsize=11, fontweight="bold",
    )
    # annotate with class list
    for yi, (groups, nrec) in enumerate(zip(per_sub["Group"], per_sub["n_rec"])):
        ax.text(nrec + 0.15, yi, ",".join(groups),
                va="center", fontsize=7.5, color="#444")
    ax.set_axisbelow(True); ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "fig_s3_stress_dass.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_FIG / "fig_s3_stress_dass.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_FIG / 'fig_s3_stress_dass.pdf'}")
else:
    print(f"SKIP S3 — {STRESS_CSV} not found")

# ======================================================================
# S4 — HHSA + raw per-seed ΔBA (LaTeX table)
# ======================================================================
# HHSA contrast per dataset (5 datasets)
with open(TBL / "hhsa_contrast.json") as fh:
    hhsa = json.load(fh)

lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{HHSA median $g$ contrast per dataset. C4 (HHSA contrast tracks FM performance) was not supported at $n=9$ between-arm cells ($\rho=-0.26$, CI wide).}",
    r"\label{tab:hhsa_raw}",
    r"\begin{tabular}{lcr}",
    r"\toprule",
    r"Dataset & Arm & HHSA median $g$ \\",
    r"\midrule",
]
DS_LAB = {"eegmat": "EEGMAT", "sleepdep": "SleepDep", "stress": "Stress",
          "meditation": "Meditation", "tdbrain": "TDBRAIN", "adftd": "ADFTD"}
DS_ARM = {"eegmat": "Within", "sleepdep": "Within", "stress": "Within",
          "adftd": "Between", "tdbrain": "Between", "meditation": "Between"}
for row in hhsa:
    ds = row.get("dataset")
    g = row.get("hhsa_median_g")
    if ds is None:
        continue
    g_str = f"{g:.3f}" if (g is not None and not (isinstance(g, float) and np.isnan(g))) else "n/a"
    lines.append(f"{DS_LAB.get(ds, ds)} & {DS_ARM.get(ds,'?')} & {g_str} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
(OUT_TBL / "table_s4_hhsa_raw.tex").write_text("\n".join(lines) + "\n")

# per-seed ΔBA table (18 cells where available)
lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Raw per-seed $\Delta$BA (FT$-$LP) for each (dataset, FM) cell. Seeds 42, 123, 2024.}",
    r"\label{tab:raw_seeds}",
    r"\begin{tabular}{llrrrr}",
    r"\toprule",
    r"Dataset & FM & s42 & s123 & s2024 & mean \\",
    r"\midrule",
]
for ds in ["eegmat", "sleepdep", "stress", "adftd", "tdbrain", "meditation"]:
    for fm in ["labram", "cbramod", "reve"]:
        ds_lab = DS_LAB[ds] if fm == "labram" else ""
        deltas = per_seed_delta_ba(ds, fm)
        if len(deltas) >= 3:
            vals = [f"{d*100:+.1f}" for d in deltas[:3]]
            mean = f"{float(np.mean(deltas[:3]))*100:+.1f}"
        else:
            vals = ["--"] * 3
            mean = f"{float(deltas.mean())*100:+.1f}" if len(deltas) else "--"
        lines.append(f"{ds_lab} & {fm.upper()} & {vals[0]} & {vals[1]} & {vals[2]} & {mean} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
(OUT_TBL / "table_s4b_raw_seeds.tex").write_text("\n".join(lines) + "\n")

print(f"wrote {OUT_TBL / 'table_s1_excluded.tex'}")
print(f"wrote {OUT_TBL / 'table_s4_hhsa_raw.tex'}")
print(f"wrote {OUT_TBL / 'table_s4b_raw_seeds.tex'}")
