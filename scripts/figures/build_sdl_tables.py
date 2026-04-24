"""Emit LaTeX for Tables I–IV of the SDL-critique TNSRE paper.

Outputs:
    paper/tables/sdl_critique/table1_datasets.tex
    paper/tables/sdl_critique/table2_benchmark_gap.tex
    paper/tables/sdl_critique/table3_fm_performance.tex
    paper/tables/sdl_critique/table4_correlations.tex

All tables use IEEEtran-compatible \\begin{table}[t] ... \\end{table}
with booktabs (\\toprule, \\midrule, \\bottomrule).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/raid/jupyter-linjimmy1003.md10/UCSD_stress")
TBL = ROOT / "results/studies/exp_30_sdl_vs_between/tables"
OUT = ROOT / "paper/tables/sdl_critique"
OUT.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Table I — Dataset summary
# ======================================================================
DS_ROWS = [
    # dataset, arm, N sub, N rec, label design, task type, source
    ("EEGMAT",     "Within",  36,  72,  "Task vs Rest",     "Mental arithmetic",       "PhysioNet"),
    ("SleepDep",   "Within",  36,  72,  "NS vs SD",         "Sleep deprivation",       "OpenNeuro ds004902"),
    ("Stress$^\\dagger$", "Within",  17,  70,  "DASS (3 cross)",   "Longitudinal rest",       "Komarov 2020"),
    ("ADFTD",      "Between", 65,  195, "AD vs CN",         "Diagnostic (dementia)",   "OpenNeuro ds004504"),
    ("TDBRAIN",    "Between", 359, 734, "MDD vs CN",        "Diagnostic (mood)",       "TDBRAIN"),
    ("Meditation", "Between", 24,  40,  "Expert vs Novice", "Longitudinal rest",       "Private"),
]

lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Dataset summary (six EEG benchmarks). $^{\dagger}$Stress is excluded from the clean within-subject arm because only $3/17$ subjects cross DASS classes.}",
    r"\label{tab:datasets}",
    r"\begin{tabular}{lcccll}",
    r"\toprule",
    r"Dataset & Arm & $N_{\mathrm{sub}}$ & $N_{\mathrm{rec}}$ & Label design & Source \\",
    r"\midrule",
]
for ds, arm, nsub, nrec, lab, task, src in DS_ROWS:
    lines.append(f"{ds} & {arm} & {nsub} & {nrec} & {lab} & {src} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

(OUT / "table1_datasets.tex").write_text("\n".join(lines) + "\n")


# ======================================================================
# Table II — Benchmark gap (literature + ours, subject-level CV only)
# ======================================================================
# Literature rows from EEG-FM-Bench arXiv 2508.17742v2, Appendix 6:
#   Frozen (F) = Table 4 (multi-task, avg-pool head)
#   Fine-tuned (FT) = Table 3 (multi-task FT, avg-pool head)
#   Both multi-task → apples-to-apples pairing.
BENCH = [
    # dataset, arm, n_cls, n_sub, lab_f, lab_ft, cbm_f, cbm_ft, reve_f, reve_ft, source
    ("TUAB",         "Between", 2, "$\\sim$600", 75.87, 79.36, 73.15, 80.49, 63.80, 80.32, "EEG-FM-Bench T3/T4"),
    ("Siena",        "Between", 2, 18,     50.00, 71.99, 64.17, 82.75, 68.60, 70.65, "EEG-FM-Bench T3/T4"),
    ("ADFTD",        "Between", 2, 65,     69.5,  70.9,  55.8,  53.7,  69.2,  65.8,  "Ours"),
    ("TDBRAIN",      "Between", 2, 359,    67.9,  66.5,  56.4,  48.9,  54.4,  48.8,  "Ours"),
    ("Meditation",   "Between", 2, 24,     47.3,  51.5,  71.0,  68.3,  53.8,  43.3,  "Ours"),
    ("HMC",          "Within",  5, 151,    59.80, 71.63, 51.81, 71.08, 63.80, 71.43, "EEG-FM-Bench T3/T4"),
    ("PhysioMI",     "Within",  4, 109,    29.63, 43.19, 26.90, 31.15, 27.37, 30.63, "EEG-FM-Bench T3/T4"),
    ("BCIC-IV-2a",   "Within",  4, 9,      28.40, 34.58, 29.17, 35.50, 28.63, 36.89, "EEG-FM-Bench T3/T4"),
    ("SEED-VII",     "Within",  7, 20,     23.23, 26.13, 19.43, 26.05, 20.57, 20.76, "EEG-FM-Bench T3/T4"),
    ("Things-EEG-2", "Within",  2, 10,     50.00, 50.90, 50.00, 50.70, 50.00, 59.43, "EEG-FM-Bench T3/T4"),
    ("EEGMAT",       "Within",  2, 36,     67.1,  73.1,  73.1,  62.0,  67.1,  72.7,  "Ours"),
    ("SleepDep",     "Within",  2, 36,     50.0,  53.2,  55.7,  55.6,  54.4,  54.2,  "Ours"),
]

lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Frozen$\rightarrow$FT balanced accuracy (\%) for three EEG foundation models on benchmarks with explicit subject-level cross-validation. Literature rows from EEG-FM-Bench (arXiv~2508.17742v2, 2025): Frozen from Table~4 (multi-task, avg-pool head), FT from Table~3 (multi-task FT, avg-pool head); this pairing is apples-to-apples. Rows labelled \emph{Ours} are from our exp\_30 pipeline. Benchmarks using stratified or subject-dependent splits in source papers are excluded (see Table S1). Stress (UCSD) is omitted because $3/17$ subjects cross DASS classes.}",
    r"\label{tab:benchmark_gap}",
    r"\begin{tabular}{llccrrrrrrl}",
    r"\toprule",
    r" & & & & \multicolumn{2}{c}{LaBraM} & \multicolumn{2}{c}{CBraMod} & \multicolumn{2}{c}{REVE} & \\",
    r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
    r"Benchmark & Arm & $K$ & $N_{\mathrm{sub}}$ & F & FT & F & FT & F & FT & Source \\",
    r"\midrule",
]
last_arm = None
for row in BENCH:
    ds, arm, k, nsub, lf, lft, cf, cft, rf, rft, src = row
    if last_arm is not None and arm != last_arm:
        lines.append(r"\midrule")
    last_arm = arm
    nsub_fmt = nsub if isinstance(nsub, str) else f"{nsub}"
    lines.append(
        f"{ds} & {arm} & {k} & {nsub_fmt} & "
        f"{lf:.1f} & {lft:.1f} & {cf:.1f} & {cft:.1f} & {rf:.1f} & {rft:.1f} & {src} \\\\"
    )

# mean rows
def mean_rows():
    bet = [r for r in BENCH if r[1] == "Between"]
    wit = [r for r in BENCH if r[1] == "Within"]
    def means(rows):
        arr = np.array([[r[i] for i in (4, 5, 6, 7, 8, 9)] for r in rows], dtype=float)
        return arr.mean(axis=0)
    bm = means(bet)
    wm = means(wit)
    return bm, wm
bm, wm = mean_rows()
lines.append(r"\midrule")
lines.append(
    f"\\emph{{Between mean}} & & & & "
    f"{bm[0]:.1f} & {bm[1]:.1f} & {bm[2]:.1f} & {bm[3]:.1f} & {bm[4]:.1f} & {bm[5]:.1f} & \\\\"
)
lines.append(
    f"\\emph{{Within mean}}  & & & & "
    f"{wm[0]:.1f} & {wm[1]:.1f} & {wm[2]:.1f} & {wm[3]:.1f} & {wm[4]:.1f} & {wm[5]:.1f} & \\\\"
)
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table*}")

(OUT / "table2_benchmark_gap.tex").write_text("\n".join(lines) + "\n")


# ======================================================================
# Table III — FM performance matrix (18 cells with LP ± std, FT ± std, ΔBA)
# ======================================================================
perf = pd.read_csv(TBL / "fm_performance.csv")

DATASETS_ORDER = ["eegmat", "sleepdep", "stress", "adftd", "tdbrain", "meditation"]
ARMS = {
    "eegmat": "Within", "sleepdep": "Within", "stress": "Within",
    "adftd": "Between", "tdbrain": "Between", "meditation": "Between",
}
DS_LABEL = {
    "eegmat": "EEGMAT", "sleepdep": "SleepDep", "stress": "Stress",
    "adftd": "ADFTD", "tdbrain": "TDBRAIN", "meditation": "Meditation",
}
FM_ORDER = ["labram", "cbramod", "reve"]
FM_LABEL = {"labram": "LaBraM", "cbramod": "CBraMod", "reve": "REVE"}


def summarize_cell(ds, fm, mode):
    sub = perf[(perf["dataset"] == ds) & (perf["fm"] == fm) & (perf["mode"] == mode)]
    # If rows have per-seed bal_acc values, compute directly; otherwise use bal_acc + std
    vals = sub["bal_acc"].dropna().to_numpy()
    if len(vals) >= 2:
        return float(vals.mean()), float(vals.std(ddof=1)), int(len(vals))
    if len(vals) == 1:
        # single row with pre-computed std and n
        s = sub.iloc[0]
        m = float(s["bal_acc"])
        sd = float(s["bal_acc_std"]) if pd.notna(s["bal_acc_std"]) else 0.0
        nn = int(s["n"]) if pd.notna(s["n"]) else 1
        return m, sd, nn
    return float("nan"), float("nan"), 0


lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Frozen linear-probe (LP) vs fine-tuned (FT) balanced accuracy for three EEG foundation models across six datasets. Each cell reports mean $\pm$ sample std over three seeds unless otherwise indicated; $\Delta$BA $=$ FT $-$ LP. Negative values indicate FT underperforms LP.}",
    r"\label{tab:fm_performance}",
    r"\begin{tabular}{llcccr}",
    r"\toprule",
    r"Dataset & FM & LP BA & FT BA & $\Delta$BA & $n_{\mathrm{seed}}$ \\",
    r"\midrule",
]
last_arm = None
for ds in DATASETS_ORDER:
    arm = ARMS[ds]
    if last_arm is not None and arm != last_arm:
        lines.append(r"\midrule")
    last_arm = arm
    for fi, fm in enumerate(FM_ORDER):
        lp_m, lp_s, _ = summarize_cell(ds, fm, "lp")
        ft_m, ft_s, n = summarize_cell(ds, fm, "ft")
        delta = ft_m - lp_m
        ds_cell = DS_LABEL[ds] if fi == 0 else ""
        sign = "$+$" if delta >= 0 else "$-$"
        dval = abs(delta) * 100
        lines.append(
            f"{ds_cell} & {FM_LABEL[fm]} & "
            f"{lp_m*100:.1f} $\\pm$ {lp_s*100:.1f} & "
            f"{ft_m*100:.1f} $\\pm$ {ft_s*100:.1f} & "
            f"{sign}{dval:.1f} & {n} \\\\"
        )
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table*}")

(OUT / "table3_fm_performance.tex").write_text("\n".join(lines) + "\n")


# ======================================================================
# Table IV — Master correlations (ρ point + CI + seed-robust median ρ + CI)
# ======================================================================
with open(TBL / "correlations.json") as fh:
    corrs = json.load(fh)
with open(TBL / "seed_noise_bootstrap.json") as fh:
    seed_boot = json.load(fh)

PRED_LABEL = {
    "frozen_label_frac":        "Label variance fraction",
    "frozen_subject_frac":      "Subject variance fraction",
    "subject_to_label_ratio":   "Subject/label variance ratio",
    "rsa_subject":              "RSA (subject)",
    "rsa_label":                "RSA (label)",
    "subject_id_ba_lr":         "Subject-ID LP BA",
    "subject_id_ba_mlp":        "Subject-ID MLP BA",
    "permanova_pseudo_F":       "PERMANOVA pseudo-$F$",
}
ARM_ORDER = ["within_strict", "between", "pooled"]
ARM_LABEL = {"within_strict": "Within-strict", "between": "Between", "pooled": "Pooled"}

lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Spearman correlations between representation-level predictors and $\Delta$BA. \emph{Point} $\rho$ is computed on the raw point estimates; \emph{seed-robust} $\rho$ is the median of 10\,000 bootstrap samples drawn by resampling each cell's $\Delta$BA from its per-seed distribution (available for within-strict and between arms only).}",
    r"\label{tab:correlations}",
    r"\begin{tabular}{llcrrrr}",
    r"\toprule",
    r" & & & \multicolumn{2}{c}{Point} & \multicolumn{2}{c}{Seed-robust} \\",
    r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
    r"Predictor & Arm & $n$ & $\rho$ & 95\% CI & median $\rho$ & 95\% CI \\",
    r"\midrule",
]

predictor_order = [
    "subject_id_ba_lr",
    "subject_to_label_ratio",
    "rsa_label",
    "frozen_subject_frac",
    "frozen_label_frac",
    "rsa_subject",
    "subject_id_ba_mlp",
    "permanova_pseudo_F",
]
for pi, pred in enumerate(predictor_order):
    for ai, arm in enumerate(ARM_ORDER):
        row = next((r for r in corrs if r["predictor"] == pred and r["arm"] == arm), None)
        if row is None or row.get("rho") is None:
            continue
        rho = row["rho"]
        ci_lo = row["ci_low"]; ci_hi = row["ci_high"]
        p = row.get("p")
        p_str = "" if p is None else f" ({p:.3f})"
        seed_key = f"{pred}_{arm}"
        if seed_key in seed_boot:
            sb = seed_boot[seed_key]
            seed_med = f"{sb['median']:+.2f}"
            seed_ci  = f"[{sb['ci_low']:+.2f},{sb['ci_high']:+.2f}]"
        else:
            seed_med = "--"
            seed_ci = "--"
        pred_cell = PRED_LABEL[pred] if ai == 0 else ""
        lines.append(
            f"{pred_cell} & {ARM_LABEL[arm]} & {row['n']} & "
            f"{rho:+.2f}{p_str} & [{ci_lo:+.2f},{ci_hi:+.2f}] & "
            f"{seed_med} & {seed_ci} \\\\"
        )
    if pi < len(predictor_order) - 1:
        lines.append(r"\addlinespace")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table*}")

(OUT / "table4_correlations.tex").write_text("\n".join(lines) + "\n")

print("wrote:")
for f in ("table1_datasets.tex", "table2_benchmark_gap.tex",
          "table3_fm_performance.tex", "table4_correlations.tex"):
    print(f"  {OUT / f}")
