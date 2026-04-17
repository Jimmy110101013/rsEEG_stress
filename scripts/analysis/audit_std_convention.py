"""Audit std convention across findings.md numeric claims.

For every `mean ± std` claim, locate the source data and recompute both
population (n) and sample (n-1) std. Report findings.md value vs correct
sample-std value.
"""
import json
import math
from pathlib import Path
from collections import OrderedDict

ROOT = Path(".")


def sample_stats(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return None, None, None
    mean = sum(vals) / n
    if n == 1:
        return mean, 0.0, 0.0
    var_n   = sum((x - mean) ** 2 for x in vals) / n
    var_nm1 = sum((x - mean) ** 2 for x in vals) / (n - 1)
    return mean, math.sqrt(var_n), math.sqrt(var_nm1)


def load_ba(path, field="subject_bal_acc"):
    try:
        with open(path) as f:
            d = json.load(f)
        # Try multiple common keys
        for k in [field, "subject_bal_acc", "mean_bal_acc", "bal_acc"]:
            if k in d and isinstance(d[k], (int, float)):
                return d[k]
        return None
    except FileNotFoundError:
        return None


rows = []

# ---- F05 Stress frozen LP multi-seed ----
frozen_lp_path = ROOT / "results/studies/exp03_stress_erosion/frozen_lp"
for model in ["labram", "cbramod", "reve"]:
    fn = frozen_lp_path / f"{model}_multi_seed.json"
    if not fn.exists():
        # fallback to older multi_seed.json for labram only
        fn = frozen_lp_path / "multi_seed.json"
    if fn.exists():
        with open(fn) as f:
            d = json.load(f)
        # Find per-seed BAs
        per_seed = d.get("per_seed_ba") or d.get("per_seed") or d
        if isinstance(per_seed, dict):
            vals = [v for v in per_seed.values() if isinstance(v, (int, float))]
        else:
            vals = list(per_seed)
        m, std_n, std_nm1 = sample_stats(vals)
        rows.append((f"F05 {model} frozen LP", len(vals), m, std_n, std_nm1, str(fn)))

# ---- F05 Stress best FT (from hp_sweep) ----
sweep = ROOT / "results/hp_sweep/20260410_dass"
best_configs = {
    "labram":  ("labram",  "lr1e-4",  "encoderlrscale1.0"),
    "cbramod": ("cbramod", "lr1e-5",  "encoderlrscale0.1"),
    "reve":    ("reve",    "lr3e-5",  "encoderlrscale0.1"),
}
for tag, (model, lr, elrs) in best_configs.items():
    vals = []
    for seed in ["s42", "s123", "s2024"]:
        run = sweep / model / f"{model}_{elrs}_{lr}_{seed}" / "summary.json"
        ba = load_ba(run)
        if ba is not None:
            vals.append(ba)
    m, std_n, std_nm1 = sample_stats(vals)
    rows.append((f"F05 {tag} best FT", len(vals), m, std_n, std_nm1, f"sweep/{model}/*_{lr}_{elrs}"))

# ---- F05 LaBraM canonical FT (lr=1e-5, elrs=0.1) ----
vals = []
for seed in ["s42", "s123", "s2024"]:
    run = sweep / "labram" / f"labram_encoderlrscale0.1_lr1e-5_{seed}" / "summary.json"
    ba = load_ba(run)
    if ba is not None:
        vals.append(ba)
m, std_n, std_nm1 = sample_stats(vals)
rows.append(("F05/F06 labram canonical FT", len(vals), m, std_n, std_nm1, "sweep/labram/*_lr1e-5_elrs0.1"))

# ---- F06 labram null FT ----
null_root = ROOT / "results/studies/exp03_stress_erosion/ft_null"
null_vals = []
if null_root.exists():
    for run in null_root.iterdir():
        ba = load_ba(run / "summary.json")
        if ba is not None:
            null_vals.append(ba)
m, std_n, std_nm1 = sample_stats(null_vals)
rows.append(("F06 labram null FT (10-perm)", len(null_vals), m, std_n, std_nm1, str(null_root)))

# ---- F19 CBraMod/REVE null FT ----
for model in ["cbramod", "reve"]:
    null_root = ROOT / f"results/studies/exp03_stress_erosion/ft_null_{model}"
    null_vals = []
    if null_root.exists():
        for run in null_root.iterdir():
            ba = load_ba(run / "summary.json")
            if ba is not None:
                null_vals.append(ba)
    m, std_n, std_nm1 = sample_stats(null_vals)
    rows.append((f"F19 {model} null FT", len(null_vals), m, std_n, std_nm1, str(null_root)))

# ---- F09 EEGMAT LaBraM FT ----
emat = ROOT / "results/studies/exp04_eegmat_feat_multiseed"
vals = []
for seed in ["s42_llrd1.0", "s123_llrd1.0", "s2024_llrd1.0"]:
    ba = load_ba(emat / seed / "summary.json")
    if ba is not None:
        vals.append(ba)
m, std_n, std_nm1 = sample_stats(vals)
rows.append(("F09 EEGMAT labram FT", len(vals), m, std_n, std_nm1, str(emat)))

# ---- F20 non-FM baselines ----
nfm = ROOT / "results/studies/exp15_nonfm_baselines/sweep"
for cfg in ["shallowconvnet_lr1e-4", "eegnet_lr5e-4"]:
    vals = []
    for seed in ["s42", "s123", "s2024"]:
        ba = load_ba(nfm / f"{cfg}_{seed}" / "summary.json")
        if ba is not None:
            vals.append(ba)
    m, std_n, std_nm1 = sample_stats(vals)
    rows.append((f"F20 {cfg}", len(vals), m, std_n, std_nm1, str(nfm)))

# ---- F17 pp delta (representation-level, from variance summaries) ----
for dataset_exp in [("adftd", "exp07_adftd_multiseed"), ("tdbrain", "exp08_tdbrain_multiseed")]:
    ds, exp = dataset_exp
    summ_path = ROOT / f"results/studies/{exp}/multiseed_variance_summary.json"
    if summ_path.exists():
        with open(summ_path) as f:
            summ = json.load(f)
        # Structure may have per-model deltas already aggregated
        for model in ["labram", "cbramod", "reve"]:
            if model in summ:
                entry = summ[model]
                # try a few common keys for per-seed delta
                per_seed = entry.get("per_seed_delta") or entry.get("delta_per_seed")
                if per_seed:
                    if isinstance(per_seed, dict):
                        vals = list(per_seed.values())
                    else:
                        vals = list(per_seed)
                    m, std_n, std_nm1 = sample_stats(vals)
                    rows.append((f"F17 {model}×{ds} pp Δ", len(vals), m, std_n, std_nm1, str(summ_path)))

# ---- Report ----
print(f"{'Label':<35} {'n':>3}  {'mean':>8}  {'std(n)':>8}  {'std(n-1)':>9}  {'note':<45}")
print("-" * 125)
for label, n, m, sn, snm1, src in rows:
    if m is None:
        print(f"{label:<35} {n:>3}  {'—':>8}  {'—':>8}  {'—':>9}  no data at {src}")
        continue
    print(f"{label:<35} {n:>3}  {m:>8.4f}  {sn:>8.4f}  {snm1:>9.4f}  {src[-45:]}")
