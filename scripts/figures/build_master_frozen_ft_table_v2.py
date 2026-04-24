"""Build master Frozen-LP vs FT table (v2, 2026-04-20).

Uses:
- Frozen LP: per-window protocol (matched to FT's per-window training + pooling).
  Source: results/studies/perwindow_lp_all/{dataset}/{model}_multi_seed.json
  Seeds: 8 (42, 123, 2024, 7, 0, 1, 99, 31337)

- FT: canonical per-FM recipe (NOT per-dataset best-HP). Recipe = LaBraM
  {lr=1e-5, encoder_lr_scale=0.1, norm=zscore}; CBraMod {lr=1e-5, elrs=0.1, none};
  REVE {lr=3e-5, elrs=0.1, none}. Sources:
    stress LaBraM   → exp03_stress_erosion/ft_real/s{seed}_llrd1.0 (canonical lr=1e-5)
    stress CBraMod  → hp_sweep/20260410_dass/cbramod/cbramod_encoderlrscale0.1_lr1e-5_s{seed}
    stress REVE     → hp_sweep/20260410_dass/reve/reve_encoderlrscale0.1_lr3e-5_s{seed}
    adftd/tdbrain   → exp07/exp08 multiseed
    eegmat LaBraM   → exp04 multiseed
    eegmat CBraMod/REVE → exp17
  Seeds: 3 (42, 123, 2024).

Outputs:
- paper/figures/source_tables/master_frozen_ft_table_v2.md
- paper/figures/source_tables/master_frozen_ft_table_v2.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FT_PATHS = {
    ("stress",  "labram"):  [f"results/studies/exp03_stress_erosion/ft_real/s{s}_llrd1.0/summary.json" for s in [42, 123, 2024]],
    ("stress",  "cbramod"): [f"results/hp_sweep/20260410_dass/cbramod/cbramod_encoderlrscale0.1_lr1e-5_s{s}/summary.json" for s in [42, 123, 2024]],
    ("stress",  "reve"):    [f"results/hp_sweep/20260410_dass/reve/reve_encoderlrscale0.1_lr3e-5_s{s}/summary.json" for s in [42, 123, 2024]],
    ("eegmat",  "labram"):  [f"results/studies/exp04_eegmat_feat_multiseed/s{s}_llrd1.0/summary.json" for s in [42, 123, 2024]],
    ("eegmat",  "cbramod"): [f"results/studies/exp17_eegmat_cbramod_reve_ft/cbramod_s{s}/summary.json" for s in [42, 123, 2024]],
    ("eegmat",  "reve"):    [f"results/studies/exp17_eegmat_cbramod_reve_ft/reve_s{s}/summary.json" for s in [42, 123, 2024]],
    ("adftd",   "labram"):  [f"results/studies/exp07_adftd_multiseed/labram_s{s}/summary.json" for s in [42, 123, 2024]],
    ("adftd",   "cbramod"): [f"results/studies/exp07_adftd_multiseed/cbramod_s{s}/summary.json" for s in [42, 123, 2024]],
    ("adftd",   "reve"):    [f"results/studies/exp07_adftd_multiseed/reve_s{s}/summary.json" for s in [42, 123, 2024]],
    ("tdbrain", "labram"):  [f"results/studies/exp08_tdbrain_multiseed/labram_s{s}/summary.json" for s in [42, 123, 2024]],
    ("tdbrain", "cbramod"): [f"results/studies/exp08_tdbrain_multiseed/cbramod_s{s}/summary.json" for s in [42, 123, 2024]],
    ("tdbrain", "reve"):    [f"results/studies/exp08_tdbrain_multiseed/reve_s{s}/summary.json" for s in [42, 123, 2024]],
}

DATASETS = ["stress", "adftd", "tdbrain", "eegmat"]
MODELS = ["labram", "cbramod", "reve"]
DATASET_LABEL = {"stress": "Stress", "adftd": "ADFTD", "tdbrain": "TDBRAIN", "eegmat": "EEGMAT"}
MODEL_LABEL = {"labram": "LaBraM", "cbramod": "CBraMod", "reve": "REVE"}


def load_ft(paths):
    bas = []
    for p in paths:
        if not Path(p).exists():
            continue
        d = json.load(open(p))
        for k in ("subject_bal_acc", "bal_acc"):
            if k in d:
                bas.append(d[k])
                break
    if not bas:
        return None, None, 0
    a = np.array(bas)
    return float(a.mean()), float(a.std(ddof=1) if len(a) > 1 else 0.0), len(a)


def load_lp(dataset, model):
    p = f"results/studies/perwindow_lp_all/{dataset}/{model}_multi_seed.json"
    if not Path(p).exists():
        return None, None, 0
    d = json.load(open(p))
    return float(d["mean_8seed"]), float(d["std_8seed_ddof1"]), 8


# Build table
table = {}
for m in MODELS:
    table[m] = {}
    for ds in DATASETS:
        lp_m, lp_s, lp_n = load_lp(ds, m)
        ft_m, ft_s, ft_n = load_ft(FT_PATHS.get((ds, m), []))
        table[m][ds] = {
            "frozen_mean": lp_m, "frozen_std": lp_s, "frozen_n": lp_n,
            "ft_mean": ft_m, "ft_std": ft_s, "ft_n": ft_n,
            "delta_mean": (ft_m - lp_m) if (lp_m is not None and ft_m is not None) else None,
        }

# JSON
out_json = Path("paper/figures/source_tables/master_frozen_ft_table_v2.json")
out_json.parent.mkdir(parents=True, exist_ok=True)
json.dump(
    {
        "version": "v2",
        "date": "2026-04-20",
        "protocol_notes": {
            "frozen_lp": "per-window LogReg (liblinear, 1-99pct clip, StandardScaler), recording-level StratifiedGroupKFold(5) by patient_id, window-level train, mean-pooled probs per recording → 0.5 threshold, 8 seeds",
            "fine_tuning": "per-FM canonical recipe (LaBraM: lr=1e-5, encoder_lr_scale=0.1, zscore; CBraMod: lr=1e-5, elrs=0.1, none; REVE: lr=3e-5, elrs=0.1, none). All datasets use the same per-FM recipe (no per-dataset HP tuning). 3 seeds (42, 123, 2024).",
            "stress_labram_source": "exp03_stress_erosion/ft_real (canonical lr=1e-5), NOT hp_sweep best-HP (lr=1e-4). Matches the per-FM recipe used on ADFTD/TDBRAIN/EEGMAT.",
        },
        "models": MODELS,
        "datasets": DATASETS,
        "table": table,
    },
    open(out_json, "w"), indent=2,
)
print(f"Wrote {out_json}")

# Markdown
def fmt(m, s, n):
    if m is None:
        return "—"
    return f"{m:.3f} ± {s:.3f} (n={n})"


md = []
md.append("# Master table v2: Frozen LP (per-window) vs FT (canonical per-FM recipe)\n")
md.append("**Date**: 2026-04-20. Replaces `master_frozen_ft_table.md` (v1).\n")
md.append("## Protocol\n")
md.append("- **Frozen LP**: per-window LogReg matched to FT's per-window training + "
          "prediction-pooling; percentile-clip + StandardScaler; `liblinear` solver; "
          "`StratifiedGroupKFold(5)` subject-disjoint; 8 seeds.")
md.append("- **Fine-tuning**: per-FM canonical recipe — LaBraM `(lr=1e-5, encoder_lr_scale=0.1, zscore)`; "
          "CBraMod `(lr=1e-5, elrs=0.1, none)`; REVE `(lr=3e-5, elrs=0.1, none)`. "
          "Same recipe across all 4 datasets (no per-dataset HP tuning). 3 seeds.")
md.append("- **Stress LaBraM FT source**: `exp03_stress_erosion/ft_real/` (canonical lr=1e-5), "
          "**NOT** `hp_sweep/20260410_dass/` best-HP (lr=1e-4). Consistency across "
          "datasets is the criterion.")
md.append("- Balanced accuracy; sample std (ddof=1).\n")

# Unified table (Frozen and FT interleaved per model)
md.append("## Results table\n")
md.append("| Model | Phase | " + " | ".join(DATASET_LABEL[d] for d in DATASETS) + " |")
md.append("|---" * (len(DATASETS) + 2) + "|")
for m in MODELS:
    row_frozen = [MODEL_LABEL[m], "Frozen LP"]
    row_ft = ["", "Fine-tune"]
    for ds in DATASETS:
        cell = table[m][ds]
        row_frozen.append(fmt(cell["frozen_mean"], cell["frozen_std"], cell["frozen_n"]))
        row_ft.append(fmt(cell["ft_mean"], cell["ft_std"], cell["ft_n"]))
    md.append("| " + " | ".join(row_frozen) + " |")
    md.append("| " + " | ".join(row_ft) + " |")

# Delta table
md.append("\n## Δ (FT − Frozen LP), percentage points\n")
md.append("| Model | " + " | ".join(DATASET_LABEL[d] for d in DATASETS) + " |")
md.append("|---" * (len(DATASETS) + 1) + "|")
for m in MODELS:
    row = [MODEL_LABEL[m]]
    for ds in DATASETS:
        d = table[m][ds]["delta_mean"]
        if d is None:
            row.append("—")
        else:
            row.append(f"{d * 100:+.1f}")
    md.append("| " + " | ".join(row) + " |")

# Verdict count
pos = sum(1 for m in MODELS for ds in DATASETS
          if table[m][ds]["delta_mean"] is not None and table[m][ds]["delta_mean"] > 0.01)
neg = sum(1 for m in MODELS for ds in DATASETS
          if table[m][ds]["delta_mean"] is not None and table[m][ds]["delta_mean"] < -0.01)
tied = sum(1 for m in MODELS for ds in DATASETS
           if table[m][ds]["delta_mean"] is not None and abs(table[m][ds]["delta_mean"]) <= 0.01)

md.append(f"\n## Verdict count (12 cells)\n")
md.append(f"- **FT > LP (Δ > +0.01)**: {pos} cells")
md.append(f"- **FT ≈ LP (|Δ| ≤ 0.01)**: {tied} cells")
md.append(f"- **FT < LP (Δ < −0.01)**: {neg} cells")

# Per-cell verdicts
md.append("\n## Per-cell verdicts\n")
md.append("| Cell | Δ | Verdict |")
md.append("|---|---:|---|")
for m in MODELS:
    for ds in DATASETS:
        d = table[m][ds]["delta_mean"]
        if d is None:
            continue
        verdict = "↑ FT rescues" if d > 0.01 else ("↓ FT degrades" if d < -0.01 else "≈ tied")
        md.append(f"| {MODEL_LABEL[m]} × {DATASET_LABEL[ds]} | {d * 100:+.1f} pp | {verdict} |")

md.append("\n## Key differences vs v1\n")
md.append("| Cell | v1 LP (feature-avg 8s) | v2 LP (per-window 8s) | v1 FT | v2 FT | Notes |")
md.append("|---|---:|---:|---:|---:|---|")
md.append("| LaBraM × Stress  | 0.605 ± 0.032 | **0.525 ± 0.040** | 0.524 ± 0.010 | **0.443 ± 0.083** | v1 LP was feat-avg (inflated), FT was hp_sweep best-HP (lr=1e-4) not canonical |")
md.append("| CBraMod × Stress | 0.452 ± 0.032 | **0.430 ± 0.033** | 0.548 ± 0.031 | 0.548 ± 0.031 | LP per-window lower; FT identical |")
md.append("| REVE × Stress    | 0.494 ± 0.018 | **0.441 ± 0.022** | 0.577 ± 0.051 | 0.577 ± 0.051 | LP per-window lower; FT identical |")
md.append("| Others (9 cells) | old 3-seed feat-avg | **new 8-seed per-window** | unchanged | unchanged | LP more rigorous, FT unchanged |\n")

md.append("## Dataset metadata\n")
md.append("| Dataset | Task | N subjects | N recordings | Labels |")
md.append("|---|---|---|---|---|")
md.append("| Stress  | DASS stress state        | 17  | 70  | 14 pos / 56 neg (per-recording) |")
md.append("| ADFTD   | AD vs HC                 | 65  | 195 | 108 pos / 87 neg |")
md.append("| TDBRAIN | MDD vs HC                | 359 | 734 | 640 pos / 94 neg |")
md.append("| EEGMAT  | rest vs arithmetic task  | 36  | 72  | 36 rest / 36 task (paired within-subject) |")

out_md = Path("paper/figures/source_tables/master_frozen_ft_table_v2.md")
out_md.write_text("\n".join(md) + "\n")
print(f"Wrote {out_md}")
