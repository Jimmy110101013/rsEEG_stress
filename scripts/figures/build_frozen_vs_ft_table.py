"""Build the complete Model × Dataset frozen-LP + FT BA table.

Frozen LP: compute subject-level StratifiedGroupKFold(5) LogReg BA from
cached npz features for ADFTD/TDBRAIN/EEGMAT. Reuse F05 8-seed numbers for
Stress (already in findings.md).

FT: aggregate per-seed FT BA from exp07/exp08 (ADFTD/TDBRAIN 3-seed),
exp04 (EEGMAT LaBraM), and findings.md F05 best-HP (Stress 3-seed).

All std values use sample convention (n-1 divisor).

Output:
- paper/figures/source_tables/master_frozen_ft_table.json (numeric)
- paper/figures/source_tables/master_frozen_ft_table.md (readable)
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = [42, 123, 2024]  # 3-seed for uniformity with FT

# -------- Frozen LP compute --------

def frozen_lp_ba(npz_path: str, seed: int) -> float:
    d = np.load(npz_path, allow_pickle=True)
    F, y, g = d["features"], d["labels"], d["patient_ids"]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    y_pred = np.zeros_like(y)
    for tr, te in cv.split(F, y, groups=g):
        clf = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
        ])
        clf.fit(F[tr], y[tr])
        y_pred[te] = clf.predict(F[te])
    return float(balanced_accuracy_score(y, y_pred))


def agg(vals):
    if not vals:
        return None, None
    m = mean(vals)
    s = stdev(vals) if len(vals) > 1 else 0.0  # sample std (n-1)
    return m, s


# -------- Frozen LP: ADFTD, TDBRAIN, EEGMAT (3-seed) --------

frozen = {}
for model in ["labram", "cbramod", "reve"]:
    for ds in ["adftd", "tdbrain", "eegmat"]:
        npz = f"results/features_cache/frozen_{model}_{ds}_19ch.npz"
        if not Path(npz).exists():
            frozen[(model, ds)] = (None, None, 0)
            continue
        bas = [frozen_lp_ba(npz, s) for s in SEEDS]
        m, s = agg(bas)
        frozen[(model, ds)] = (m, s, len(bas))
        print(f"  frozen {model:<8} {ds:<8}: {m:.4f} ± {s:.4f}  (n={len(bas)})")

# -------- Frozen LP: Stress (use F05 8-seed from existing JSON) --------

stress_frozen = {}
for model in ["labram", "cbramod", "reve"]:
    f = f"results/studies/exp03_stress_erosion/frozen_lp/{model}_multi_seed.json"
    if not Path(f).exists():
        f = f"results/studies/exp03_stress_erosion/frozen_lp/multi_seed.json"  # labram old
    with open(f) as fh:
        d = json.load(fh)
    psb = d.get("per_seed_ba") or d.get("per_seed") or {}
    vals = [v for v in psb.values() if isinstance(v, (int, float))]
    m, s = agg(vals)
    stress_frozen[model] = (m, s, len(vals))
    print(f"  frozen {model:<8} stress  : {m:.4f} ± {s:.4f}  (n={len(vals)})")

# -------- FT: ADFTD/TDBRAIN from exp07/exp08 per-seed summaries --------

def agg_ft(exp_dir, model):
    vals = []
    for seed in ["s42", "s123", "s2024"]:
        p = Path(exp_dir) / f"{model}_{seed}" / "summary.json"
        if not p.exists():
            continue
        with open(p) as fh:
            s = json.load(fh)
        ba = s.get("subject_bal_acc") or s.get("mean_bal_acc")
        if ba is not None:
            vals.append(ba)
    m, sd = agg(vals)
    return m, sd, len(vals)

ft = {}
for ds, exp in [("adftd", "results/studies/exp07_adftd_multiseed"),
                ("tdbrain", "results/studies/exp08_tdbrain_multiseed")]:
    for model in ["labram", "cbramod", "reve"]:
        m, s, n = agg_ft(exp, model)
        ft[(model, ds)] = (m, s, n)
        if m is not None:
            print(f"  FT     {model:<8} {ds:<8}: {m:.4f} ± {s:.4f}  (n={n})")

# -------- FT: EEGMAT all three FMs --------
# LaBraM from exp04 (s{seed}_llrd1.0 naming), CBraMod/REVE from exp17 (s{seed}).

emat_ft = {}
for model in ["labram", "cbramod", "reve"]:
    vals = []
    if model == "labram":
        paths = [Path(f"results/studies/exp04_eegmat_feat_multiseed/s{s}_llrd1.0/summary.json")
                 for s in [42, 123, 2024]]
    else:
        paths = [Path(f"results/studies/exp17_eegmat_cbramod_reve_ft/{model}_s{s}/summary.json")
                 for s in [42, 123, 2024]]
    for p in paths:
        if p.exists():
            with open(p) as fh:
                ba = json.load(fh).get("subject_bal_acc")
            if ba is not None:
                vals.append(ba)
    m, s = agg(vals)
    emat_ft[model] = (m, s, len(vals))
    if m is not None:
        print(f"  FT     {model:<8} eegmat  : {m:.4f} ± {s:.4f}  (n={len(vals)})")

# -------- FT: Stress best-HP from hp_sweep --------

best = {"labram": ("lr1e-4", "encoderlrscale1.0"),
        "cbramod": ("lr1e-5", "encoderlrscale0.1"),
        "reve": ("lr3e-5", "encoderlrscale0.1")}
stress_ft = {}
for model, (lr, elrs) in best.items():
    vals = []
    for seed in ["s42", "s123", "s2024"]:
        p = Path(f"results/hp_sweep/20260410_dass/{model}/{model}_{elrs}_{lr}_{seed}/summary.json")
        if p.exists():
            with open(p) as fh:
                ba = json.load(fh).get("subject_bal_acc")
            if ba is not None:
                vals.append(ba)
    m, s = agg(vals)
    stress_ft[model] = (m, s, len(vals))
    print(f"  FT     {model:<8} stress  : {m:.4f} ± {s:.4f}  (n={len(vals)})")

# -------- Assemble & emit --------

datasets = ["stress", "adftd", "tdbrain", "eegmat"]
dataset_label = {"stress": "Stress", "adftd": "ADFTD", "tdbrain": "TDBRAIN", "eegmat": "EEGMAT"}
models = ["labram", "cbramod", "reve"]
model_label = {"labram": "LaBraM", "cbramod": "CBraMod", "reve": "REVE"}

table = {}
for m in models:
    table[m] = {}
    for ds in datasets:
        if ds == "stress":
            fr = stress_frozen[m]
            fti = stress_ft[m]
        elif ds == "eegmat":
            fr = frozen[(m, ds)]
            fti = emat_ft.get(m, (None, None, 0))
        else:
            fr = frozen[(m, ds)]
            fti = ft[(m, ds)]
        table[m][ds] = {"frozen": fr, "ft": fti}

# JSON
out_json = "paper/figures/source_tables/master_frozen_ft_table.json"
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(
        {"models": models, "datasets": datasets, "table": {
            m: {ds: {"frozen_mean": table[m][ds]["frozen"][0],
                     "frozen_std":  table[m][ds]["frozen"][1],
                     "frozen_n":    table[m][ds]["frozen"][2],
                     "ft_mean":     table[m][ds]["ft"][0],
                     "ft_std":      table[m][ds]["ft"][1],
                     "ft_n":        table[m][ds]["ft"][2]}
                 for ds in datasets} for m in models}},
        f, indent=2,
    )
print(f"\nWrote {out_json}")

# Markdown
def fmt(cell):
    m, s, n = cell
    if m is None:
        return "—"
    return f"{m:.3f} ± {s:.3f} (n={n})"

md = []
md.append("# Master table: Frozen LP vs Fine-Tuned BA\n")
md.append("All values are subject-level StratifiedGroupKFold(5) balanced accuracy,")
md.append("sample std (n−1 divisor). Stress uses per-recording DASS; ADFTD/TDBRAIN/EEGMAT")
md.append("use their native labels. Generated by `scripts/build_frozen_vs_ft_table.py`.\n")

md.append("## Unified table (frozen and FT interleaved per model)\n")
md.append("| Model | Phase | " + " | ".join(dataset_label[d] for d in datasets) + " |")
md.append("|---" * (len(datasets)+2) + "|")
for m in models:
    row_frozen = [model_label[m], "Frozen LP"]
    row_ft     = ["",             "Fine-tuned"]
    for ds in datasets:
        row_frozen.append(fmt(table[m][ds]["frozen"]))
        row_ft.append(fmt(table[m][ds]["ft"]))
    md.append("| " + " | ".join(row_frozen) + " |")
    md.append("| " + " | ".join(row_ft) + " |")

md.append("\n## Δ (FT − Frozen), pp\n")
md.append("| Model | " + " | ".join(dataset_label[d] for d in datasets) + " |")
md.append("|---" * (len(datasets)+1) + "|")
for m in models:
    row = [model_label[m]]
    for ds in datasets:
        fr = table[m][ds]["frozen"]
        fti = table[m][ds]["ft"]
        if fr[0] is not None and fti[0] is not None:
            delta = (fti[0] - fr[0]) * 100
            row.append(f"{delta:+.1f}")
        else:
            row.append("—")
    md.append("| " + " | ".join(row) + " |")

md.append("\n## Dataset metadata\n")
md.append("| Dataset | Task | N subjects | N recordings | Labels |")
md.append("|---|---|---|---|---|")
md.append("| Stress  | DASS stress state        | 17  | 70  | 14 pos / 56 neg (per-recording) |")
md.append("| ADFTD   | AD vs HC                 | 65  | 195 | Binary |")
md.append("| TDBRAIN | MDD vs HC                | 359 | 734 | Binary |")
md.append("| EEGMAT  | rest vs arithmetic task  | 36  | 72  | 36+36 (paired within-subject) |")

out_md = "paper/figures/source_tables/master_frozen_ft_table.md"
with open(out_md, "w") as f:
    f.write("\n".join(md) + "\n")
print(f"Wrote {out_md}")
