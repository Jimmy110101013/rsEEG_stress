"""Aggregate all LP / FT / classical / non-FM results across datasets.

Output:
  docs/master_results_table.md   — human-readable, for decision making
  paper/tables/_source/master_results.json  — machine-readable, for later figures
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/raid/jupyter-linjimmy1003.md10/UCSD_stress")

# dataset → (pretty, n_rec, n_subj, label, quadrant_2x2)
# 2×2 factorial: (within- vs between-subject) × (coherent vs absent/incoherent label signal)
# TDBRAIN dropped (duplicates ADFTD cell; supplementary only).
DATASETS = {
    "stress":   ("Stress (DASS)",  70,  17, "per-recording DASS binary (14/17 subjects consistent)",  "between-subject, absent signal"),
    "adftd":    ("ADFTD",          82,  82, "subject-level AD/FTD vs HC",                              "between-subject, coherent signal"),
    "sleepdep": ("SleepDep",       72,  36, "within-subject rested vs SD",                             "within-subject, incoherent signal"),
    "eegmat":   ("EEGMAT",         72,  36, "within-subject rest vs arithmetic",                       "within-subject, coherent signal"),
}
FMS = ["labram", "cbramod", "reve"]
SEEDS = [42, 123, 2024]


# ───────────────────────── helpers ─────────────────────────
def _load_json(p: Path) -> dict | None:
    if p.exists():
        try: return json.loads(p.read_text())
        except Exception: return None
    return None


def lp_row(ds: str, fm: str) -> dict | None:
    p = REPO / f"results/studies/perwindow_lp_all/{ds}/{fm}_multi_seed.json"
    d = _load_json(p)
    if d is None: return None
    return {
        "mean": float(d["mean_3seed_42_123_2024"]),
        "std":  float(d["std_3seed_42_123_2024_ddof1"]),
        "n_seeds": 3,
        "source": str(p.relative_to(REPO)),
    }


# ─── FT locations per dataset ───
FT_LOCATIONS = {
    "stress":   {
        "labram":  [f"results/studies/exp05_stress_feat_multiseed/s{s}_llrd1.0/summary.json"  for s in SEEDS],
        "cbramod": [f"results/features_cache/ft_cbramod_stress/summary.json"],
        "reve":    [f"results/features_cache/ft_reve_stress/summary.json"],
    },
    "sleepdep": {fm: [f"results/studies/exp_newdata/sleepdep_ft_{fm}_s{s}/summary.json" for s in SEEDS] for fm in FMS},
    "eegmat":   {
        "labram":  [f"results/studies/exp04_eegmat_feat_multiseed/s{s}_llrd1.0/summary.json" for s in SEEDS],
        "cbramod": [f"results/features_cache/ft_cbramod_eegmat/summary.json"],
        "reve":    [f"results/features_cache/ft_reve_eegmat/summary.json"],
    },
    "adftd":    {fm: [f"results/studies/exp07_adftd_multiseed/{fm}_s{s}/summary.json"    for s in SEEDS] for fm in FMS},
}


def ft_row(ds: str, fm: str) -> dict | None:
    paths = [REPO / p for p in FT_LOCATIONS[ds][fm]]
    vals, cfgs, sources = [], [], []
    for p in paths:
        d = _load_json(p)
        if d is None: continue
        ba = d.get("subject_bal_acc")
        if ba is None: continue
        vals.append(float(ba))
        sources.append(str(p.relative_to(REPO)))
        # pull config.json next to summary
        c = _load_json(p.parent / "config.json")
        if c: cfgs.append(c)
    if not vals: return None
    row: dict[str, Any] = {
        "mean":     float(np.mean(vals)),
        "std":      float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
        "n_seeds":  len(vals),
        "sources":  sources,
    }
    if cfgs:
        c0 = cfgs[0]
        row["hp"] = {k: c0.get(k) for k in
                     ["lr", "llrd", "encoder_lr_scale", "weight_decay",
                      "norm", "window_sec", "epochs", "patience", "batch_size"]}
    return row


# ─── Classical ML (new multi-seed format) ───
def classical_row(ds: str) -> dict | None:
    p = REPO / f"results/studies/exp02_classical_dass/{ds}/summary.json"
    d = _load_json(p)
    if d is None: return None
    agg = d.get("aggregated")
    if agg is None:
        # legacy single-seed (Stress old run)
        models = d.get("models") or {}
        out = {}
        for name, rec in models.items():
            ba = rec.get("bal_acc")
            if ba is not None:
                out[name] = {"mean": float(ba), "std": None, "n_seeds": 1}
        return out or None
    return {name: {"mean": agg[name]["mean_bal_acc"], "std": agg[name]["std_bal_acc"],
                   "n_seeds": len(agg[name]["bal_acc_per_seed"])}
            for name in agg}


# ─── Non-FM deep (eegnet, shallowconvnet) ───
def nonfm_row(ds: str) -> dict | None:
    base_by_ds = {
        "stress":   REPO / "results/studies/exp15_nonfm_baselines/sweep",
        "eegmat":   REPO / "results/studies/exp15_nonfm_baselines/eegmat",
        "sleepdep": REPO / "results/studies/exp15_nonfm_baselines/sleepdep",
    }
    if ds not in base_by_ds: return None
    base = base_by_ds[ds]
    if not base.exists(): return None
    out = {}
    for arch in ["eegnet", "shallowconvnet"]:
        vals = []
        for sub in base.glob(f"{arch}_*_s*"):
            d = _load_json(sub / "summary.json")
            if d and "subject_bal_acc" in d:
                vals.append(float(d["subject_bal_acc"]))
        if vals:
            out[arch] = {
                "mean":    float(np.mean(vals)),
                "std":     float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "n_seeds": len(vals),
            }
    return out or None


# ───────────────────────── build ─────────────────────────
def main():
    out: dict = {"datasets": {}}
    for ds in DATASETS:
        lp = {fm: lp_row(ds, fm) for fm in FMS}
        ft = {fm: ft_row(ds, fm) for fm in FMS}
        cl = classical_row(ds)
        nf = nonfm_row(ds)
        out["datasets"][ds] = {
            "pretty": DATASETS[ds][0], "n_rec": DATASETS[ds][1], "n_subj": DATASETS[ds][2],
            "label":  DATASETS[ds][3], "quadrant": DATASETS[ds][4],
            "fm_lp":  lp, "fm_ft": ft,
            "classical": cl, "nonfm": nf,
        }

    # JSON
    src = REPO / "paper/tables/_source/master_results.json"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(json.dumps(out, indent=2))
    print(f"json → {src.relative_to(REPO)}")

    # Markdown
    lines: list[str] = ["# Master Results Table",
                        "",
                        "Auto-generated by `scripts/analysis/build_master_results_table.py`.",
                        "All BA = subject-level 5-fold CV, 3 seeds unless flagged `n=1`.",
                        ""]

    def fmt(r):
        if r is None: return "—"
        m = r["mean"]
        if r.get("std") is None: return f"{m:.3f}¹"
        return f"{m:.3f} ± {r['std']:.3f}"

    # Per-dataset detail
    for ds, meta in out["datasets"].items():
        lines.append(f"## {meta['pretty']}")
        lines.append(f"- Recordings / Subjects: **{meta['n_rec']} / {meta['n_subj']}**")
        lines.append(f"- Label: {meta['label']}")
        lines.append(f"- Regime: *{meta['quadrant']}*")
        lines.append("")
        lines.append("| FM | LP (frozen, 3-seed) | FT (3-seed or n=1¹) | FT HP |")
        lines.append("|---|---|---|---|")
        for fm in FMS:
            lp = meta["fm_lp"].get(fm); ft = meta["fm_ft"].get(fm)
            hp = (ft or {}).get("hp") or {}
            hp_s = (
                f"lr={hp.get('lr','?')}, wd={hp.get('weight_decay','?')}, "
                f"enc_scale={hp.get('encoder_lr_scale','?')}, norm={hp.get('norm','?')}"
                if hp else "—"
            )
            ft_s = fmt(ft)
            if ft and ft.get("n_seeds") == 1: ft_s = f"{ft['mean']:.3f}¹"
            lines.append(f"| {fm} | {fmt(lp)} | {ft_s} | {hp_s} |")
        lines.append("")

        # classical
        if meta["classical"]:
            lines.append("**Classical ML** (per-rec features, 3 seeds):")
            lines.append("")
            lines.append("| Model | BA |")
            lines.append("|---|---|")
            for name, r in meta["classical"].items():
                lines.append(f"| {name} | {fmt(r)} |")
            lines.append("")
        if meta["nonfm"]:
            lines.append("**Non-FM deep** (from-scratch, 3 seeds):")
            lines.append("")
            lines.append("| Arch | BA |")
            lines.append("|---|---|")
            for arch, r in meta["nonfm"].items():
                lines.append(f"| {arch} | {fmt(r)} |")
            lines.append("")

    # Cross-dataset summary
    lines.append("---")
    lines.append("## 2×2 quick-view (subject-level BA, 3-seed mean)")
    lines.append("")
    lines.append("| Dataset | Regime | n_rec / n_subj | LaBraM LP | LaBraM FT | CBraMod LP | CBraMod FT | REVE LP | REVE FT | Classical best | Non-FM best |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for ds, meta in out["datasets"].items():
        def cell(cat, fm):
            r = meta[cat].get(fm)
            return fmt(r) if r else "—"
        cl = meta["classical"] or {}
        nf = meta["nonfm"] or {}
        cl_best = max(cl.items(), key=lambda kv: kv[1]["mean"], default=(None, None))
        nf_best = max(nf.items(), key=lambda kv: kv[1]["mean"], default=(None, None))
        cl_s = f"{cl_best[0]} {fmt(cl_best[1])}" if cl_best[0] else "—"
        nf_s = f"{nf_best[0]} {fmt(nf_best[1])}" if nf_best[0] else "—"
        lines.append(f"| {meta['pretty']} | {meta['quadrant']} | {meta['n_rec']}/{meta['n_subj']} | "
                     f"{cell('fm_lp','labram')} | {cell('fm_ft','labram')} | "
                     f"{cell('fm_lp','cbramod')} | {cell('fm_ft','cbramod')} | "
                     f"{cell('fm_lp','reve')} | {cell('fm_ft','reve')} | {cl_s} | {nf_s} |")
    lines.append("")
    lines.append("¹ = single-seed (reproduction pending); needs proper multi-seed with per-FM canonical recipe before publication.")
    lines.append("")

    md_path = REPO / "docs/master_results_table.md"
    md_path.write_text("\n".join(lines))
    print(f"md   → {md_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
