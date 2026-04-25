"""Rebuild results/final/ snapshots from the raw exp##_* layer.

One command reproduces the paper-surface JSONs. Every snapshot carries a
`provenance` field pointing back to its raw source + commit.

Usage:
    PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
    $PY scripts/analysis/rebuild_final_snapshots.py --all
    $PY scripts/analysis/rebuild_final_snapshots.py --section lp
    $PY scripts/analysis/rebuild_final_snapshots.py --cell adftd --section ft
    $PY scripts/analysis/rebuild_final_snapshots.py --dry-run --all

Sections implemented:
    lp                            — per-window frozen LP 8-seed (pass-through + provenance)
    ft                            — FT per-FM × 3-seed provenance stamps
    perm_null                     — exp27 paired-null aggregated to one JSON / cell
    classical                     — exp02 classical baselines (pass-through + provenance)
    fooof_ablation                — fooof_ablation/<ds>_probes (pass-through + provenance)
    subject_probe_temporal_block  — exp33 temporal-block probe (pass-through + provenance)
    band_stop                     — per-cell slice of exp14 band_stop_ablation

Sections pending (TODO):
    nonfm_deep, variance_triangulation, cross_cell.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FINAL = REPO / "results" / "final"
STUDIES = REPO / "results" / "studies"

CELLS = ["adftd", "eegmat", "sleepdep", "stress"]  # tdbrain handled separately for App A
MODELS = ["labram", "cbramod", "reve"]


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------
def current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO
        ).decode().strip()
    except Exception:
        return "unknown"


def _provenance(raw_dir: str, notes: str = "") -> dict:
    from datetime import date
    return {
        "raw_dir": raw_dir,
        "snapshot_date": str(date.today()),
        "commit": current_commit(),
        "script": "scripts/analysis/rebuild_final_snapshots.py",
        "notes": notes,
    }


def _write(path: Path, payload: dict, dry: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"  [dry] would write {path.relative_to(REPO)} "
              f"({len(json.dumps(payload))} bytes)")
        return
    path.write_text(json.dumps(payload, indent=2))
    print(f"  → {path.relative_to(REPO)}")


# ---------------------------------------------------------------------------
# Section: LP
# ---------------------------------------------------------------------------
def rebuild_lp(cell: str, model: str, dry: bool = False):
    src = STUDIES / "perwindow_lp_all" / cell / f"{model}_multi_seed.json"
    if not src.exists():
        print(f"  skip lp {cell}/{model}: {src.relative_to(REPO)} missing")
        return
    raw = json.loads(src.read_text())
    payload = {
        "provenance": _provenance(
            raw_dir=str(src.parent.relative_to(REPO)),
            notes="Per-window frozen LP, 8-seed sklearn LogReg + test-prob mean-pool.",
        ),
        **raw,
    }
    _write(FINAL / cell / "lp" / f"{model}.json", payload, dry)


# ---------------------------------------------------------------------------
# Section: FT provenance stamps
# ---------------------------------------------------------------------------
def rebuild_ft_provenance(cell: str, model: str, dry: bool = False):
    ft_dir = FINAL / cell / "ft" / model
    if not ft_dir.exists():
        print(f"  skip ft {cell}/{model}: {ft_dir.relative_to(REPO)} missing")
        return
    for seed_dir in sorted(ft_dir.glob("seed*")):
        config_path = seed_dir / "config.json"
        if not config_path.exists():
            print(f"  skip {seed_dir.relative_to(REPO)}: no config.json")
            continue
        cfg = json.loads(config_path.read_text())
        hp = {k: cfg.get(k) for k in [
            "lr", "wd", "batch_size", "window_sec", "norm", "label",
            "n_splits", "layer_decay", "encoder_lr_scale", "loss",
            "dataset",
        ] if k in cfg}
        payload = {
            "provenance": _provenance(
                raw_dir=str(seed_dir.relative_to(REPO)),
                notes=(
                    f"FT run for {cell} × {model}. Per-FM canonical HP "
                    f"(see docs/methodology_notes.md G-F09)."
                ),
            ),
            "hp": hp,
            "model": model,
            "cell": cell,
            "seed": seed_dir.name,
        }
        _write(seed_dir / "provenance.json", payload, dry)


# ---------------------------------------------------------------------------
# Section: perm_null (aggregate 30 per-seed summaries → 1 JSON per cell)
# ---------------------------------------------------------------------------
def rebuild_perm_null(cell: str, model: str, dry: bool = False):
    if model != "labram":
        return  # only LaBraM null chains exist per exp27 design
    src_dir = STUDIES / "exp27_paired_null" / cell
    if not src_dir.exists():
        print(f"  skip perm_null {cell}: {src_dir.relative_to(REPO)} missing")
        return
    seeds = sorted(src_dir.glob("perm_s*/summary.json"))
    if not seeds:
        print(f"  skip perm_null {cell}: no perm_s*/summary.json under {src_dir.relative_to(REPO)}")
        return
    bas, per_seed = [], {}
    for p in seeds:
        d = json.loads(p.read_text())
        seed_id = p.parent.name.replace("perm_", "")  # e.g. "s0"
        per_seed[seed_id] = float(d["subject_bal_acc"])
        bas.append(per_seed[seed_id])
    import numpy as np
    payload = {
        "provenance": _provenance(
            raw_dir=str(src_dir.relative_to(REPO)),
            notes=f"30-seed LaBraM FT permutation null for {cell}; aggregated.",
        ),
        "model": model,
        "cell": cell,
        "n_seeds": len(bas),
        "subject_bal_acc_per_seed": per_seed,
        "subject_bal_acc_mean": float(np.mean(bas)),
        "subject_bal_acc_std_ddof1": float(np.std(bas, ddof=1)) if len(bas) > 1 else 0.0,
        "subject_bal_acc_min": float(np.min(bas)),
        "subject_bal_acc_max": float(np.max(bas)),
    }
    _write(FINAL / cell / "perm_null" / f"{model}_null.json", payload, dry)


# ---------------------------------------------------------------------------
# Section: classical (exp02, pass-through + provenance)
# ---------------------------------------------------------------------------
def rebuild_classical(cell: str, model: str, dry: bool = False):
    if model != "labram":
        return  # classical baselines are model-agnostic; one snapshot per cell
    src = STUDIES / "exp02_classical_dass" / cell / "summary.json"
    if not src.exists():
        print(f"  skip classical {cell}: {src.relative_to(REPO)} missing")
        return
    raw = json.loads(src.read_text())
    payload = {
        "provenance": _provenance(
            raw_dir=str(src.parent.relative_to(REPO)),
            notes="Classical (LogReg/SVM/RF/XGB) multi-seed baseline.",
        ),
        **raw,
    }
    _write(FINAL / cell / "classical" / "summary.json", payload, dry)


# ---------------------------------------------------------------------------
# Section: fooof_ablation (per-cell pass-through + provenance)
# ---------------------------------------------------------------------------
def rebuild_fooof_ablation(cell: str, model: str, dry: bool = False):
    if model != "labram":
        return  # cell-level (all FMs in one file)
    src = STUDIES / "fooof_ablation" / f"{cell}_probes.json"
    if not src.exists():
        print(f"  skip fooof_ablation {cell}: {src.relative_to(REPO)} missing")
        return
    raw = json.loads(src.read_text())
    payload = {
        "provenance": _provenance(
            raw_dir=str(src.relative_to(REPO)),
            notes="FOOOF {aperiodic,periodic,both}_removed → re-extract → state probe.",
        ),
        **raw,
    }
    _write(FINAL / cell / "fooof_ablation" / "probes.json", payload, dry)


# ---------------------------------------------------------------------------
# Section: subject_probe_temporal_block (exp33, pass-through + provenance)
# ---------------------------------------------------------------------------
def rebuild_subject_probe_temporal_block(cell: str, model: str, dry: bool = False):
    if model != "labram":
        return  # cell-level
    src = STUDIES / "exp33_temporal_block_probe" / f"{cell}_probes.json"
    if not src.exists():
        print(f"  skip subject_probe {cell}: {src.relative_to(REPO)} missing")
        return
    raw = json.loads(src.read_text())
    payload = {
        "provenance": _provenance(
            raw_dir=str(src.relative_to(REPO)),
            notes="Temporal-block subject-ID probe (uniform across 4 cells).",
        ),
        **raw,
    }
    _write(FINAL / cell / "subject_probe_temporal_block" / "probes.json",
           payload, dry)


# ---------------------------------------------------------------------------
# Section: band_stop (slice cross-cell exp14 JSON into per-cell snapshots)
# ---------------------------------------------------------------------------
def rebuild_band_stop(cell: str, model: str, dry: bool = False):
    if model != "labram":
        return  # cell-level (all FMs in one slice)
    src = STUDIES / "exp14_channel_importance" / "band_stop_ablation.json"
    if not src.exists():
        print(f"  skip band_stop {cell}: {src.relative_to(REPO)} missing")
        return
    full = json.loads(src.read_text())
    if cell not in full:
        print(f"  skip band_stop {cell}: not present in {src.relative_to(REPO)}")
        return
    payload = {
        "provenance": _provenance(
            raw_dir=str(src.relative_to(REPO)),
            notes=f"Band-stop cosine-distance probes for {cell} (sliced from cross-cell file).",
        ),
        "cell": cell,
        "bands": full[cell],
    }
    _write(FINAL / cell / "band_stop" / "probes.json", payload, dry)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
SECTIONS = {
    "lp": rebuild_lp,
    "ft": rebuild_ft_provenance,
    "perm_null": rebuild_perm_null,
    "classical": rebuild_classical,
    "fooof_ablation": rebuild_fooof_ablation,
    "subject_probe_temporal_block": rebuild_subject_probe_temporal_block,
    "band_stop": rebuild_band_stop,
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", choices=CELLS + ["all"], default="all")
    p.add_argument("--section", choices=list(SECTIONS.keys()) + ["all"],
                   default="all")
    p.add_argument("--dry-run", action="store_true",
                   help="Print actions without writing files")
    args = p.parse_args()

    cells = CELLS if args.cell == "all" else [args.cell]
    sections = list(SECTIONS.keys()) if args.section == "all" else [args.section]

    for sec in sections:
        print(f"=== {sec} ===")
        fn = SECTIONS[sec]
        for cell in cells:
            for model in MODELS:
                fn(cell, model, dry=args.dry_run)

    print("\ndone.")


if __name__ == "__main__":
    main()
