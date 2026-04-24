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
    lp      — per-window frozen LP 8-seed results (pass-through + provenance)
    ft      — FT per-FM × 3-seed provenance stamps (summary.json already in place)

Sections pending (TODO):
    classical, nonfm_deep, perm_null, fooof_ablation, band_stop, cross_cell.
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
# Orchestrator
# ---------------------------------------------------------------------------
SECTIONS = {
    "lp": rebuild_lp,
    "ft": rebuild_ft_provenance,
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
