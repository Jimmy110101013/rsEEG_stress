"""Scan results/ and build a unified index of all runs.

Emits two deliverables under results/:
    results_index.csv   — flat, one row per (run × model), machine-readable
    results_index.md    — grouped, human-readable summary

Run:
    python scripts/build_results_index.py
"""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_DIR = os.path.abspath(RESULTS_DIR)

RUN_DIR_RE = re.compile(r"^(\d{8})_(\d{4})_(.+)$")

# Columns for the flat CSV
COLUMNS = [
    "run_id", "date", "time", "paradigm", "dataset", "extractor", "mode",
    "model_variant", "label", "n_folds", "n_samples",
    "subject_bal_acc", "subject_acc", "subject_f1", "subject_kappa",
    "lr", "weight_decay", "epochs", "batch_size", "aug_overlap",
    "has_features", "status", "notes",
]


@dataclass
class RunRow:
    run_id: str
    date: str = ""
    time: str = ""
    paradigm: str = ""          # subject-level | trial-level | classical | cross-dataset
    dataset: str = "stress"
    extractor: str = ""
    mode: str = ""              # ft | lp | lora | classical
    model_variant: str = ""     # for classical baselines: LogReg_L2, RF, ...
    label: str = ""
    n_folds: Optional[int] = None
    n_samples: Optional[int] = None
    subject_bal_acc: Optional[float] = None
    subject_acc: Optional[float] = None
    subject_f1: Optional[float] = None
    subject_kappa: Optional[float] = None
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    aug_overlap: Optional[float] = None
    has_features: bool = False
    status: str = ""            # complete | incomplete | running
    notes: str = ""

    def as_csv(self) -> Dict[str, Any]:
        d = asdict(self)
        # CSV-friendly: round floats
        for k in ("subject_bal_acc", "subject_acc", "subject_f1", "subject_kappa", "aug_overlap"):
            v = d.get(k)
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


def _load_json(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _infer_paradigm(run_name: str, cfg: Optional[dict]) -> str:
    if "classical" in run_name:
        return "classical"
    if run_name.startswith("trial_") or "_trial_" in run_name:
        return "trial-level"
    # cross-dataset feat runs can be either; treat ADFTD/TDBRAIN as cross-dataset
    if cfg and cfg.get("dataset") in ("adftd", "tdbrain"):
        return "cross-dataset"
    return "subject-level"


def _count_folds_and_feats(run_dir: str) -> tuple[int, bool]:
    fold_files = [f for f in os.listdir(run_dir) if f.startswith("curves_fold") and f.endswith(".csv")]
    feat_files = [f for f in os.listdir(run_dir) if f.startswith("fold") and f.endswith("_features.npz")]
    return len(fold_files), len(feat_files) > 0


def _parse_run_id(name: str) -> tuple[str, str, str]:
    m = RUN_DIR_RE.match(name)
    if not m:
        return name, "", ""
    date, time, _rest = m.groups()
    pretty_date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    pretty_time = f"{time[:2]}:{time[2:]}"
    return name, pretty_date, pretty_time


def parse_run(run_dir: str) -> List[RunRow]:
    """Return one or more rows per run (classical baselines emit multiple)."""
    name = os.path.basename(run_dir)
    run_id, date, time_ = _parse_run_id(name)

    cfg = _load_json(os.path.join(run_dir, "config.json"))
    summary = _load_json(os.path.join(run_dir, "summary.json"))
    n_folds, has_feats = _count_folds_and_feats(run_dir)

    base = RunRow(run_id=run_id, date=date, time=time_, n_folds=n_folds, has_features=has_feats)
    base.paradigm = _infer_paradigm(name, cfg)

    if cfg:
        base.dataset = cfg.get("dataset") or base.dataset
        base.extractor = cfg.get("extractor") or ""
        base.mode = cfg.get("mode") or ""
        base.label = cfg.get("label") or ""
        base.lr = cfg.get("lr")
        base.weight_decay = cfg.get("weight_decay")
        base.epochs = cfg.get("epochs")
        base.batch_size = cfg.get("batch_size")
        base.aug_overlap = cfg.get("aug_overlap")

    # Classical baselines have nested models dict
    if summary and isinstance(summary.get("models"), dict):
        base.paradigm = base.paradigm or "classical"
        base.mode = base.mode or "classical"
        base.n_samples = (summary.get("features") or {}).get("n_samples")
        base.label = (summary.get("config") or {}).get("label") or base.label
        base.status = "complete"
        rows = []
        for model_name, metrics in summary["models"].items():
            r = RunRow(**asdict(base))
            r.model_variant = model_name
            r.subject_acc = metrics.get("acc")
            r.subject_bal_acc = metrics.get("bal_acc")
            r.subject_f1 = metrics.get("f1")
            r.subject_kappa = metrics.get("kappa")
            rows.append(r)
        return rows

    # FT/LP runs: flat metrics (subject_* for subject-level, bare keys for trial-level)
    if summary:
        base.subject_acc = summary.get("subject_acc", summary.get("acc"))
        base.subject_bal_acc = summary.get("subject_bal_acc", summary.get("bal_acc"))
        base.subject_f1 = summary.get("subject_f1", summary.get("f1"))
        base.subject_kappa = summary.get("subject_kappa", summary.get("kappa"))
        base.n_samples = summary.get("n_samples")
        base.status = "complete"
    else:
        # No summary.json — either running or killed
        if cfg and n_folds > 0:
            base.status = "incomplete"
            base.notes = f"{n_folds} fold(s) written, no summary"
        elif cfg:
            base.status = "running_or_failed"
            base.notes = "config only"
        else:
            base.status = "unknown"

    return [base]


def collect_runs(results_dir: str) -> List[RunRow]:
    """Walk results_dir recursively and collect any directory whose name
    matches RUN_DIR_RE (i.e. ``YYYYMMDD_HHMM_*``). This allows runs to be
    organised into category subfolders like ``ft_subject/labram/`` without
    breaking index generation.
    """
    rows: List[RunRow] = []
    skip_dirs = {"archive", "features_cache", ".ipynb_checkpoints"}
    for dirpath, dirnames, _ in os.walk(results_dir):
        # Prune skipped top-level branches in place so os.walk doesn't descend.
        dirnames[:] = sorted(d for d in dirnames if d not in skip_dirs)
        for entry in list(dirnames):
            if not RUN_DIR_RE.match(entry):
                continue
            full = os.path.join(dirpath, entry)
            # Don't descend into run dirs.
            dirnames.remove(entry)
            try:
                rows.extend(parse_run(full))
            except Exception as e:
                rows.append(RunRow(run_id=entry, status="parse_error", notes=str(e)))
    rows.sort(key=lambda r: r.run_id)
    return rows


def write_csv(rows: List[RunRow], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.as_csv().get(k, "") for k in COLUMNS})


def _fmt(v: Any, decimals: int = 4) -> str:
    if v is None or v == "":
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def write_markdown(rows: List[RunRow], path: str) -> None:
    # Group by (paradigm, dataset)
    groups: Dict[tuple, List[RunRow]] = {}
    for r in rows:
        groups.setdefault((r.paradigm or "unknown", r.dataset or "unknown"), []).append(r)

    lines = [
        "# Results Index",
        "",
        f"Auto-generated from `scripts/build_results_index.py`. Source of truth: each run's `config.json` + `summary.json`.",
        "",
        f"**Total runs:** {len({r.run_id for r in rows})}  |  **Rows:** {len(rows)}",
        "",
        "Flat CSV companion: `results/results_index.csv`.",
        "",
    ]

    # Best-by-paradigm leaderboard
    lines += ["## Leaderboard (best subject_bal_acc per paradigm × dataset × extractor)", ""]
    lines += ["| paradigm | dataset | extractor | mode | best bal_acc | run_id | model |",
              "|---|---|---|---|---|---|---|"]
    leaderboard: Dict[tuple, RunRow] = {}
    for r in rows:
        if r.subject_bal_acc is None:
            continue
        key = (r.paradigm, r.dataset, r.extractor or r.mode or "?", r.mode or "?")
        cur = leaderboard.get(key)
        if cur is None or (r.subject_bal_acc or 0) > (cur.subject_bal_acc or 0):
            leaderboard[key] = r
    for key, r in sorted(leaderboard.items()):
        lines.append(
            f"| {r.paradigm} | {r.dataset} | {r.extractor or '—'} | {r.mode or '—'} | "
            f"**{_fmt(r.subject_bal_acc)}** | `{r.run_id}` | {r.model_variant or '—'} |"
        )
    lines.append("")

    # Per-group tables
    for (paradigm, dataset), group in sorted(groups.items()):
        lines += [f"## {paradigm} · {dataset}", ""]
        lines += ["| run_id | extractor | mode | model | folds | bal_acc | acc | f1 | kappa | n | status |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(group, key=lambda x: (x.run_id, x.model_variant)):
            lines.append(
                f"| `{r.run_id}` | {r.extractor or '—'} | {r.mode or '—'} | {r.model_variant or '—'} | "
                f"{r.n_folds or '—'} | {_fmt(r.subject_bal_acc)} | {_fmt(r.subject_acc)} | "
                f"{_fmt(r.subject_f1)} | {_fmt(r.subject_kappa)} | {r.n_samples or '—'} | {r.status} |"
            )
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    rows = collect_runs(RESULTS_DIR)
    csv_path = os.path.join(RESULTS_DIR, "results_index.csv")
    md_path = os.path.join(RESULTS_DIR, "results_index.md")
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    n_runs = len({r.run_id for r in rows})
    n_complete = sum(1 for r in rows if r.status == "complete")
    print(f"Indexed {n_runs} runs, {len(rows)} rows ({n_complete} complete).")
    print(f"  → {csv_path}")
    print(f"  → {md_path}")


if __name__ == "__main__":
    main()
