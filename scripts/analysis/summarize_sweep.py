"""Aggregate a finished HP sweep into per-model leaderboards.

Reads ``results/hp_sweep/<sweep_id>/sweep_manifest.json`` + every run's
``summary.json`` + ``window_metrics.csv`` and produces:

  * ``leaderboard.csv`` / ``leaderboard.md`` — per-(model, config)
    mean ± std of subject-level balanced accuracy across seeds, plus
    per-fold statistics, ranked inside each model.
  * ``best_per_model.csv`` / ``best_per_model.md`` — the top config
    per model (by mean subject BA over seeds), ready to drop into the
    paper's baseline comparison table.

Run from project root:
    /raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python \
        scripts/summarize_sweep.py --sweep-id 20260409_main
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _load_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "summary.json"
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"    WARN: failed to read {path}: {e}", file=sys.stderr)
        return None


def _per_fold_ba(run_dir: Path) -> dict[int, float]:
    """Recompute subject-level BA for each fold from `window_metrics.csv`.

    Each row in window_metrics.csv is one window; `subject_pred` is the
    same value within a (fold, patient_id) pair (the majority vote made
    downstream). Dedupe per (fold, patient) and score against `y_true`.
    """
    path = run_dir / "window_metrics.csv"
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    if df.empty or not {"fold", "patient_id", "y_true", "subject_pred"}.issubset(df.columns):
        return {}
    subjects = df.drop_duplicates(["fold", "patient_id"])[
        ["fold", "patient_id", "y_true", "subject_pred"]
    ]
    out: dict[int, float] = {}
    for fold, grp in subjects.groupby("fold"):
        y_true = grp["y_true"].astype(int).to_numpy()
        y_pred = grp["subject_pred"].astype(int).to_numpy()
        if len(np.unique(y_true)) < 2:
            # Single-class fold — BA degenerates; fall back to accuracy.
            out[int(fold)] = float((y_true == y_pred).mean())
        else:
            out[int(fold)] = float(balanced_accuracy_score(y_true, y_pred))
    return out


def load_runs(sweep_dir: Path) -> pd.DataFrame:
    """Return one row per completed run with its summary + per-fold BA."""
    manifest_path = sweep_dir / "sweep_manifest.json"
    if not manifest_path.is_file():
        sys.exit(f"missing manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    workdir = Path(__file__).resolve().parent.parent

    rows = []
    for entry in manifest["runs"]:
        if entry["returncode"] != 0:
            continue
        run_dir = workdir / entry["results_dir"]
        summary = _load_summary(run_dir)
        if summary is None:
            continue
        fold_ba = _per_fold_ba(run_dir)
        row = {
            "model": entry["model"],
            "seed": entry["seed"],
            "run_name": entry["run_name"],
            "hp_signature": _signature_from_hp(entry["hp"]),
            "subject_bal_acc": summary.get("subject_bal_acc"),
            "subject_acc": summary.get("subject_acc"),
            "subject_f1": summary.get("subject_f1"),
            "subject_kappa": summary.get("subject_kappa"),
            "n_samples": summary.get("n_samples"),
            "fold_ba_mean": float(np.mean(list(fold_ba.values()))) if fold_ba else float("nan"),
            "fold_ba_std": float(np.std(list(fold_ba.values()))) if fold_ba else float("nan"),
            "fold_ba_min": float(np.min(list(fold_ba.values()))) if fold_ba else float("nan"),
            "fold_ba_max": float(np.max(list(fold_ba.values()))) if fold_ba else float("nan"),
        }
        row.update({f"hp_{k}": v for k, v in entry["hp"].items()})
        rows.append(row)

    return pd.DataFrame(rows)


def _signature_from_hp(hp: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(hp.items()))


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, hp_signature): mean ± std across seeds."""
    if df.empty:
        return df
    grouped = df.groupby(["model", "hp_signature"]).agg(
        n_seeds=("seed", "count"),
        bal_acc_mean=("subject_bal_acc", "mean"),
        bal_acc_std=("subject_bal_acc", "std"),
        bal_acc_min=("subject_bal_acc", "min"),
        bal_acc_max=("subject_bal_acc", "max"),
        fold_ba_mean=("fold_ba_mean", "mean"),
        fold_ba_std_seeds=("fold_ba_std", "mean"),  # avg per-fold std within a seed
        fold_ba_worst=("fold_ba_min", "mean"),
    ).reset_index()
    grouped["bal_acc_std"] = grouped["bal_acc_std"].fillna(0.0)
    # Sort: best config per model on top.
    grouped = grouped.sort_values(
        ["model", "bal_acc_mean"], ascending=[True, False]
    ).reset_index(drop=True)
    return grouped


def best_per_model(lb: pd.DataFrame) -> pd.DataFrame:
    """Top config per model, paper-ready."""
    if lb.empty:
        return lb
    return lb.loc[lb.groupby("model")["bal_acc_mean"].idxmax()].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Markdown renderers
# --------------------------------------------------------------------------- #

def _fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x*100:.2f}"


def render_leaderboard_md(lb: pd.DataFrame) -> str:
    lines = [
        "# HP Sweep Leaderboard",
        "",
        "All values are subject-level balanced accuracy (%) averaged across 3 seeds.",
        "`fold_ba_worst` = mean (over seeds) of the single-fold minimum — the ",
        "classic small-N fold-collapse indicator.",
        "",
    ]
    for model, grp in lb.groupby("model"):
        lines.append(f"## {model}")
        lines.append("")
        header = (
            "| rank | config | n_seeds | bal_acc mean | bal_acc std | min | max | "
            "fold mean | fold worst |"
        )
        sep = "|---|---|---|---|---|---|---|---|---|"
        lines.append(header)
        lines.append(sep)
        for i, r in enumerate(grp.itertuples(), start=1):
            lines.append(
                f"| {i} | `{r.hp_signature}` | {r.n_seeds} | "
                f"**{_fmt_pct(r.bal_acc_mean)}** | {_fmt_pct(r.bal_acc_std)} | "
                f"{_fmt_pct(r.bal_acc_min)} | {_fmt_pct(r.bal_acc_max)} | "
                f"{_fmt_pct(r.fold_ba_mean)} | {_fmt_pct(r.fold_ba_worst)} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_best_md(best: pd.DataFrame) -> str:
    lines = [
        "# Best HP per Model",
        "",
        "Top config per model selected by mean subject BA across 3 seeds.",
        "",
        "| model | best config | bal_acc mean | bal_acc std | worst fold BA |",
        "|---|---|---|---|---|",
    ]
    for r in best.itertuples():
        lines.append(
            f"| **{r.model}** | `{r.hp_signature}` | "
            f"**{_fmt_pct(r.bal_acc_mean)}** | {_fmt_pct(r.bal_acc_std)} | "
            f"{_fmt_pct(r.fold_ba_worst)} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-id", required=True,
                   help="Sweep id used when the runs were launched.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    workdir = Path(__file__).resolve().parent.parent
    sweep_dir = workdir / "results" / "hp_sweep" / args.sweep_id
    if not sweep_dir.is_dir():
        sys.exit(f"sweep dir not found: {sweep_dir}")

    runs = load_runs(sweep_dir)
    print(f"Loaded {len(runs)} successful runs from {sweep_dir}")
    if runs.empty:
        sys.exit("no successful runs to summarise")

    runs.to_csv(sweep_dir / "runs_flat.csv", index=False)

    lb = leaderboard(runs)
    lb.to_csv(sweep_dir / "leaderboard.csv", index=False)
    (sweep_dir / "leaderboard.md").write_text(render_leaderboard_md(lb))

    best = best_per_model(lb)
    best.to_csv(sweep_dir / "best_per_model.csv", index=False)
    (sweep_dir / "best_per_model.md").write_text(render_best_md(best))

    print(f"→ runs_flat.csv      ({len(runs)} rows)")
    print(f"→ leaderboard.csv    ({len(lb)} configs)")
    print(f"→ leaderboard.md")
    print(f"→ best_per_model.csv ({len(best)} models)")
    print(f"→ best_per_model.md")
    print()
    print("Best per model:")
    for r in best.itertuples():
        print(f"  {r.model:15s}  {_fmt_pct(r.bal_acc_mean)} ± "
              f"{_fmt_pct(r.bal_acc_std)}   [{r.hp_signature}]")


if __name__ == "__main__":
    main()
