"""Hyperparameter sweep + multi-seed launcher for subject-level stress CV.

Drives `train_ft.py` as a subprocess for every (model × HP config × seed) cell
in a per-model HP grid. Runs are distributed across a GPU pool so every GPU
holds at most one live run at a time. Each run writes into
``results/hp_sweep/<sweep_id>/<model>/<run_name>/``; the sweep-wide manifest
lands at ``results/hp_sweep/<sweep_id>/sweep_manifest.json``.

Pair this script with ``scripts/summarize_sweep.py`` to turn the raw runs
into per-model mean ± std leaderboards.

Usage:
    /raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python \
        scripts/run_hp_sweep.py --sweep-id 20260409_main \
        --gpus 1 2 3 4 5 6 7 \
        --models eegnet shallowconvnet deepconvnet eegconformer \
                 labram cbramod reve

    # Dry run (print plan only, don't launch anything)
    python scripts/run_hp_sweep.py --dry-run ...
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# HP grids
# --------------------------------------------------------------------------- #
# Each grid is a dict of flag-name -> list of values. Only the flags that
# actually vary per config should live here; shared defaults come from the
# COMMON args block below.
#
# Budget per model: 6 configs × 3 seeds = 18 runs.

DL_GRID: dict[str, list] = {
    "lr": [1e-4, 3e-4, 1e-3],
    "dropout": [0.1, 0.5],        # head dropout
}

FM_GRID: dict[str, list] = {
    "lr": [1e-5, 3e-5, 1e-4],
    "encoder_lr_scale": [0.1, 1.0],
}

SEEDS = [42, 123, 2024]

# Flags shared across every run in the sweep (fixed by protocol).
COMMON_ARGS = {
    "mode": "ft",
    "folds": 5,
    "epochs": 50,
    "patience": 15,
    "batch-size": 32,
    "label": "dass",
    "loss": "focal",
    "head-hidden": 128,
    "csv": "data/comprehensive_labels.csv",
    "weight-decay": 0.05,
    "grad-clip": 2.0,
    "warmup-epochs": 3,
    "aug-overlap": 0.75,
}

# Per-model fixed args. `norm` is model-specific because each FM expects a
# different input scale:
#   * LaBraM     — zscore per-channel per-window (matches paper fine-tune
#                  recipe and our canonical 0.656 run).
#   * CBraMod    — raw µV; the extractor internally divides by 100 to reach
#                  ~[-1, 1]. Passing zscored input would double-scale.
#   * REVE       — raw µV; patch embedding is a linear projection trained
#                  on µV-scale pretraining data, so it is scale-sensitive.
#   * DL models  — zscore (Lawhern/Schirrmeister/Song conventions, plus
#                  early BatchNorm absorbs any residual scale).
MODELS: dict[str, dict[str, Any]] = {
    # DL from-scratch baselines
    "eegnet":         {"type": "dl", "fixed": {"norm": "zscore"}},
    "shallowconvnet": {"type": "dl", "fixed": {"norm": "zscore"}},
    "deepconvnet":    {"type": "dl", "fixed": {"norm": "zscore"}},
    "eegconformer":   {"type": "dl", "fixed": {"norm": "zscore"}},
    # Foundation models
    "labram":         {"type": "fm", "fixed": {"norm": "none"}},  # 2026-04-26: was "zscore"
    "cbramod":        {"type": "fm", "fixed": {"norm": "none"}},
    "reve":           {"type": "fm", "fixed": {"norm": "none"}},
}


# --------------------------------------------------------------------------- #
# Sweep plan
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RunSpec:
    """A single (model, HP config, seed) cell in the sweep."""
    model: str
    seed: int
    hp: tuple[tuple[str, Any], ...]  # frozen key-value pairs
    run_name: str
    run_id: str                       # path-like run_id passed to train_ft.py
    results_dir: str                  # absolute path for post-run inspection


def _config_signature(hp: dict[str, Any]) -> str:
    """Short human-readable signature for an HP config.

    Produces strings like ``lr3e-04_do0.5`` that slot nicely into run names.
    """
    parts = []
    for k, v in sorted(hp.items()):
        short = k.replace("_", "").replace("lr", "lr", 1)
        if isinstance(v, float):
            if abs(v) < 1e-2 or abs(v) >= 1e3:
                parts.append(f"{short}{v:.0e}".replace("e-0", "e-"))
            else:
                parts.append(f"{short}{v:g}")
        else:
            parts.append(f"{short}{v}")
    return "_".join(parts)


def build_plan(sweep_id: str, models: list[str]) -> list[RunSpec]:
    """Expand the HP grid × seeds × models into concrete RunSpec entries."""
    plan: list[RunSpec] = []
    for model in models:
        if model not in MODELS:
            raise ValueError(f"Unknown model '{model}'. Known: {list(MODELS)}")
        grid = DL_GRID if MODELS[model]["type"] == "dl" else FM_GRID
        keys = sorted(grid.keys())
        for combo in itertools.product(*[grid[k] for k in keys]):
            hp_dict = dict(zip(keys, combo))
            sig = _config_signature(hp_dict)
            for seed in SEEDS:
                run_name = f"{model}_{sig}_s{seed}"
                # run_id has slashes so train_ft.py creates nested dirs under
                # results/hp_sweep/<sweep_id>/<model>/...
                run_id = f"hp_sweep/{sweep_id}/{model}/{run_name}"
                results_dir = os.path.join("results", run_id)
                plan.append(
                    RunSpec(
                        model=model,
                        seed=seed,
                        hp=tuple(sorted(hp_dict.items())),
                        run_name=run_name,
                        run_id=run_id,
                        results_dir=results_dir,
                    )
                )
    return plan


def _build_command(run: RunSpec, device: str, python_bin: str) -> list[str]:
    """Assemble the `train_ft.py` CLI for a single run."""
    cmd = [
        python_bin, "train_ft.py",
        "--extractor", run.model,
        "--seed", str(run.seed),
        "--device", device,
        "--run-id", run.run_id,
    ]
    for key, val in COMMON_ARGS.items():
        cmd.extend([f"--{key}", str(val)])
    # Per-model fixed args (e.g. the input normalisation mode).
    for key, val in MODELS[run.model].get("fixed", {}).items():
        cli_key = key.replace("_", "-")
        cmd.extend([f"--{cli_key}", str(val)])
    for key, val in run.hp:
        cli_key = key.replace("_", "-")
        cmd.extend([f"--{cli_key}", str(val)])
    return cmd


# --------------------------------------------------------------------------- #
# GPU-aware worker pool
# --------------------------------------------------------------------------- #

@dataclass
class RunResult:
    run: RunSpec
    device: str
    returncode: int
    duration_sec: float
    stdout_log: str
    stderr_tail: str = ""


def _run_worker(
    run: RunSpec,
    device: str,
    python_bin: str,
    log_dir: Path,
    workdir: Path,
) -> RunResult:
    cmd = _build_command(run, device, python_bin)
    log_path = log_dir / f"{run.run_name}.log"
    start = time.monotonic()
    with open(log_path, "w") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=str(workdir),
            check=False,
        )
    duration = time.monotonic() - start

    stderr_tail = ""
    if proc.returncode != 0:
        try:
            with open(log_path) as f:
                lines = f.readlines()
            stderr_tail = "".join(lines[-25:])
        except Exception:
            pass

    return RunResult(
        run=run,
        device=device,
        returncode=proc.returncode,
        duration_sec=duration,
        stdout_log=str(log_path),
        stderr_tail=stderr_tail,
    )


def launch_pool(
    plan: list[RunSpec],
    gpus: list[int],
    python_bin: str,
    log_dir: Path,
    workdir: Path,
) -> list[RunResult]:
    """Run `plan` across a pool of GPU workers (one run per GPU at a time)."""
    gpu_q: "queue.Queue[int]" = queue.Queue()
    for g in gpus:
        gpu_q.put(g)

    lock = threading.Lock()
    results: list[RunResult] = []
    progress = {"done": 0, "total": len(plan)}

    def worker(run: RunSpec):
        gpu = gpu_q.get()
        device = f"cuda:{gpu}"
        try:
            with lock:
                print(f"[{progress['done']+1:3d}/{progress['total']}] "
                      f"START on {device}: {run.run_name}", flush=True)
            r = _run_worker(run, device, python_bin, log_dir, workdir)
            with lock:
                progress["done"] += 1
                status = "OK " if r.returncode == 0 else "FAIL"
                print(f"[{progress['done']:3d}/{progress['total']}] "
                      f"{status} on {device} ({r.duration_sec/60:5.1f} min): "
                      f"{run.run_name}", flush=True)
                if r.returncode != 0:
                    print("    " + r.stderr_tail.replace("\n", "\n    "),
                          flush=True)
            results.append(r)
        finally:
            gpu_q.put(gpu)

    threads: list[threading.Thread] = []
    for run in plan:
        t = threading.Thread(target=worker, args=(run,))
        t.start()
        threads.append(t)
        # Keep at most len(gpus) runs outstanding at a time by blocking on
        # the GPU queue — worker() only starts once it grabs a GPU slot, so
        # thread spawn rate is naturally throttled.
        time.sleep(0.1)

    for t in threads:
        t.join()

    return results


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

def write_manifest(sweep_dir: Path, sweep_id: str, plan: list[RunSpec],
                   results: list[RunResult], gpus: list[int]) -> None:
    manifest = {
        "sweep_id": sweep_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "gpus": gpus,
        "seeds": SEEDS,
        "dl_grid": DL_GRID,
        "fm_grid": FM_GRID,
        "common_args": COMMON_ARGS,
        "per_model_fixed": {m: MODELS[m].get("fixed", {}) for m in MODELS},
        "n_runs_planned": len(plan),
        "n_runs_completed_ok": sum(1 for r in results if r.returncode == 0),
        "n_runs_failed": sum(1 for r in results if r.returncode != 0),
        "runs": [
            {
                "model": r.run.model,
                "seed": r.run.seed,
                "hp": dict(r.run.hp),
                "run_name": r.run.run_name,
                "run_id": r.run.run_id,
                "results_dir": r.run.results_dir,
                "device": r.device,
                "duration_sec": r.duration_sec,
                "returncode": r.returncode,
                "log": r.stdout_log,
            }
            for r in sorted(results, key=lambda x: x.run.run_name)
        ],
    }
    out_path = sweep_dir / "sweep_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n→ Manifest: {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-id", default=None,
                   help="Unique id for this sweep (default: current timestamp).")
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                   help="Models to sweep.")
    p.add_argument("--gpus", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7],
                   help="GPU indices to use (one run per GPU at a time).")
    p.add_argument("--python-bin",
                   default="/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python",
                   help="Python interpreter to invoke train_ft.py with.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan without launching any runs.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only run the first N cells of the plan (debug helper).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.sweep_id:
        args.sweep_id = datetime.now().strftime("%Y%m%d_%H%M")

    workdir = Path(__file__).resolve().parent.parent
    sweep_dir = workdir / "results" / "hp_sweep" / args.sweep_id
    log_dir = sweep_dir / "_logs"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    plan = build_plan(args.sweep_id, args.models)
    if args.limit:
        plan = plan[: args.limit]

    print(f"Sweep id          : {args.sweep_id}")
    print(f"Models            : {args.models}")
    print(f"GPUs              : {args.gpus}")
    print(f"Seeds             : {SEEDS}")
    print(f"Total runs        : {len(plan)}")
    print(f"Sweep dir         : {sweep_dir}")
    print(f"Per-run log dir   : {log_dir}")
    print()
    print("First 5 planned runs:")
    for r in plan[:5]:
        print(f"  {r.run_name}")
    if len(plan) > 5:
        print(f"  ... (+{len(plan) - 5} more)")
    print()

    if args.dry_run:
        print("(dry run — nothing was launched)")
        return

    t0 = time.monotonic()
    results = launch_pool(
        plan=plan,
        gpus=args.gpus,
        python_bin=args.python_bin,
        log_dir=log_dir,
        workdir=workdir,
    )
    total_min = (time.monotonic() - t0) / 60
    ok = sum(1 for r in results if r.returncode == 0)
    fail = len(results) - ok
    print(f"\nSweep finished in {total_min:.1f} min  |  ok={ok}  fail={fail}")

    write_manifest(sweep_dir, args.sweep_id, plan, results, args.gpus)


if __name__ == "__main__":
    main()
