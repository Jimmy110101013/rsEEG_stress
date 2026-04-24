"""Non-FM deep learning baselines on Stress (per-rec DASS).

Trains EEGNet and ShallowConvNet from scratch to answer: "Is subject
dominance in EEG representations FM-specific, or a property of the signal?"

Two phases:
  Phase 1 — Small LR sweep (3 LRs × 1 seed) to find best LR per model.
  Phase 2 — Multi-seed (3 seeds) at best LR with --save-features for
             variance decomposition.

These are supervised-from-scratch models (no pretrained weights), so:
  - encoder_lr_scale = 1.0  (all params trained equally)
  - warmup_freeze_epochs = 0  (nothing to freeze)
  - norm = zscore  (per CLAUDE.md conventions)

Usage:
    PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python

    # Phase 1: LR sweep (1 seed each, ~30 min on 3 GPUs)
    $PY scripts/run_nonfm_baselines.py --phase sweep --gpus 5 6 7

    # Phase 2: multi-seed at best LR (after reviewing sweep results)
    $PY scripts/run_nonfm_baselines.py --phase multiseed \
        --best-lr-eegnet 1e-3 --best-lr-shallowconvnet 5e-4 \
        --gpus 5 6 7
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

PY = "/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python"
STUDY_ROOT = Path("results/studies/exp15_nonfm_baselines")

MODELS = ["eegnet", "shallowconvnet"]
SWEEP_LRS = ["1e-3", "5e-4", "1e-4"]
SEEDS = [42, 123, 2024]

# Shared training args for from-scratch EEG CNNs
SHARED_ARGS = [
    "--mode", "ft",
    "--folds", "5", "--epochs", "80", "--patience", "20",
    "--batch-size", "32",
    "--label", "dass", "--loss", "focal", "--head-hidden", "128",
    "--norm", "zscore", "--csv", "data/comprehensive_labels.csv",
    "--weight-decay", "0.01", "--grad-clip", "2.0",
    "--warmup-epochs", "5",
    "--encoder-lr-scale", "1.0",
    "--warmup-freeze-epochs", "0",
    "--aug-overlap", "0.75",
    "--llrd", "1.0",
]


def make_cmd(model: str, lr: str, seed: int, run_id: str, gpu: int,
             save_features: bool = False) -> list[str]:
    cmd = [
        PY, "train_ft.py",
        *SHARED_ARGS,
        "--extractor", model,
        "--lr", lr,
        "--seed", str(seed),
        "--device", f"cuda:{gpu}",
        "--run-id", run_id,
    ]
    if save_features:
        cmd.append("--save-features")
    return cmd


def run_job(cmd: list[str], log_path: Path) -> tuple[int, float]:
    import os
    t0 = time.monotonic()
    with open(log_path, "w") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(
            cmd, stdout=lf, stderr=subprocess.STDOUT, check=False,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    return proc.returncode, time.monotonic() - t0


def dispatch_jobs(jobs: list[dict], gpus: list[int]) -> list[dict]:
    """Run jobs across GPU pool with threading."""
    gpu_q: queue.Queue[int] = queue.Queue()
    for g in gpus:
        gpu_q.put(g)

    results = []
    lock = threading.Lock()
    done = {"n": 0}
    total = len(jobs)

    def worker(job: dict):
        gpu = gpu_q.get()
        try:
            cmd = make_cmd(
                job["model"], job["lr"], job["seed"],
                job["run_id"], gpu, job.get("save_features", False),
            )
            with lock:
                print(f"[{done['n']+1:2d}/{total}] START {job['name']} on cuda:{gpu}",
                      flush=True)
            rc, dur = run_job(cmd, job["log_path"])
            with lock:
                done["n"] += 1
                status = "OK" if rc == 0 else "FAIL"
                print(f"[{done['n']:2d}/{total}] {status} {job['name']} "
                      f"(cuda:{gpu}, {dur/60:.1f} min)", flush=True)
            results.append({**job, "returncode": rc, "duration_sec": dur})
        finally:
            gpu_q.put(gpu)

    threads = []
    for job in jobs:
        t = threading.Thread(target=worker, args=(job,))
        t.start()
        threads.append(t)
        time.sleep(0.3)
    for t in threads:
        t.join()

    return results


def phase_sweep(gpus: list[int]):
    """Phase 1: LR sweep with seed=42 only."""
    log_root = STUDY_ROOT / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for model in MODELS:
        for lr in SWEEP_LRS:
            name = f"{model}_lr{lr}_s42"
            run_id = f"studies/exp15_nonfm_baselines/sweep/{name}"
            jobs.append({
                "model": model, "lr": lr, "seed": 42,
                "run_id": run_id, "name": name,
                "log_path": log_root / f"{name}.log",
            })

    print(f"Phase 1: LR sweep — {len(jobs)} jobs on GPUs {gpus}")
    results = dispatch_jobs(jobs, gpus)

    # Summarize
    print(f"\n{'='*60}")
    print("LR Sweep Results")
    print(f"{'='*60}")
    for model in MODELS:
        print(f"\n  {model.upper()}:")
        for lr in SWEEP_LRS:
            name = f"{model}_lr{lr}_s42"
            summary = STUDY_ROOT / "sweep" / name / "summary.json"
            if summary.exists():
                m = json.loads(summary.read_text())
                ba = m.get("subject_bal_acc", "N/A")
                print(f"    lr={lr}: BA={ba}")
            else:
                print(f"    lr={lr}: no summary found")

    print(f"\nDone. {sum(1 for r in results if r['returncode']==0)}/{len(results)} OK.")
    print("\nNext: pick best LR per model and run phase 2:")
    print(f"  {PY} scripts/run_nonfm_baselines.py --phase multiseed \\")
    print(f"      --best-lr-eegnet <LR> --best-lr-shallowconvnet <LR> \\")
    print(f"      --gpus {' '.join(str(g) for g in gpus)}")


def phase_multiseed(gpus: list[int], best_lrs: dict[str, str]):
    """Phase 2: Multi-seed at best LR with feature extraction."""
    log_root = STUDY_ROOT / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for model in MODELS:
        lr = best_lrs[model]
        for seed in SEEDS:
            name = f"{model}_lr{lr}_s{seed}"
            run_id = f"studies/exp15_nonfm_baselines/multiseed/{name}"
            jobs.append({
                "model": model, "lr": lr, "seed": seed,
                "run_id": run_id, "name": name,
                "save_features": True,
                "log_path": log_root / f"ms_{name}.log",
            })

    print(f"Phase 2: Multi-seed — {len(jobs)} jobs on GPUs {gpus}")
    print(f"  EEGNet lr={best_lrs['eegnet']}, ShallowConvNet lr={best_lrs['shallowconvnet']}")
    results = dispatch_jobs(jobs, gpus)

    # Summarize
    print(f"\n{'='*60}")
    print("Multi-Seed Results")
    print(f"{'='*60}")
    for model in MODELS:
        lr = best_lrs[model]
        bas = []
        for seed in SEEDS:
            name = f"{model}_lr{lr}_s{seed}"
            summary = STUDY_ROOT / "multiseed" / name / "summary.json"
            if summary.exists():
                m = json.loads(summary.read_text())
                ba = m.get("subject_bal_acc")
                if ba is not None:
                    bas.append(ba)
                    print(f"  {model} s{seed}: BA={ba:.4f}")

        if bas:
            print(f"  {model.upper()} mean: {np.mean(bas):.4f} ± {np.std(bas):.4f}")
        print()

    ok = sum(1 for r in results if r["returncode"] == 0)
    print(f"Done. {ok}/{len(results)} OK.")

    if ok == len(results):
        print("\nFeatures saved. For variance decomposition, use "
              "scripts/analysis/run_variance_analysis.py.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["sweep", "multiseed"], required=True)
    p.add_argument("--gpus", nargs="+", type=int, required=True)
    p.add_argument("--best-lr-eegnet", type=str, default="1e-3")
    p.add_argument("--best-lr-shallowconvnet", type=str, default="5e-4")
    args = p.parse_args()

    STUDY_ROOT.mkdir(parents=True, exist_ok=True)

    if args.phase == "sweep":
        phase_sweep(args.gpus)
    else:
        best_lrs = {
            "eegnet": args.best_lr_eegnet,
            "shallowconvnet": args.best_lr_shallowconvnet,
        }
        phase_multiseed(args.gpus, best_lrs)


if __name__ == "__main__":
    main()
