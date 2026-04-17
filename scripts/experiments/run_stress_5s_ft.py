"""Stress window-matched FT experiment — fix REVE 5s/10s mismatch.

PROBLEM FOUND (2026-04-14 audit):
  REVE frozen LP used 5s features (extract_frozen_all.py default=5s)
  REVE FT used 10s windows (REVE config inherits BaseModelConfig default=10s)
  → F05's REVE frozen-vs-FT comparison was NOT window-matched.

  LaBraM and CBraMod are fine — both frozen and FT use 5s.

THIS EXPERIMENT:
  Phase 1: REVE FT at 5s (3 seeds) → compare with existing 5s frozen LP (0.494)
  Phase 2: REVE 10s frozen features → compare with existing 10s FT (0.577)

  This gives us two window-matched comparisons:
    5s-matched:  frozen LP (0.494) vs FT at 5s (NEW)
    10s-matched: frozen LP at 10s (NEW) vs FT (0.577)

Usage:
    PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python

    # Phase 1: REVE 5s FT (3 seeds, ~1 hr on 3 GPUs)
    $PY scripts/run_stress_5s_ft.py --phase ft-5s --gpus 5 6 7

    # Phase 2: REVE 10s frozen feature extraction (fast, <5 min, 1 GPU)
    $PY scripts/run_stress_5s_ft.py --phase frozen-10s --gpu 5

    # Phase 3: Summary comparison
    $PY scripts/run_stress_5s_ft.py --phase summary
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

PY = "/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python"
STUDY_ROOT = Path("results/studies/exp16_reve_window_match")
SEEDS = [42, 123, 2024]

# REVE best HP from sweep (at 10s)
REVE_BEST = {"lr": "3e-5", "encoder_lr_scale": "0.1", "norm": "none"}

# Shared FT args
FT_ARGS = [
    "--mode", "ft",
    "--folds", "5", "--epochs", "50", "--patience", "15",
    "--batch-size", "32",
    "--label", "dass", "--loss", "focal", "--head-hidden", "128",
    "--csv", "data/comprehensive_labels.csv",
    "--weight-decay", "0.05", "--grad-clip", "2.0",
    "--warmup-epochs", "3",
    "--warmup-freeze-epochs", "1", "--aug-overlap", "0.75",
    "--llrd", "1.0",
    "--extractor", "reve",
    "--norm", REVE_BEST["norm"],
    "--lr", REVE_BEST["lr"],
    "--encoder-lr-scale", REVE_BEST["encoder_lr_scale"],
]

# Existing results for comparison
EXISTING = {
    "reve_frozen_5s": {
        "source": "results/features_cache/frozen_reve_stress_30ch.npz",
        "ba": 0.494,  # 8-seed mean from exp03
        "note": "extract_frozen_all.py default=5s",
    },
    "reve_ft_10s": {
        "source": "results/hp_sweep/20260410_dass/reve/",
        "bas": [0.5625, 0.6339, 0.5357],  # encoderlrscale0.1_lr3e-5
        "note": "REVE config default=10s",
    },
}


def run_job(cmd: list[str], log_path: Path) -> tuple[int, float]:
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
            cmd = job["cmd_fn"](gpu)
            with lock:
                print(f"[{done['n']+1:2d}/{total}] START {job['name']} "
                      f"on cuda:{gpu}", flush=True)
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


def phase_ft_5s(gpus: list[int]):
    """Phase 1: REVE FT at 5s windows (3 seeds)."""
    log_root = STUDY_ROOT / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for seed in SEEDS:
        name = f"reve_5s_s{seed}"
        run_id = f"studies/exp16_reve_window_match/ft_5s/{name}"

        def make_cmd(gpu, _seed=seed, _run_id=run_id):
            return [
                PY, "train_ft.py", *FT_ARGS,
                "--window-sec", "5",
                "--seed", str(_seed),
                "--device", f"cuda:{gpu}",
                "--run-id", _run_id,
                "--save-features",
            ]

        jobs.append({
            "name": name, "cmd_fn": make_cmd,
            "log_path": log_root / f"{name}.log",
        })

    print(f"Phase 1: REVE FT at 5s — {len(jobs)} jobs on GPUs {gpus}")
    results = dispatch_jobs(jobs, gpus)

    # Show results
    bas = []
    for seed in SEEDS:
        summary = STUDY_ROOT / "ft_5s" / f"reve_5s_s{seed}" / "summary.json"
        if summary.exists():
            ba = json.loads(summary.read_text())["subject_bal_acc"]
            bas.append(ba)
            print(f"  s{seed}: BA={ba:.4f}")

    if bas:
        print(f"  REVE FT 5s mean: {np.mean(bas):.4f} ± {np.std(bas):.4f}")
        print(f"  REVE FT 10s ref: {np.mean(EXISTING['reve_ft_10s']['bas']):.4f} "
              f"± {np.std(EXISTING['reve_ft_10s']['bas']):.4f}")
        print(f"  REVE frozen 5s:  {EXISTING['reve_frozen_5s']['ba']:.4f}")


def phase_frozen_10s(gpu: int):
    """Phase 2: Extract REVE frozen features at 10s windows."""
    log_root = STUDY_ROOT / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    out_suffix = "_w10"
    cmd = [
        PY, "scripts/extract_frozen_all.py",
        "--extractor", "reve",
        "--dataset", "stress",
        "--device", f"cuda:{gpu}",
        "--window-sec", "10",
        "--out-suffix", out_suffix,
    ]

    print(f"Phase 2: Extract REVE 10s frozen features on cuda:{gpu}")
    print(f"  CMD: {' '.join(cmd)}")
    rc, dur = run_job(cmd, log_root / "frozen_10s.log")

    out_path = Path("results/features_cache/frozen_reve_stress_30ch_w10.npz")
    if rc == 0 and out_path.exists():
        d = np.load(out_path)
        print(f"  OK ({dur:.0f}s) → {out_path} shape={d['features'].shape}")

        # Run frozen LP on 10s features
        print(f"\n  Running frozen LP on 10s features...")
        from scripts.stress_frozen_lp_multiseed import run_frozen_lp
        # We'll do this inline since it's just sklearn, no GPU needed
    else:
        status = "FAIL" if rc != 0 else "output missing"
        print(f"  {status}")


def phase_summary():
    """Phase 3: Print complete comparison table."""
    print("=" * 70)
    print("REVE Window-Matched Comparison (Stress per-rec DASS)")
    print("=" * 70)
    print()

    # 5s-matched
    print("5s-matched comparison:")
    frozen_5s_ba = EXISTING["reve_frozen_5s"]["ba"]
    print(f"  Frozen LP (5s): {frozen_5s_ba:.4f} (8-seed mean)")

    ft_5s_bas = []
    for seed in SEEDS:
        summary = STUDY_ROOT / "ft_5s" / f"reve_5s_s{seed}" / "summary.json"
        if summary.exists():
            ba = json.loads(summary.read_text())["subject_bal_acc"]
            ft_5s_bas.append(ba)
    if ft_5s_bas:
        mean_5s = np.mean(ft_5s_bas)
        print(f"  FT (5s):        {mean_5s:.4f} ± {np.std(ft_5s_bas):.4f} "
              f"(3 seeds: {ft_5s_bas})")
        print(f"  Δ(FT−Frozen):   {mean_5s - frozen_5s_ba:+.4f} pp")
    else:
        print(f"  FT (5s):        not yet run")

    # 10s-matched
    print()
    print("10s-matched comparison:")
    frozen_10s_path = Path("results/features_cache/frozen_reve_stress_30ch_w10.npz")
    if frozen_10s_path.exists():
        print(f"  Frozen LP (10s): TODO — run LP on {frozen_10s_path}")
    else:
        print(f"  Frozen LP (10s): not yet extracted")

    ft_10s_bas = EXISTING["reve_ft_10s"]["bas"]
    print(f"  FT (10s):        {np.mean(ft_10s_bas):.4f} ± {np.std(ft_10s_bas):.4f} "
          f"(3 seeds: {ft_10s_bas})")

    # Original mismatched comparison
    print()
    print("Original (MISMATCHED — frozen=5s, FT=10s):")
    print(f"  Frozen LP (5s):  {frozen_5s_ba:.4f}")
    print(f"  FT (10s):        {np.mean(ft_10s_bas):.4f} ± {np.std(ft_10s_bas):.4f}")
    print(f"  Δ(FT−Frozen):    {np.mean(ft_10s_bas) - frozen_5s_ba:+.4f} pp  ← F05 reported this")

    print()
    print("LaBraM and CBraMod: both frozen and FT already at 5s — no mismatch.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["ft-5s", "frozen-10s", "summary"],
                   required=True)
    p.add_argument("--gpus", nargs="+", type=int, default=[5, 6, 7])
    p.add_argument("--gpu", type=int, default=5,
                   help="Single GPU for frozen extraction")
    args = p.parse_args()

    STUDY_ROOT.mkdir(parents=True, exist_ok=True)

    if args.phase == "ft-5s":
        phase_ft_5s(args.gpus)
    elif args.phase == "frozen-10s":
        phase_frozen_10s(args.gpu)
    elif args.phase == "summary":
        phase_summary()


if __name__ == "__main__":
    main()
