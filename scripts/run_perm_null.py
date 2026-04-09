"""Permutation-null launcher for Stress per-rec dass FT.

Runs N permutation-label FT runs across a GPU pool to build a null
distribution of subject BA under random labels. Each permutation uses a
fixed canonical HP recipe (LaBraM lr=1e-5, encoder_lr_scale=0.1, llrd=1.0,
epochs=50, patience=15, seed=42 for splits), only the label permutation
RNG seed varies.

Usage:
    /raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python \
        scripts/run_perm_null.py --n-perms 10 --gpus 6 7
"""
from __future__ import annotations

import argparse
import queue
import subprocess
import threading
import time
from pathlib import Path


PY = "/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python"

BASE_ARGS = [
    "--mode", "ft", "--extractor", "labram",
    "--folds", "5", "--epochs", "50", "--patience", "15",
    "--lr", "1e-5", "--batch-size", "32", "--seed", "42",
    "--label", "dass", "--loss", "focal", "--head-hidden", "128",
    "--norm", "zscore", "--csv", "data/comprehensive_labels.csv",
    "--weight-decay", "0.05", "--grad-clip", "2.0",
    "--warmup-epochs", "3", "--encoder-lr-scale", "0.1",
    "--warmup-freeze-epochs", "1", "--aug-overlap", "0.75",
    "--llrd", "1.0",
]


def run_perm(perm_seed: int, gpu: int, out_root: Path) -> tuple[int, int, float]:
    run_id = f"studies/2026-04-10_stress_erosion/ft_null/perm_s{perm_seed}"
    log_path = out_root / f"perm_s{perm_seed}.log"
    cmd = [
        PY, "train_ft.py",
        *BASE_ARGS,
        "--device", f"cuda:{gpu}",
        "--run-id", run_id,
        "--permute-labels", str(perm_seed),
    ]
    t0 = time.monotonic()
    with open(log_path, "w") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False)
    dur = time.monotonic() - t0
    return perm_seed, proc.returncode, dur


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-perms", type=int, default=10)
    p.add_argument("--gpus", nargs="+", type=int, required=True)
    args = p.parse_args()

    out_root = Path("results/studies/2026-04-10_stress_erosion/logs")
    out_root.mkdir(parents=True, exist_ok=True)

    gpu_q: "queue.Queue[int]" = queue.Queue()
    for g in args.gpus:
        gpu_q.put(g)

    results = []
    lock = threading.Lock()
    done = {"n": 0}

    def worker(perm_seed: int):
        gpu = gpu_q.get()
        try:
            with lock:
                print(f"[{done['n']+1:2d}/{args.n_perms}] START perm_s{perm_seed} on cuda:{gpu}", flush=True)
            r = run_perm(perm_seed, gpu, out_root)
            with lock:
                done["n"] += 1
                status = "OK" if r[1] == 0 else "FAIL"
                print(f"[{done['n']:2d}/{args.n_perms}] {status} perm_s{perm_seed} "
                      f"(cuda:{gpu}, {r[2]/60:.1f} min)", flush=True)
            results.append(r)
        finally:
            gpu_q.put(gpu)

    threads = []
    for s in range(args.n_perms):
        t = threading.Thread(target=worker, args=(s,))
        t.start()
        threads.append(t)
        time.sleep(0.1)
    for t in threads:
        t.join()

    print(f"\nDone. {sum(1 for r in results if r[1]==0)}/{args.n_perms} OK.")


if __name__ == "__main__":
    main()
