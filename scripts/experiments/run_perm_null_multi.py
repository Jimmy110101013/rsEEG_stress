"""Permutation-null launcher for CBraMod and REVE on Stress per-rec dass FT.

Builds null distributions of subject BA under random labels to test whether
the observed FT injection (F05: CBraMod +8.3pp, REVE +8.0pp) is real signal
or indistinguishable from noise.

Each model uses its best HP config from the 20260410_dass sweep:
  - CBraMod: lr=1e-5, encoder_lr_scale=0.1, norm=none
  - REVE:    lr=3e-5, encoder_lr_scale=0.1, norm=none

Usage:
    /raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python \
        scripts/run_perm_null_multi.py --models cbramod reve \
        --n-perms 10 --gpus 5 6 7
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import threading
import time
from pathlib import Path

PY = "/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python"

# Shared args across all models (matching HP sweep config)
SHARED_ARGS = [
    "--mode", "ft",
    "--folds", "5", "--epochs", "50", "--patience", "15",
    "--batch-size", "32", "--seed", "42",
    "--label", "dass", "--loss", "focal", "--head-hidden", "128",
    "--csv", "data/comprehensive_labels.csv",
    "--weight-decay", "0.05", "--grad-clip", "2.0",
    "--warmup-epochs", "3",
    "--warmup-freeze-epochs", "1", "--aug-overlap", "0.75",
    "--llrd", "1.0",
]

# Per-model best HP configs from sweep
MODEL_CONFIGS = {
    "cbramod": {
        "args": [
            "--extractor", "cbramod",
            "--norm", "none",
            "--lr", "1e-5",
            "--encoder-lr-scale", "0.1",
        ],
        "out_dir": "ft_null_cbramod",
    },
    "reve": {
        "args": [
            "--extractor", "reve",
            "--norm", "none",
            "--lr", "3e-5",
            "--encoder-lr-scale", "0.1",
        ],
        "out_dir": "ft_null_reve",
    },
}

STUDY_ROOT = Path("results/studies/exp03_stress_erosion")


def run_perm(model: str, perm_seed: int, gpu: int, log_root: Path) -> tuple[str, int, int, float]:
    cfg = MODEL_CONFIGS[model]
    run_id = f"studies/exp03_stress_erosion/{cfg['out_dir']}/perm_s{perm_seed}"
    log_path = log_root / f"{model}_perm_s{perm_seed}.log"
    cmd = [
        PY, "train_ft.py",
        *SHARED_ARGS,
        *cfg["args"],
        "--device", f"cuda:{gpu}",
        "--run-id", run_id,
        "--permute-labels", str(perm_seed),
    ]
    t0 = time.monotonic()
    with open(log_path, "w") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(
            cmd, stdout=lf, stderr=subprocess.STDOUT, check=False,
            env={"PYTHONUNBUFFERED": "1", "PATH": "/usr/bin:/bin",
                 "HOME": str(Path.home()),
                 "CUDA_VISIBLE_DEVICES": "",  # let --device handle it
                 **{k: v for k, v in __import__('os').environ.items()
                    if k not in ("CUDA_VISIBLE_DEVICES",)}},
        )
    dur = time.monotonic() - t0
    return model, perm_seed, proc.returncode, dur


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", choices=list(MODEL_CONFIGS),
                   default=list(MODEL_CONFIGS), help="Models to run null for")
    p.add_argument("--n-perms", type=int, default=10)
    p.add_argument("--gpus", nargs="+", type=int, required=True)
    args = p.parse_args()

    log_root = STUDY_ROOT / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    # Build job list: (model, perm_seed) pairs
    jobs = []
    for model in args.models:
        out = STUDY_ROOT / MODEL_CONFIGS[model]["out_dir"]
        out.mkdir(parents=True, exist_ok=True)
        for s in range(args.n_perms):
            jobs.append((model, s))

    total = len(jobs)
    print(f"Launching {total} permutation-null runs "
          f"({', '.join(args.models)}) on GPUs {args.gpus}")

    gpu_q: "queue.Queue[int]" = queue.Queue()
    for g in args.gpus:
        gpu_q.put(g)

    results: list[tuple[str, int, int, float]] = []
    lock = threading.Lock()
    done = {"n": 0}

    def worker(model: str, perm_seed: int):
        gpu = gpu_q.get()
        try:
            with lock:
                print(f"[{done['n']+1:2d}/{total}] START {model} perm_s{perm_seed} "
                      f"on cuda:{gpu}", flush=True)
            r = run_perm(model, perm_seed, gpu, log_root)
            with lock:
                done["n"] += 1
                status = "OK" if r[2] == 0 else "FAIL"
                print(f"[{done['n']:2d}/{total}] {status} {model} perm_s{perm_seed} "
                      f"(cuda:{gpu}, {r[3]/60:.1f} min)", flush=True)
            results.append(r)
        finally:
            gpu_q.put(gpu)

    threads = []
    for model, perm_seed in jobs:
        t = threading.Thread(target=worker, args=(model, perm_seed))
        t.start()
        threads.append(t)
        time.sleep(0.2)  # stagger launches slightly
    for t in threads:
        t.join()

    # Summary
    print(f"\nDone. {sum(1 for r in results if r[2]==0)}/{total} OK.\n")

    # Collect BAs per model
    for model in args.models:
        cfg = MODEL_CONFIGS[model]
        bas = {}
        for s in range(args.n_perms):
            summary = STUDY_ROOT / cfg["out_dir"] / f"perm_s{s}" / "summary.json"
            if summary.exists():
                m = json.loads(summary.read_text())
                ba = m.get("subject_bal_acc", m.get("bal_acc"))
                if ba is not None:
                    bas[f"perm_s{s}"] = ba

        if bas:
            import numpy as np
            vals = list(bas.values())
            print(f"{model.upper()} null distribution ({len(vals)} perms):")
            print(f"  mean={np.mean(vals):.4f} ± {np.std(vals):.4f}")
            print(f"  range=[{np.min(vals):.4f}, {np.max(vals):.4f}]")
            print(f"  per-perm: {bas}")
            print()


if __name__ == "__main__":
    main()
