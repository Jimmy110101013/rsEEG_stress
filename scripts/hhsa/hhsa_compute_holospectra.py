"""Compute holospectra from Layer-1 CEEMDAN cache.

Reads cached (IF, IA) per recording → windows into 5s slices →
Layer-2 AM decomposition → holospectrum binning → saves per-recording.

Usage:
    python scripts/hhsa_compute_holospectra.py --dataset both --n-workers 32
"""
import argparse
import glob
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE_DIR = "results/hhsa/cache"
OUT_DIR = "results/hhsa/holospectra"
FS = 200.0
WIN_SAMP = 12000  # 60s at 200Hz (per Ho et al. 2026 methodology)


def _compute_one_window(args):
    """Worker: compute holospectrum for one (channel, window) pair."""
    try:
        IF_win, IA_win, seed = args
        import pipeline.hhsa as hhsa
        res = hhsa.compute_holospectrum_from_l1(
            IF_win, IA_win, FS, noise_seed=seed,
        )
        return res.holospectrum.astype(np.float32)
    except Exception:
        return None


def process_recording(cache_path, out_path, n_workers):
    """Load one cached recording, compute holospectra for all windows × channels.

    Processes sequentially (no inner pool) to avoid nested spawn issues.
    Uses a persistent pool passed from main() instead.
    """
    if os.path.exists(out_path):
        return "cached"

    d = np.load(cache_path)
    IF = d["IF"]    # (n_ch, n_samp, n_imf)
    IA = d["IA"]    # (n_ch, n_samp, n_imf)
    ch_names = d["ch_names"]
    n_ch, n_samp, n_imf = IF.shape

    # Window into 5s non-overlapping slices
    n_win = n_samp // WIN_SAMP
    if n_win == 0:
        return "too_short"

    # Build task list: (IF_window, IA_window, seed) — each ~64KB
    tasks = []
    for wi in range(n_win):
        s = wi * WIN_SAMP
        e = s + WIN_SAMP
        for ci in range(n_ch):
            IF_win = np.ascontiguousarray(IF[ci, s:e, :])
            IA_win = np.ascontiguousarray(IA[ci, s:e, :])
            seed = wi * n_ch * 10000 + ci * 10000
            tasks.append((IF_win, IA_win, seed))
    del IF, IA, d

    # Run Layer 2 in parallel using a fresh pool per recording
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(n_workers, len(tasks))) as pool:
        results = pool.map(_compute_one_window, tasks, chunksize=4)

    # Get grid shape from first non-None result
    n_fc = n_fa = None
    for r in results:
        if r is not None:
            n_fc, n_fa = r.shape
            break
    if n_fc is None:
        return "all_failed"

    # Reshape: (n_win, n_ch, n_fc, n_fa)
    H_all = np.zeros((n_win, n_ch, n_fc, n_fa), dtype=np.float32)
    idx = 0
    for wi in range(n_win):
        for ci in range(n_ch):
            if results[idx] is not None:
                H_all[wi, ci] = results[idx]
            idx += 1
    del results

    # Channel-aggregate (geometric mean)
    eps = 1e-12
    H_agg = np.exp(np.mean(np.log(H_all + eps), axis=1))  # (n_win, n_fc, n_fa)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        H_windows=H_all,
        H_chan_agg=H_agg,
        ch_names=ch_names,
        fs=FS,
        win_samp=WIN_SAMP,
        n_windows=n_win,
    )
    size_mb = os.path.getsize(out_path) / 1e6
    del H_all, H_agg
    return f"ok:{n_win}win,{n_ch}ch,{size_mb:.1f}MB"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["stress", "eegmat", "sam40", "meditation",
                                              "sleepdep", "both", "all"], default="both")
    parser.add_argument("--n-workers", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    ds_list = []
    if args.dataset == "both":
        ds_list = ["stress", "eegmat"]
    elif args.dataset == "all":
        ds_list = ["stress", "eegmat", "sam40", "meditation", "sleep_deprivation"]
    else:
        ds_list = [args.dataset if args.dataset != "sleepdep" else "sleep_deprivation"]

    cache_files = []
    for ds in ds_list:
        cache_files += [(f, f.replace(CACHE_DIR, OUT_DIR))
                        for f in sorted(glob.glob(f"{CACHE_DIR}/{ds}/*.npz"))]

    pending = [(c, o) for c, o in cache_files if not os.path.exists(o)]
    n_cached = len(cache_files) - len(pending)
    print(f"{n_cached} cached, {len(pending)} to process with {args.n_workers} workers")

    for i, (cache_path, out_path) in enumerate(pending):
        rec_id = os.path.splitext(os.path.basename(cache_path))[0]
        t_rec = time.time()
        status = process_recording(cache_path, out_path, args.n_workers)
        dt = time.time() - t_rec
        print(f"  [{time.time()-t0:6.0f}s] ({i+1}/{len(pending)}) {rec_id}: {status} ({dt:.0f}s)")

    total = time.time() - t0
    n_files = len(glob.glob(f"{OUT_DIR}/**/*.npz", recursive=True))
    print(f"\nDone: {n_files} holospectrum files, {total:.0f}s total")


if __name__ == "__main__":
    main()
