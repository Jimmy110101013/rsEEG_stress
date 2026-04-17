"""Phase 1 WSCI sanity check: EEGMAT + Stress-DSS.

Expected outcome: WSCI(EEGMAT) >> WSCI(Stress-DSS).
If this fails, HHSA hypothesis is falsified — stop investing.

Usage:
    python scripts/run_wsci_phase1.py --fast        # N=24 ensemble, ~20 min
    python scripts/run_wsci_phase1.py --production  # N=100 ensemble, ~4 hr
"""
import argparse
import glob
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pipeline.hhsa as hhsa
from src.wsci import wsci_subject, wsci_dataset
from pipeline.hhsa import aggregate_channels_geometric

OUT_DIR = "results/wsci_phase1"


def set_hhsa_params(fast: bool):
    """Override HHSA module-level params for speed."""
    if fast:
        hhsa.N_ENSEMBLES_L1 = 24
        hhsa.N_ENSEMBLES_L2 = 12
    else:
        hhsa.N_ENSEMBLES_L1 = 100
        hhsa.N_ENSEMBLES_L2 = 24


def compute_single_epoch_channel(x: np.ndarray, fs: float, seed: int) -> np.ndarray:
    """Compute holospectrum for one (epoch, channel). Returns (n_fc, n_fa)."""
    res = hhsa.compute_holospectrum(x.astype(np.float64), fs, noise_seed=seed)
    return res.holospectrum.astype(np.float32)


def compute_recording_hhsa(
    epochs: np.ndarray,
    fs: float,
    base_seed: int,
    n_jobs: int,
) -> np.ndarray:
    """Compute holospectra for all (epoch, channel) in a recording.

    Parameters
    ----------
    epochs : ndarray, shape (n_epoch, n_ch, n_samp)
    fs : float
    base_seed : int
    n_jobs : int

    Returns
    -------
    H : ndarray, shape (n_epoch, n_ch, n_fc, n_fa)
    """
    n_ep, n_ch, _ = epochs.shape
    STRIDE = 10_000

    tasks = []
    for ei in range(n_ep):
        for ci in range(n_ch):
            seed = base_seed + (ei * n_ch + ci) * STRIDE
            tasks.append((epochs[ei, ci, :], fs, seed))

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(compute_single_epoch_channel)(x, fs, s)
        for x, fs, s in tasks
    )

    n_fc, n_fa = results[0].shape
    H = np.zeros((n_ep, n_ch, n_fc, n_fa), dtype=np.float32)
    idx = 0
    for ei in range(n_ep):
        for ci in range(n_ch):
            H[ei, ci] = results[idx]
            idx += 1
    return H


# ---------- EEGMAT data loader ----------

def load_eegmat_subjects() -> dict:
    """Returns {subject_id: {'rest': (n_ep, 19, 1000), 'arith': (n_ep, 19, 1000)}}."""
    cache_dir = "data/cache_eegmat"
    subjects = {}
    for f in sorted(glob.glob(f"{cache_dir}/eegmat_*_w5.0_sr200.0.pt")):
        bn = os.path.basename(f)
        parts = bn.replace("_w5.0_sr200.0.pt", "").split("_")
        sid = parts[1]  # Subject00, Subject01, ...
        cond = "rest" if parts[2] == "1" else "arith"
        epochs = torch.load(f, map_location="cpu", weights_only=True).numpy()
        subjects.setdefault(sid, {})[cond] = epochs
    return subjects


# ---------- Stress-DSS data loader ----------

def load_stress_dss_subjects(min_per_cond: int = 2) -> dict:
    """Returns {pid: {'above': (n_ep_above, 30, 1000), 'below': (n_ep_below, 30, 1000)}}.

    Uses personal-median DSS split. Only includes subjects with
    >= min_per_cond recordings in each condition.
    """
    df = pd.read_csv("data/comprehensive_labels.csv")
    cache_dir = "data/cache"

    # Build cache lookup: (group, pid_str, recording_id) → path
    cache_lookup = {}
    for f in sorted(glob.glob(f"{cache_dir}/*_w5.0.pt")):
        bn = os.path.basename(f).replace("_w5.0.pt", "")
        parts = bn.split("_")
        group, pid_str, rec_seq = parts[0], parts[1], int(parts[2])
        cache_lookup[(pid_str, rec_seq)] = f

    subjects = {}
    for pid in sorted(df["Patient_ID"].unique()):
        sub = df[df["Patient_ID"] == pid].copy()
        if len(sub) < 2 * min_per_cond:
            continue
        median_dss = sub["Stress_Score"].median()
        above_recs = sub[sub["Stress_Score"] >= median_dss]
        below_recs = sub[sub["Stress_Score"] < median_dss]
        if len(above_recs) < min_per_cond or len(below_recs) < min_per_cond:
            continue

        pid_str = f"p{pid:02d}"
        above_epochs, below_epochs = [], []
        for _, row in above_recs.iterrows():
            key = (pid_str, int(row["Recording_ID"]))
            if key in cache_lookup:
                ep = torch.load(cache_lookup[key], map_location="cpu", weights_only=True).numpy()
                above_epochs.append(ep)
        for _, row in below_recs.iterrows():
            key = (pid_str, int(row["Recording_ID"]))
            if key in cache_lookup:
                ep = torch.load(cache_lookup[key], map_location="cpu", weights_only=True).numpy()
                below_epochs.append(ep)

        if above_epochs and below_epochs:
            subjects[pid_str] = {
                "above": np.concatenate(above_epochs, axis=0),
                "below": np.concatenate(below_epochs, axis=0),
            }
    return subjects


# ---------- Main ----------

def run_dataset(name, subjects, cond_keys, fs, n_jobs, max_subjects=None):
    """Run HHSA + WSCI on one dataset.

    Parameters
    ----------
    name : str
    subjects : dict of {sid: {cond_key0: epochs, cond_key1: epochs}}
    cond_keys : (str, str) — condition 0 and 1 key names
    fs : float
    n_jobs : int
    max_subjects : int or None
    """
    cond0_key, cond1_key = cond_keys
    sids = sorted(subjects.keys())
    if max_subjects:
        sids = sids[:max_subjects]

    print(f"\n{'='*60}")
    print(f"Dataset: {name} ({len(sids)} subjects)")
    print(f"Conditions: {cond0_key} vs {cond1_key}")
    print(f"{'='*60}")

    wsci_values = []
    for i, sid in enumerate(sids):
        data = subjects[sid]
        ep0, ep1 = data[cond0_key], data[cond1_key]
        print(f"\n[{i+1}/{len(sids)}] Subject {sid}: "
              f"{cond0_key}={ep0.shape[0]} ep, {cond1_key}={ep1.shape[0]} ep, "
              f"ch={ep0.shape[1]}")

        t0 = time.time()
        H0 = compute_recording_hhsa(ep0, fs, base_seed=hash(sid) % 2**31, n_jobs=n_jobs)
        H1 = compute_recording_hhsa(ep1, fs, base_seed=(hash(sid)+1) % 2**31, n_jobs=n_jobs)
        t_hhsa = time.time() - t0

        # Channel aggregation (geometric mean)
        H0_agg = aggregate_channels_geometric(H0)  # (n_ep, n_fc, n_fa)
        H1_agg = aggregate_channels_geometric(H1)

        # WSCI
        sub_wsci = wsci_subject(H0_agg, H1_agg, n_perm=500, seed=i)
        wsci_values.append(sub_wsci.wsci)
        print(f"  HHSA: {t_hhsa:.0f}s | WSCI={sub_wsci.wsci:.4f} | "
              f"clusters={sub_wsci.n_surviving_clusters} | "
              f"null_p95={sub_wsci.null_max_mass_p95:.2f}")

    # Dataset summary
    w = np.array(wsci_values)
    ds = wsci_dataset(w, seed=42)
    print(f"\n--- {name} summary ---")
    print(f"  WSCI median: {ds.median:.4f}  CI95: [{ds.ci_lower:.4f}, {ds.ci_upper:.4f}]")
    print(f"  Per-subject: {[f'{v:.4f}' for v in w]}")

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(
        f"{OUT_DIR}/{name}_wsci.npz",
        wsci_per_subject=w,
        subject_ids=np.array(sids),
        median=ds.median,
        ci_lower=ds.ci_lower,
        ci_upper=ds.ci_upper,
    )
    print(f"  Saved to {OUT_DIR}/{name}_wsci.npz")
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Use N=24 ensemble (faster)")
    parser.add_argument("--production", action="store_true", help="Use N=100 ensemble")
    parser.add_argument("--n-jobs", type=int, default=64)
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Cap subjects per dataset (for quick tests)")
    parser.add_argument("--dataset", choices=["eegmat", "stress", "both"], default="both")
    args = parser.parse_args()

    set_hhsa_params(fast=not args.production)
    print(f"HHSA params: L1={hhsa.N_ENSEMBLES_L1}, L2={hhsa.N_ENSEMBLES_L2}")

    results = {}

    if args.dataset in ("eegmat", "both"):
        subjects = load_eegmat_subjects()
        results["eegmat"] = run_dataset(
            "eegmat", subjects, ("rest", "arith"), 200.0,
            args.n_jobs, args.max_subjects,
        )

    if args.dataset in ("stress", "both"):
        subjects = load_stress_dss_subjects(min_per_cond=2)
        results["stress_dss"] = run_dataset(
            "stress_dss", subjects, ("below", "above"), 200.0,
            args.n_jobs, args.max_subjects,
        )

    # Final comparison
    if "eegmat" in results and "stress_dss" in results:
        e, s = results["eegmat"], results["stress_dss"]
        print(f"\n{'='*60}")
        print(f"SANITY CHECK: WSCI(EEGMAT) vs WSCI(Stress-DSS)")
        print(f"  EEGMAT:     {e.median:.4f}  [{e.ci_lower:.4f}, {e.ci_upper:.4f}]")
        print(f"  Stress-DSS: {s.median:.4f}  [{s.ci_lower:.4f}, {s.ci_upper:.4f}]")
        if e.median > s.median:
            print(f"  → WSCI(EEGMAT) > WSCI(Stress): PASS ✓")
        else:
            print(f"  → WSCI(EEGMAT) <= WSCI(Stress): FAIL ✗ — HHSA hypothesis may be wrong")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
