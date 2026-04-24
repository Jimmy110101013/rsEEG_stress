"""Sanity-check RSA at per-window level (matches §5/§6/§8 LP protocol unit).

Compares to recording-avg RSA (current §4 Panel B source). If per-window RSA
preserves the subject_r >> label_r pattern, the §4 geometric claim is robust
to the representation unit.

Output: results/studies/exp06_fm_task_fitness/rsa_perwindow_sanity.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

REPO = Path("/raid/jupyter-linjimmy1003.md10/UCSD_stress")
CACHE = REPO / "results/features_cache"
OUT = REPO / "results/studies/exp06_fm_task_fitness/rsa_perwindow_sanity.json"

MODELS = ["labram", "cbramod", "reve"]
DATASETS = ["stress", "eegmat", "adftd", "tdbrain"]

# Subsample ceiling — pdist is O(N^2) in memory, spearmanr is O(N^2 log N) time.
# N=3000 → condensed vec ~4.5M entries → manageable (~30s per cell).
MAX_N = 3000
RNG = np.random.default_rng(0)


def stress_mixed_filter(y, p):
    """Drop Stress subjects that have recordings in BOTH DASS classes
    (matches fig_variance_atlas convention)."""
    mixed = {pid for pid in np.unique(p) if len(np.unique(y[p == pid])) > 1}
    keep = np.array([pid not in mixed for pid in p])
    return keep


def compute_rsa(feats, labels, pids):
    n = len(feats)
    if n > MAX_N:
        idx = RNG.choice(n, size=MAX_N, replace=False)
        feats, labels, pids = feats[idx], labels[idx], pids[idx]
        n = MAX_N
    # condensed distances (N*(N-1)/2)
    feat_d = pdist(feats, metric="cosine")
    # build condensed RDMs from broadcasting
    lab = np.asarray(labels)
    pid = np.asarray(pids)
    lab_rdm_full = (lab[:, None] != lab[None, :]).astype(np.int8)
    sub_rdm_full = (pid[:, None] != pid[None, :]).astype(np.int8)
    iu = np.triu_indices(n, k=1)
    lab_d = lab_rdm_full[iu]
    sub_d = sub_rdm_full[iu]
    r_lab, p_lab = spearmanr(feat_d, lab_d)
    r_sub, p_sub = spearmanr(feat_d, sub_d)
    return {
        "n_used": int(n),
        "rsa_label_r": float(r_lab),
        "rsa_label_p": float(p_lab),
        "rsa_subject_r": float(r_sub),
        "rsa_subject_p": float(p_sub),
    }


def main():
    out = {}
    for m in MODELS:
        for d in DATASETS:
            pw = CACHE / f"frozen_{m}_{d}_perwindow.npz"
            if not pw.exists():
                print(f"  SKIP {m}×{d}: {pw.name} missing")
                continue
            npz = np.load(pw, allow_pickle=True)
            f = npz["features"]
            y = npz["window_labels"]
            p = npz["window_pids"]
            if d == "stress":
                keep = stress_mixed_filter(y, p)
                f, y, p = f[keep], y[keep], p[keep]
            r_pw = compute_rsa(f, y, p)
            # recording-avg comparison from the same window cache
            rec_ids = npz["window_rec_idx"][:len(npz["features"])]
            # re-apply same filter
            if d == "stress":
                rec_ids_f = rec_ids[keep]
            else:
                rec_ids_f = rec_ids
            # average per recording
            uniq_recs = np.unique(rec_ids_f)
            f_rec = np.stack([f[rec_ids_f == r].mean(axis=0) for r in uniq_recs])
            y_rec = np.array([y[rec_ids_f == r][0] for r in uniq_recs])
            p_rec = np.array([p[rec_ids_f == r][0] for r in uniq_recs])
            r_ra = compute_rsa(f_rec, y_rec, p_rec)

            key = f"{m}_{d}"
            out[key] = {"per_window": r_pw, "recording_avg_from_pw": r_ra}
            print(f"  {key:22s}  pw: lab={r_pw['rsa_label_r']:+.3f} "
                  f"sub={r_pw['rsa_subject_r']:+.3f} (n={r_pw['n_used']})  "
                  f"| rec-avg: lab={r_ra['rsa_label_r']:+.3f} "
                  f"sub={r_ra['rsa_subject_r']:+.3f} (n={r_ra['n_used']})")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
