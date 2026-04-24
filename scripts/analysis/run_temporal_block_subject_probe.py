"""Temporal-block subject-ID probe (uniform across 4 cells).

Unified subject-ID probe protocol that works on all four cells including ADFTD
(1 rec/subject). For each subject, concatenate all their windows in temporal
order and split into 5 contiguous blocks. Run 5-fold CV predicting subject
identity from frozen / FOOOF-ablated FM features. Report 3-seed mean BA.

Semantics per cell:
  EEGMAT / SleepDep / Stress: blocks span across recordings -> tests temporal
    stability + cross-session drift.
  ADFTD (1 rec/subject): blocks span within-recording -> tests within-session
    subject fingerprint stability only.

Chance rate = 1 / n_subjects. ADFTD = 1/65 = 0.0154; Stress-pure = 1/14 = 0.0714;
EEGMAT / SleepDep = 1/36 = 0.0278.

Output schema (matches existing fooof_ablation_probes.py structure):
  results/studies/exp33_temporal_block_probe/{cell}_probes.json
  {"dataset": <cell>, "protocol": "temporal_block_5fold",
   "results": {model: {cond: {"subject_probe_mean", "_std", "_bas",
                              "n_subjects", "chance_rate"}}}}

Usage:
  PY scripts/analysis/run_temporal_block_subject_probe.py --dataset eegmat
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]

SEEDS = [42]  # deterministic protocol; LDA is closed-form — multiple seeds redundant
N_BLOCKS = 5
# LDA (closed-form, O(d^3)) replaced lbfgs LogReg: 10k-window × 65-class benchmark
# went from 600s to <1s/fold. LDA shrinkage='auto' gives Ledoit-Wolf covariance
# regularization, making it robust to d ≈ n_samples / n_classes.


def clip_and_scale(X_tr, X_te):
    lo = np.percentile(X_tr, 1, axis=0)
    hi = np.percentile(X_tr, 99, axis=0)
    X_tr_c = np.clip(X_tr, lo, hi)
    X_te_c = np.clip(X_te, lo, hi)
    sc = StandardScaler().fit(X_tr_c)
    return sc.transform(X_tr_c), sc.transform(X_te_c)


def load_features(dataset: str, model: str, condition: str):
    """Load per-window features for the given condition.

    condition in {original, aperiodic_removed, periodic_removed, both_removed}.
    """
    if condition == "original":
        p = REPO / f"results/features_cache/frozen_{model}_{dataset}_perwindow.npz"
    else:
        p = REPO / f"results/features_cache/fooof_ablation/feat_{model}_{dataset}_{condition}.npz"
    d = np.load(p)
    return {
        "features": d["features"],
        "window_rec_idx": d["window_rec_idx"].astype(np.int64),
        "rec_labels": d["rec_labels"].astype(np.int64),
        "rec_pids": d["rec_pids"].astype(np.int64),
    }


def apply_stress_filter(data: dict) -> dict:
    """Keep only Stress subjects with consistent labels across recordings.

    Drops pids {3, 11, 17} which straddle both DASS classes (see CLAUDE.md §4).
    """
    keep_pids = set(int(p) for p in data["rec_pids"] if int(p) not in {3, 11, 17})
    rec_keep = np.array([int(p) in keep_pids for p in data["rec_pids"]])
    rec_old2new = -np.ones(len(data["rec_pids"]), dtype=np.int64)
    rec_old2new[rec_keep] = np.arange(rec_keep.sum())

    win_keep = rec_keep[data["window_rec_idx"]]
    return {
        "features": data["features"][win_keep],
        "window_rec_idx": rec_old2new[data["window_rec_idx"][win_keep]],
        "rec_labels": data["rec_labels"][rec_keep],
        "rec_pids": data["rec_pids"][rec_keep],
    }


def temporal_block_subject_probe(features, window_rec_idx, rec_pids, seed):
    """5-fold temporal-block multi-class subject-ID probe.

    For each subject, concatenate windows in the order they appear in the
    feature matrix (ordered by window_rec_idx then by original row order,
    which preserves within-recording temporal order), split into 5 contiguous
    blocks. 5-fold CV: hold out block f, train on other 4, predict subject ID.
    Report per-fold balanced_accuracy (macro-recall over subject classes),
    then average across 5 folds for this seed.
    """
    n_windows = len(window_rec_idx)
    window_pids = rec_pids[window_rec_idx]  # subject id per window

    unique_subjects = np.unique(window_pids)
    # Per-subject temporal order: windows appear in rec order in the cache,
    # and within a rec they are in temporal order (confirmed by upstream
    # extraction pipeline). So np.where(window_pids == pid) preserves that.
    subj_blocks = {}  # pid -> list of 5 np.arrays of window indices
    for pid in unique_subjects:
        idx = np.where(window_pids == pid)[0]
        if len(idx) < N_BLOCKS:
            # Too few windows to form 5 blocks; broadcast the available ones
            # (each block gets ceil distribution; leftover blocks are empty).
            blocks = [np.array([i], dtype=np.int64) if i < len(idx) else np.array([], dtype=np.int64)
                      for i in range(N_BLOCKS)]
            blocks = [idx[b] for b in blocks if len(b)]
            # pad with empty arrays if necessary (shouldn't happen in our datasets)
            while len(blocks) < N_BLOCKS:
                blocks.append(np.array([], dtype=np.int64))
        else:
            blocks = np.array_split(idx, N_BLOCKS)
        subj_blocks[int(pid)] = blocks

    fold_bas = []
    for fold_id in range(N_BLOCKS):
        test_idx_list = []
        train_idx_list = []
        for pid, blocks in subj_blocks.items():
            test_idx_list.append(blocks[fold_id])
            train_idx_list.append(np.concatenate(
                [blocks[f] for f in range(N_BLOCKS) if f != fold_id] or [np.array([], dtype=np.int64)]
            ))
        test_idx = np.concatenate(test_idx_list) if test_idx_list else np.array([], dtype=np.int64)
        train_idx = np.concatenate(train_idx_list) if train_idx_list else np.array([], dtype=np.int64)

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        X_tr = features[train_idx]
        y_tr = window_pids[train_idx]
        X_te = features[test_idx]
        y_te = window_pids[test_idx]

        # Drop any subject missing from train this fold (shouldn't happen with
        # split[5] but robustness check).
        present = np.isin(y_te, np.unique(y_tr))
        X_te = X_te[present]
        y_te = y_te[present]
        if len(y_te) == 0:
            continue

        X_tr_s, X_te_s = clip_and_scale(X_tr, X_te)
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_te_s)
        fold_bas.append(float(balanced_accuracy_score(y_te, y_pred)))

    return float(np.mean(fold_bas)) if fold_bas else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True,
                   choices=["stress", "eegmat", "sleepdep", "adftd"])
    p.add_argument("--models", nargs="+", default=["labram", "cbramod", "reve"])
    p.add_argument("--out-dir", default="results/studies/exp33_temporal_block_probe")
    args = p.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_probes.json"

    conditions = ["original", "aperiodic_removed", "periodic_removed", "both_removed"]

    results = {
        "dataset": args.dataset,
        "protocol": f"temporal_block_{N_BLOCKS}fold",
        "seeds": SEEDS,
        "results": {},
    }
    for model in args.models:
        results["results"][model] = {}
        for cond in conditions:
            print(f"[{args.dataset} | {model} | {cond}]", flush=True)
            try:
                d = load_features(args.dataset, model, cond)
            except FileNotFoundError as e:
                print(f"  MISSING: {e}")
                continue
            if args.dataset == "stress":
                d = apply_stress_filter(d)

            X = d["features"]
            rec_idx = d["window_rec_idx"]
            rec_pid = d["rec_pids"]
            n_subj = len(np.unique(rec_pid))
            chance = 1.0 / n_subj

            bas = []
            for s in SEEDS:
                ba = temporal_block_subject_probe(X, rec_idx, rec_pid, s)
                bas.append(ba)
                print(f"  seed={s}: BA={ba:.4f}", flush=True)

            arr = np.array(bas)
            entry = {
                "subject_probe_mean": float(arr.mean()),
                "subject_probe_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "subject_probe_bas": bas,
                "n_subjects": int(n_subj),
                "chance_rate": float(chance),
                "n_windows": int(len(X)),
                "feature_dim": int(X.shape[1]),
            }
            print(f"  mean={entry['subject_probe_mean']:.4f} ± "
                  f"{entry['subject_probe_std']:.4f} (chance={chance:.4f})", flush=True)
            results["results"][model][cond] = entry

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
