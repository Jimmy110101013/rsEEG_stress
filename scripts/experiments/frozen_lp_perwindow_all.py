"""Generic per-window frozen LP across datasets × FMs with prediction pooling.

Matches FT's protocol: LogReg trained on window-level samples, test-set window
probabilities pooled per recording for recording-level BA.

Usage:
    PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
    $PY scripts/experiments/frozen_lp_perwindow_all.py --extractor labram --dataset eegmat
    $PY scripts/experiments/frozen_lp_perwindow_all.py --extractor reve --dataset tdbrain
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


SEEDS = [42, 123, 2024, 7, 0, 1, 99, 31337]


def eval_seed(window_feats, window_rec_idx, rec_labels, rec_pids, seed, n_splits=5):
    n_rec = len(rec_labels)
    rec_indices = np.arange(n_rec)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rec_pred = np.zeros(n_rec, dtype=int)
    rec_prob = np.full(n_rec, np.nan, dtype=float)

    for train_rec, test_rec in cv.split(rec_indices, rec_labels, groups=rec_pids):
        train_mask = np.isin(window_rec_idx, train_rec)
        test_mask = np.isin(window_rec_idx, test_rec)

        X_tr = window_feats[train_mask]
        y_tr = rec_labels[window_rec_idx[train_mask]]
        X_te = window_feats[test_mask]
        te_rec = window_rec_idx[test_mask]

        # Clip raw features to per-dim 1-99 percentile BEFORE scaling.
        # REVE features on ADFTD/TDBRAIN contain extreme outliers (±650k)
        # that break LogReg convergence; training-set percentile clipping
        # removes these while preserving the bulk feature distribution.
        lo = np.percentile(X_tr, 1, axis=0)
        hi = np.percentile(X_tr, 99, axis=0)
        X_tr_c = np.clip(X_tr, lo, hi)
        X_te_c = np.clip(X_te, lo, hi)

        clf = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=5000, class_weight="balanced", C=1.0,
                solver="liblinear", tol=1e-3,
            )),
        ])
        clf.fit(X_tr_c, y_tr)
        prob_te = clf.predict_proba(X_te_c)[:, 1]

        for r in test_rec:
            m = te_rec == r
            if m.any():
                rec_prob[r] = float(prob_te[m].mean())
                rec_pred[r] = int(rec_prob[r] >= 0.5)

    ba = float(balanced_accuracy_score(rec_labels, rec_pred))
    return ba, rec_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", required=True,
                        choices=["labram", "cbramod", "reve"])
    parser.add_argument("--dataset", required=True,
                        choices=["stress", "eegmat", "adftd", "tdbrain",
                                 "meditation", "sleepdep"])
    parser.add_argument("--features-npz", default=None)
    parser.add_argument("--out-path", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    model = args.extractor
    dataset = args.dataset
    features_npz = args.features_npz or \
        f"results/features_cache/frozen_{model}_{dataset}_perwindow.npz"
    out_path = Path(
        args.out_path or
        f"results/studies/perwindow_lp_all/{dataset}/{model}_multi_seed.json"
    )

    data = np.load(features_npz)
    window_feats = data["features"]
    window_rec_idx = data["window_rec_idx"]
    rec_labels = data["rec_labels"]
    rec_pids = data["rec_pids"]
    rec_n_epochs = data["rec_n_epochs"]

    n_rec = len(rec_labels)
    n_subj = len(np.unique(rec_pids))
    pos = int(rec_labels.sum())
    neg = int((1 - rec_labels).sum())

    print(f"{model} × {dataset} per-window LP")
    print(f"  features: {window_feats.shape}")
    print(f"  n_rec={n_rec}, n_subj={n_subj}, pos={pos}, neg={neg}")
    print(f"  total windows={window_feats.shape[0]} "
          f"(avg {rec_n_epochs.mean():.1f} per recording)")

    # Adjust n_splits if folds can't be stratified
    min_class_count = min(pos, neg)
    n_splits = min(args.n_splits, min_class_count)
    if n_splits < args.n_splits:
        print(f"  Warning: reducing n_splits {args.n_splits} → {n_splits} "
              f"(min class size = {min_class_count})")

    per_seed = {}
    for s in SEEDS:
        ba, _ = eval_seed(window_feats, window_rec_idx, rec_labels, rec_pids,
                          s, n_splits=n_splits)
        per_seed[str(s)] = ba
        print(f"  seed={s:>5}  BA={ba:.4f}")

    vals = np.array(list(per_seed.values()))

    out = {
        "extractor": model,
        "dataset": dataset,
        "source_features": features_npz,
        "protocol": (
            f"Per-window LogisticRegression (C=1.0, class_weight=balanced) "
            f"on StandardScaler-normed per-window features. "
            f"Recording-level StratifiedGroupKFold({n_splits}) with groups=patient_id; "
            f"window-level training within each fold; test-set window probs "
            f"mean-pooled per recording; threshold 0.5; recording-level BA."
        ),
        "n_recordings": int(n_rec),
        "n_subjects": int(n_subj),
        "n_positive": pos,
        "n_negative": neg,
        "n_splits": n_splits,
        "embed_dim": int(window_feats.shape[1]),
        "total_windows": int(window_feats.shape[0]),
        "avg_windows_per_rec": float(rec_n_epochs.mean()),
        "seeds": SEEDS,
        "per_seed_ba": per_seed,
        "mean_8seed": float(vals.mean()),
        "std_8seed_ddof1": float(vals.std(ddof=1)),
        "std_8seed_ddof0": float(vals.std(ddof=0)),
        "mean_3seed_42_123_2024": float(vals[:3].mean()),
        "std_3seed_42_123_2024_ddof1": float(vals[:3].std(ddof=1)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n→ {out_path}")
    print(f"  8-seed mean={out['mean_8seed']:.4f} "
          f"std={out['std_8seed_ddof1']:.4f} (ddof=1)")


if __name__ == "__main__":
    main()
