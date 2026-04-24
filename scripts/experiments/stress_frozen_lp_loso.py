"""Per-window frozen LP on Stress with LOSO (leave-one-subject-out).

Robustness check against §5.2 5-fold StratifiedGroupKFold. LOSO provides 17
test partitions (one per subject) vs 5 fixed partitions, which is tighter for
small-N cohorts. Each test fold holds out ALL recordings of one subject.

Reuses the per-window feature cache from frozen_{model}_stress_perwindow.npz
produced earlier. Same apples-to-apples LP protocol as 5-fold:
  - Per-window training within training subjects
  - Per-dim 1-99 percentile clip (train-fit) + StandardScaler
  - LogisticRegression(liblinear, class_weight=balanced, C=1.0)
  - Test-set window probabilities mean-pooled per recording → threshold 0.5
  - Recording-level balanced accuracy pooled across all 17 folds.

Usage:
    PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
    $PY scripts/experiments/stress_frozen_lp_loso.py --extractor labram
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


# For LOSO there's no shuffle, but we still do multiple seeds because the
# LogReg/StandardScaler are deterministic — seed only affects ties in prob
# aggregation (very rare). So a single "seed" pass is sufficient.
# We run 3 seeds anyway for consistency with FT reporting.
SEEDS = [42, 123, 2024, 7, 0, 1, 99, 31337]


def eval_loso(window_feats, window_rec_idx, rec_labels, rec_pids, seed):
    """Run per-window LP with LOSO (leave-one-subject-out) and prediction pooling."""
    n_rec = len(rec_labels)
    rec_pred = np.zeros(n_rec, dtype=int)
    rec_prob = np.full(n_rec, np.nan, dtype=float)

    # LOSO at subject level: each fold holds out all recordings of one subject
    # Groups here = per-recording subject ids of size n_rec
    rec_indices = np.arange(n_rec)
    logo = LeaveOneGroupOut()
    for train_rec, test_rec in logo.split(rec_indices, rec_labels, groups=rec_pids):
        train_mask = np.isin(window_rec_idx, train_rec)
        test_mask = np.isin(window_rec_idx, test_rec)

        X_tr = window_feats[train_mask]
        y_tr = rec_labels[window_rec_idx[train_mask]]
        X_te = window_feats[test_mask]
        te_rec = window_rec_idx[test_mask]

        # Percentile clip + StandardScaler (same protocol as 5-fold)
        lo = np.percentile(X_tr, 1, axis=0)
        hi = np.percentile(X_tr, 99, axis=0)
        X_tr_c = np.clip(X_tr, lo, hi)
        X_te_c = np.clip(X_te, lo, hi)

        sc = StandardScaler().fit(X_tr_c)
        X_tr_s = sc.transform(X_tr_c)
        X_te_s = sc.transform(X_te_c)

        clf = LogisticRegression(
            max_iter=5000, class_weight="balanced", C=1.0,
            solver="liblinear", tol=1e-3, random_state=seed,
        )
        clf.fit(X_tr_s, y_tr)
        prob_te = clf.predict_proba(X_te_s)[:, 1]

        for r in test_rec:
            m = te_rec == r
            if m.any():
                rec_prob[r] = float(prob_te[m].mean())
                rec_pred[r] = int(rec_prob[r] >= 0.5)

    return float(balanced_accuracy_score(rec_labels, rec_pred)), rec_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", default="labram",
                        choices=["labram", "cbramod", "reve"])
    parser.add_argument("--features-npz", default=None)
    parser.add_argument("--out-path", default=None)
    args = parser.parse_args()

    model = args.extractor
    features_npz = args.features_npz or \
        f"results/features_cache/frozen_{model}_stress_perwindow.npz"
    out_path = Path(
        args.out_path or
        f"results/studies/perwindow_lp_all/stress/{model}_loso.json"
    )

    data = np.load(features_npz)
    window_feats = data["features"]
    window_rec_idx = data["window_rec_idx"]
    rec_labels = data["rec_labels"]
    rec_pids = data["rec_pids"]
    rec_n_epochs = data["rec_n_epochs"]

    n_rec = len(rec_labels)
    n_subj = len(np.unique(rec_pids))

    print(f"{model} × stress LOSO per-window LP")
    print(f"  features: {window_feats.shape}")
    print(f"  n_rec={n_rec}, n_subj={n_subj} → {n_subj} LOSO folds")
    print(f"  pos={int(rec_labels.sum())}, neg={int((1-rec_labels).sum())}")

    per_seed = {}
    for s in SEEDS:
        ba, _ = eval_loso(window_feats, window_rec_idx, rec_labels, rec_pids, s)
        per_seed[str(s)] = ba
        print(f"  seed={s:>5}  LOSO BA={ba:.4f}")

    vals = np.array(list(per_seed.values()))

    out = {
        "extractor": model,
        "dataset": "stress",
        "source_features": features_npz,
        "protocol": (
            "Per-window LogisticRegression (liblinear, C=1.0, class_weight=balanced) "
            "on percentile-clipped + StandardScaler features. "
            "LeaveOneGroupOut at subject level (17 folds); window-level training "
            "within each fold; test-set window probs mean-pooled per recording; "
            "threshold 0.5; recording-level BA."
        ),
        "cv": "LOSO subject-level (17 folds)",
        "n_recordings": int(n_rec),
        "n_subjects": int(n_subj),
        "n_positive": int(rec_labels.sum()),
        "n_negative": int((1 - rec_labels).sum()),
        "embed_dim": int(window_feats.shape[1]),
        "total_windows": int(window_feats.shape[0]),
        "seeds": SEEDS,
        "per_seed_ba": per_seed,
        "mean_8seed": float(vals.mean()),
        "std_8seed_ddof1": float(vals.std(ddof=1)),
        "mean_3seed_42_123_2024": float(vals[:3].mean()),
        "std_3seed_42_123_2024_ddof1": float(vals[:3].std(ddof=1)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n→ {out_path}")
    print(f"  LOSO 8-seed mean={out['mean_8seed']:.4f} "
          f"std={out['std_8seed_ddof1']:.4f}")


if __name__ == "__main__":
    main()
