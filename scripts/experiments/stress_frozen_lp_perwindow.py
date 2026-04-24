"""Per-window frozen LP on Stress with prediction pooling.

Matches FT's protocol: LogReg trained on window-level samples (not recording-
level mean-pooled features), test-set window probabilities pooled per recording
for recording-level BA.

Protocol:
  1. Load per-window features from
     `results/features_cache/frozen_{model}_stress_perwindow.npz`.
  2. CV at RECORDING level: StratifiedGroupKFold(5) with groups=patient_id
     and label=recording_label, using recording indices.
  3. For each fold: all train-recording windows are training samples; all test-
     recording windows are scored; per-window probabilities mean-pooled to give
     one probability per test recording; threshold at 0.5.
  4. Report recording-level BA per seed, 8 seeds total.

Run:
    PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
    $PY scripts/experiments/stress_frozen_lp_perwindow.py --extractor labram
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
from sklearn.preprocessing import StandardScaler


SEEDS = [42, 123, 2024, 7, 0, 1, 99, 31337]


def eval_seed(
    window_feats: np.ndarray,
    window_rec_idx: np.ndarray,
    rec_labels: np.ndarray,
    rec_pids: np.ndarray,
    seed: int,
) -> tuple[float, np.ndarray]:
    """Run per-window LP with prediction pooling for one seed.

    Returns (recording-level BA, per-recording pooled prob).
    """
    n_rec = len(rec_labels)
    rec_indices = np.arange(n_rec)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    rec_pred = np.zeros(n_rec, dtype=int)
    rec_prob = np.full(n_rec, np.nan, dtype=float)

    for train_rec, test_rec in cv.split(rec_indices, rec_labels, groups=rec_pids):
        train_mask = np.isin(window_rec_idx, train_rec)
        test_mask = np.isin(window_rec_idx, test_rec)

        X_tr = window_feats[train_mask]
        y_tr = rec_labels[window_rec_idx[train_mask]]
        X_te = window_feats[test_mask]
        te_rec = window_rec_idx[test_mask]

        clf = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
        ])
        clf.fit(X_tr, y_tr)
        prob_te = clf.predict_proba(X_te)[:, 1]

        # Pool window prob per recording (mean prob → threshold)
        for r in test_rec:
            m = te_rec == r
            if m.any():
                rec_prob[r] = float(prob_te[m].mean())
                rec_pred[r] = int(rec_prob[r] >= 0.5)

    ba = float(balanced_accuracy_score(rec_labels, rec_pred))
    return ba, rec_prob


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
        f"results/studies/exp03_stress_erosion/frozen_lp_perwindow/"
        f"{model}_multi_seed.json"
    )

    data = np.load(features_npz)
    window_feats = data["features"]
    window_rec_idx = data["window_rec_idx"]
    rec_labels = data["rec_labels"]
    rec_pids = data["rec_pids"]
    rec_n_epochs = data["rec_n_epochs"]

    print(f"{model} per-window LP")
    print(f"  features: {window_feats.shape}")
    print(f"  n_rec={len(rec_labels)}, n_subj={len(np.unique(rec_pids))}")
    print(f"  total windows={window_feats.shape[0]} "
          f"(avg {rec_n_epochs.mean():.1f} per recording)")
    print(f"  labels: pos={int(rec_labels.sum())}, neg={int((1-rec_labels).sum())}")

    per_seed = {}
    per_seed_prob = {}
    for s in SEEDS:
        ba, prob = eval_seed(window_feats, window_rec_idx, rec_labels, rec_pids, s)
        per_seed[str(s)] = ba
        per_seed_prob[str(s)] = prob.tolist()
        print(f"  seed={s:>5}  BA={ba:.4f}")

    vals = np.array(list(per_seed.values()))

    out = {
        "extractor": model,
        "source_features": features_npz,
        "labels_source": "data/comprehensive_labels.csv Group column "
                         "(per-recording DASS class)",
        "protocol": (
            "Per-window LogisticRegression (C=1.0, class_weight=balanced) "
            "on StandardScaler-normed per-window features. "
            "Recording-level StratifiedGroupKFold(5) with groups=patient_id; "
            "window-level training within each fold; test-set window probs "
            "mean-pooled per recording; threshold 0.5; recording-level BA."
        ),
        "n_recordings": int(len(rec_labels)),
        "n_subjects": int(len(np.unique(rec_pids))),
        "n_positive": int(rec_labels.sum()),
        "n_negative": int((1 - rec_labels).sum()),
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
    print(f"  3-seed mean={out['mean_3seed_42_123_2024']:.4f} "
          f"std={out['std_3seed_42_123_2024_ddof1']:.4f}")


if __name__ == "__main__":
    main()
