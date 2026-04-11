"""Multi-seed frozen linear probe on Stress per-recording DASS labels.

Loads frozen features from `results/features_cache/frozen_{model}_stress_19ch.npz`,
pairs them with per-recording Group column from `data/comprehensive_labels.csv`,
and runs subject-level StratifiedGroupKFold(5) logistic regression across
multiple seeds.

Run from project root:
    PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
    $PY scripts/stress_frozen_lp_multiseed.py                    # default: labram
    $PY scripts/stress_frozen_lp_multiseed.py --extractor cbramod
    $PY scripts/stress_frozen_lp_multiseed.py --extractor reve
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEEDS = [42, 123, 2024, 7, 0, 1, 99, 31337]
LABELS_CSV = "data/comprehensive_labels.csv"


def eval_seed(F: np.ndarray, labels: np.ndarray, pids: np.ndarray, seed: int) -> float:
    """Subject-level 5-fold logistic regression BA for one seed."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    y_pred = np.zeros_like(labels)
    for train_idx, test_idx in cv.split(F, labels, groups=pids):
        clf = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
        ])
        clf.fit(F[train_idx], labels[train_idx])
        y_pred[test_idx] = clf.predict(F[test_idx])
    return float(balanced_accuracy_score(labels, y_pred))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", default="labram",
                        choices=["labram", "cbramod", "reve"])
    args = parser.parse_args()

    model = args.extractor
    features_npz = f"results/features_cache/frozen_{model}_stress_19ch.npz"
    out_path = Path(f"results/studies/2026-04-10_stress_erosion/frozen_lp/{model}_multi_seed.json")

    F = np.load(features_npz)["features"]
    csv = pd.read_csv(LABELS_CSV)
    assert len(csv) == F.shape[0], f"row count mismatch: csv={len(csv)} feat={F.shape[0]}"

    labels = (csv["Group"] == "increase").astype(int).to_numpy()
    pids = csv["Patient_ID"].to_numpy()

    per_seed = {str(s): eval_seed(F, labels, pids, s) for s in SEEDS}
    vals = np.array(list(per_seed.values()))

    out = {
        "extractor": model,
        "source_features": features_npz,
        "labels_source": f"{LABELS_CSV} Group column (per-recording DASS class)",
        "protocol": f"Subject-level StratifiedGroupKFold(5), LogisticRegression "
                    f"(C=1.0, class_weight=balanced) on StandardScaler-normed "
                    f"{F.shape[1]}-d {model} features.",
        "n_recordings": int(F.shape[0]),
        "embed_dim": int(F.shape[1]),
        "n_subjects": int(len(np.unique(pids))),
        "n_positive": int(labels.sum()),
        "n_negative": int((1 - labels).sum()),
        "seeds": SEEDS,
        "per_seed_ba": per_seed,
        "mean_8seed": float(vals.mean()),
        "std_8seed": float(vals.std()),
        "mean_3seed_42_123_2024": float(vals[:3].mean()),
        "std_3seed_42_123_2024": float(vals[:3].std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"→ {out_path}")
    print(f"{model} frozen LP ({F.shape[1]}-d)")
    print(f"  8-seed mean={out['mean_8seed']:.4f} std={out['std_8seed']:.4f}")
    print(f"  3-seed mean={out['mean_3seed_42_123_2024']:.4f} "
          f"std={out['std_3seed_42_123_2024']:.4f}")


if __name__ == "__main__":
    main()
