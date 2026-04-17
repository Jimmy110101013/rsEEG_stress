"""M2: Re-run XGBoost (GradientBoostingClassifier) with explicit
sample_weight='balanced' to address R1 C3 (is classical's ≤chance an imbalance
artifact?). The other 4 classical methods already use class_weight='balanced';
only XGBoost (sklearn GradientBoostingClassifier) lacks that param natively.

Output: results/studies/exp02_classical_dass/rerun_70rec_xgb_balanced/summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import compute_sample_weight

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.dataset import StressEEGDataset  # noqa
from train_classical import extract_features, SFREQ  # noqa

CSV = "data/comprehensive_labels.csv"
DATA_ROOT = "data"


def main():
    out_dir = "results/studies/exp02_classical_dass/rerun_70rec_xgb_balanced"
    os.makedirs(out_dir, exist_ok=True)

    ds = StressEEGDataset(CSV, DATA_ROOT, norm="none", max_duration=None)
    patient_ids = np.array(ds.get_patient_ids())
    labels = np.array(ds.get_labels())
    print(f"N={len(ds)}, pos={labels.sum()}, neg={(labels==0).sum()}")

    t0 = time.time()
    feats = []
    for i in range(len(ds)):
        rec = ds.records[i]
        epochs = torch.load(
            os.path.join(ds.cache_dir, rec["cache_name"]), weights_only=True
        ).numpy()
        feats.append(extract_features(epochs, SFREQ))
    X = np.stack(feats)
    print(f"Features {X.shape} in {time.time()-t0:.1f}s")

    results = {}
    for variant in ["plain", "balanced"]:
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=3, random_state=42,
        )
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        fold_ba = []
        y_all, p_all = [], []

        for train_idx, test_idx in cv.split(X, labels, groups=patient_ids):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = labels[train_idx], labels[test_idx]

            if variant == "balanced":
                sw = compute_sample_weight("balanced", y_tr)
                model.fit(X_tr, y_tr, sample_weight=sw)
            else:
                model.fit(X_tr, y_tr)

            y_pred = model.predict(X_te)
            fold_ba.append(balanced_accuracy_score(y_te, y_pred))
            y_all.extend(y_te.tolist())
            p_all.extend(y_pred.tolist())

        results[variant] = {
            "bal_acc_mean": float(np.mean(fold_ba)),
            "bal_acc_std": float(np.std(fold_ba, ddof=1)),
            "bal_acc_folds": [float(x) for x in fold_ba],
            "pooled_bal_acc": float(balanced_accuracy_score(y_all, p_all)),
        }
        print(f"\n[{variant:8s}] 5-fold BA: {results[variant]['bal_acc_mean']:.4f} "
              f"± {results[variant]['bal_acc_std']:.4f} | "
              f"pooled: {results[variant]['pooled_bal_acc']:.4f}")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_dir}/summary.json")

    delta = results["balanced"]["bal_acc_mean"] - results["plain"]["bal_acc_mean"]
    print(f"\nΔ(balanced − plain) = {delta:+.4f}")
    if abs(delta) < 0.03:
        print("VERDICT: class imbalance effect < 3pp → R1 C3 concern addressed")
    else:
        print(f"VERDICT: class imbalance has {delta:+.1%} effect — revisit F-B framing")


if __name__ == "__main__":
    main()
