"""Classical ML Baseline for Stress Classification.

Extracts hand-crafted EEG features (band power, ratios, asymmetry) and
evaluates with the same subject-level 5-fold CV as FM experiments.

Usage:
    python train_classical.py --csv data/comprehensive_labels.csv --label subject-dass
    python train_classical.py --csv data/comprehensive_labels_stress.csv --label subject-dass --max-duration 400
"""

import argparse
import csv
import json
import os
import time
from collections import Counter
from datetime import datetime

import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold

import torch

# ──────────────────── Defaults ────────────────────
CSV_PATH = "data/comprehensive_labels.csv"
DATA_ROOT = "data"
SFREQ = 200.0

# 30 channels in standard order
CH_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FT7", "FC3", "FCz", "FC4", "FT8",
    "T3", "C3", "Cz", "C4", "T4",
    "TP7", "CP3", "CPz", "CP4", "TP8",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "Oz", "O2",
]

# Asymmetry pairs (right - left alpha power)
ASYM_PAIRS = [
    ("Fp1", "Fp2"),
    ("F3", "F4"),
    ("F7", "F8"),
    ("C3", "C4"),
    ("P3", "P4"),
    ("O1", "O2"),
]

BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}


def parse_args():
    p = argparse.ArgumentParser(description="Classical ML Stress Classification")
    p.add_argument("--csv", default=CSV_PATH)
    p.add_argument("--label", choices=["dass", "subject-dass", "dss"], default="subject-dass")
    p.add_argument("--max-duration", type=float, default=None)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=60)
    return p.parse_args()


def extract_features(epochs: np.ndarray, sfreq: float = 200.0) -> np.ndarray:
    """Extract hand-crafted EEG features from epochs.

    Args:
        epochs: (M, C, T) — raw EEG epochs (not z-scored)
        sfreq: sampling frequency

    Returns:
        features: (n_features,) — concatenated feature vector
    """
    M, C, T = epochs.shape

    # Average PSD across all epochs
    mean_epoch = epochs.mean(axis=0)  # (C, T)
    freqs, psd = welch(mean_epoch, fs=sfreq, nperseg=min(256, T))  # (C, n_freqs)

    features = []
    band_powers = {}

    # 1. Band power per channel (30 × 3 = 90 features)
    for band_name, (flo, fhi) in BANDS.items():
        band_mask = (freqs >= flo) & (freqs < fhi)
        bp = psd[:, band_mask].mean(axis=1)  # (C,) mean power in band
        band_powers[band_name] = bp
        features.append(bp)

    # 2. Band power ratios per channel (30 × 2 = 60 features)
    # Theta/Alpha ratio
    tar = band_powers["theta"] / (band_powers["alpha"] + 1e-10)
    features.append(tar)
    # Theta/Beta ratio
    tbr = band_powers["theta"] / (band_powers["beta"] + 1e-10)
    features.append(tbr)

    # 3. Frontal asymmetry — log(right) - log(left) alpha power (6 features)
    alpha_bp = band_powers["alpha"]
    for left_ch, right_ch in ASYM_PAIRS:
        if left_ch in CH_NAMES and right_ch in CH_NAMES:
            li = CH_NAMES.index(left_ch)
            ri = CH_NAMES.index(right_ch)
            asym = np.log(alpha_bp[ri] + 1e-10) - np.log(alpha_bp[li] + 1e-10)
            features.append(np.array([asym]))

    return np.concatenate(features)


def main():
    args = parse_args()
    print(f"Classical ML Baseline | CSV: {args.csv} | Label: {args.label} | Folds: {args.folds}")

    # ── Load dataset ──
    from pipeline.dataset import StressEEGDataset
    dataset = StressEEGDataset(
        args.csv, DATA_ROOT, norm="none", max_duration=args.max_duration,
    )

    patient_ids = dataset.get_patient_ids()

    # Labels
    if args.label == "subject-dass":
        increase_pids = set(
            r["patient_id"] for r in dataset.records if r["baseline_label"] == 1
        )
        labels = np.array([
            1 if r["patient_id"] in increase_pids else 0
            for r in dataset.records
        ])
        n1, n0 = int(labels.sum()), int(len(labels) - labels.sum())
        print(f"Subject-DASS labels: increase={n1}, normal={n0}")
    elif args.label == "dss":
        threshold_norm = args.threshold / 100.0
        labels = np.array([
            1 if r["stress_score"] >= threshold_norm else 0
            for r in dataset.records
        ])
    else:
        labels = dataset.get_labels()

    # ── Extract features ──
    print("\nExtracting features...")
    t0 = time.time()
    all_features = []
    for idx in range(len(dataset)):
        rec = dataset.records[idx]
        cache_path = os.path.join(dataset.cache_dir, rec["cache_name"])
        epochs = torch.load(cache_path, weights_only=True).numpy()
        feats = extract_features(epochs, SFREQ)
        all_features.append(feats)

    X = np.stack(all_features)  # (N, n_features)
    print(f"Features: {X.shape} extracted in {time.time()-t0:.1f}s")

    feature_names = []
    for band in BANDS:
        feature_names.extend([f"{band}_{ch}" for ch in CH_NAMES])
    feature_names.extend([f"tar_{ch}" for ch in CH_NAMES])
    feature_names.extend([f"tbr_{ch}" for ch in CH_NAMES])
    for left, right in ASYM_PAIRS:
        feature_names.append(f"asym_{left}_{right}")

    # ── Models ──
    models = {
        "LogReg_L2": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, penalty="l2", max_iter=1000,
                                       random_state=args.seed, class_weight="balanced")),
        ]),
        "LogReg_L1": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, penalty="l1", solver="saga",
                                       max_iter=1000, random_state=args.seed,
                                       class_weight="balanced")),
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                        class_weight="balanced", random_state=args.seed)),
        ]),
        "RF": RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=3,
            class_weight="balanced", random_state=args.seed, n_jobs=-1,
        ),
        "XGBoost": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=3, random_state=args.seed,
        ),
    }

    # ── CV ──
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    results_dir = os.path.join(
        "results",
        f"{datetime.now().strftime('%Y%m%d_%H%M')}_classical_{args.label}",
    )
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}

    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        y_true_all, y_pred_all = [], []
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            cv.split(X, labels, groups=patient_ids)
        ):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            ba = balanced_accuracy_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)

            train_pids = np.unique(patient_ids[train_idx])
            test_pids = np.unique(patient_ids[test_idx])
            print(f"  Fold {fold_idx+1}: train={len(train_idx)} ({len(train_pids)} subj), "
                  f"test={len(test_idx)} ({len(test_pids)} subj) | "
                  f"acc={acc:.3f} bal_acc={ba:.3f}")

            y_true_all.extend(y_test.tolist())
            y_pred_all.extend(y_pred.tolist())
            fold_results.append({"fold": fold_idx+1, "acc": acc, "bal_acc": ba})

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        global_ba = balanced_accuracy_score(y_true_all, y_pred_all)
        global_acc = accuracy_score(y_true_all, y_pred_all)
        global_f1 = f1_score(y_true_all, y_pred_all, average="weighted")
        global_kappa = cohen_kappa_score(y_true_all, y_pred_all)

        print(f"\n  Global: acc={global_acc:.4f} bal_acc={global_ba:.4f} "
              f"f1={global_f1:.4f} kappa={global_kappa:.4f}")

        all_results[model_name] = {
            "acc": global_acc,
            "bal_acc": global_ba,
            "f1": global_f1,
            "kappa": global_kappa,
            "folds": fold_results,
        }

        # Save per-sample predictions
        pred_path = os.path.join(results_dir, f"predictions_{model_name}.csv")
        with open(pred_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["y_true", "y_pred", "patient_id"])
            for yt, yp, pid in zip(y_true_all, y_pred_all,
                                    patient_ids[np.concatenate([
                                        test_idx for _, test_idx in
                                        cv.split(X, labels, groups=patient_ids)
                                    ])]):
                w.writerow([int(yt), int(yp), int(pid)])

    # ── Feature importance (from RF) ──
    print(f"\n{'='*60}")
    print("Top 20 Features (Random Forest importance)")
    print(f"{'='*60}")
    rf_model = models["RF"]
    rf_model.fit(X, labels)  # Fit on all data for feature importance
    importances = rf_model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    for rank, i in enumerate(top_idx):
        print(f"  {rank+1:2d}. {feature_names[i]:30s} importance={importances[i]:.4f}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY — Classical ML vs FM Baselines")
    print(f"{'='*60}")
    print(f"{'Model':20s} {'Acc':>8s} {'Bal_Acc':>8s} {'F1':>8s} {'Kappa':>8s}")
    print("-" * 56)
    for name, res in sorted(all_results.items(), key=lambda x: -x[1]["bal_acc"]):
        print(f"{name:20s} {res['acc']:8.4f} {res['bal_acc']:8.4f} "
              f"{res['f1']:8.4f} {res['kappa']:8.4f}")

    # Save summary
    summary = {
        "features": {"n_features": X.shape[1], "n_samples": X.shape[0]},
        "models": all_results,
        "config": vars(args),
    }
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
