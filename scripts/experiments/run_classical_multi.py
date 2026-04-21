"""Classical ML baselines on Stress/EEGMAT/SleepDep (architecture ceiling).

Generalises train_classical.py to 19-channel datasets for Fig 6. Uses
subject-level 5-fold CV matching the FM pipeline. Per-rec labels
(not subject-aggregated).

Usage:
    $PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
    $PY scripts/experiments/run_classical_multi.py --dataset eegmat
    $PY scripts/experiments/run_classical_multi.py --dataset sleepdep
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path("/raid/jupyter-linjimmy1003.md10/UCSD_stress")
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import numpy as np
import torch
from scipy.signal import welch
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SFREQ = 200.0

BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}

# 19ch pair list uses COMMON_19 order
ASYM_PAIRS_30 = [("Fp1", "Fp2"), ("F3", "F4"), ("F7", "F8"),
                 ("C3", "C4"), ("P3", "P4"), ("O1", "O2")]
ASYM_PAIRS_19 = [("FP1", "FP2"), ("F3", "F4"), ("F7", "F8"),
                 ("C3", "C4"), ("P3", "P4"), ("O1", "O2")]


def load_dataset(name: str):
    """Return (epochs_list, labels, pids, ch_names, asym_pairs)."""
    if name == "stress":
        from pipeline.dataset import StressEEGDataset
        ds = StressEEGDataset(
            "data/comprehensive_labels.csv", "data",
            norm="none", max_duration=None,
        )
        pids = ds.get_patient_ids()
        labels = ds.get_labels()  # per-rec DASS
        ch = [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "FT7", "FC3", "FCz", "FC4", "FT8",
            "T3", "C3", "Cz", "C4", "T4",
            "TP7", "CP3", "CPz", "CP4", "TP8",
            "T5", "P3", "Pz", "P4", "T6",
            "O1", "Oz", "O2",
        ]
        epochs = []
        for rec in ds.records:
            cache_path = os.path.join(ds.cache_dir, rec["cache_name"])
            epochs.append(torch.load(cache_path, weights_only=True).numpy())
        return epochs, labels, pids, ch, ASYM_PAIRS_30

    if name == "eegmat":
        from pipeline.eegmat_dataset import EEGMATDataset
        from pipeline.common_channels import COMMON_19
        ds = EEGMATDataset("data/eegmat", norm="none",
                           cache_dir="data/cache_eegmat_nnone")
        pids = ds.get_patient_ids()
        labels = ds.get_labels()
        epochs = []
        for rec in ds.records:
            ep = ds._preprocess(rec).numpy()  # (M, 19, T) µV
            epochs.append(ep)
        return epochs, labels, pids, COMMON_19, ASYM_PAIRS_19

    if name == "sleepdep":
        from pipeline.sleepdep_dataset import SleepDepDataset
        from pipeline.common_channels import COMMON_19
        ds = SleepDepDataset("data/sleep_deprivation", norm="none",
                             cache_dir="data/cache_sleepdep_nnone")
        pids = ds.get_patient_ids()
        labels = ds.get_labels()
        epochs = []
        for rec in ds.records:
            ep = ds._preprocess(rec).numpy()
            epochs.append(ep)
        return epochs, labels, pids, COMMON_19, ASYM_PAIRS_19

    if name == "adftd":
        from pipeline.adftd_dataset import ADFTDDataset
        from pipeline.common_channels import COMMON_19
        ds = ADFTDDataset("data/adftd", binary=True,
                          window_sec=5.0, norm="none",
                          cache_dir="data/cache_adftd_nnone_w5")
        pids = ds.get_patient_ids()
        labels = ds.get_labels()
        epochs = []
        for rec in ds.records:
            ep = ds._preprocess(rec).numpy()
            epochs.append(ep)
        return epochs, labels, pids, COMMON_19, ASYM_PAIRS_19

    raise ValueError(f"unknown dataset {name}")


def extract_features(epochs, ch_names, asym_pairs, sfreq=SFREQ):
    """epochs: (M, C, T) → (n_features,) handcrafted feature vector."""
    M, C, T = epochs.shape
    mean_epoch = epochs.mean(axis=0)
    freqs, psd = welch(mean_epoch, fs=sfreq, nperseg=min(256, T))

    features = []
    band_powers = {}
    for band, (flo, fhi) in BANDS.items():
        mask = (freqs >= flo) & (freqs < fhi)
        bp = psd[:, mask].mean(axis=1)
        band_powers[band] = bp
        features.append(bp)

    features.append(band_powers["theta"] / (band_powers["alpha"] + 1e-10))
    features.append(band_powers["theta"] / (band_powers["beta"] + 1e-10))

    alpha_bp = band_powers["alpha"]
    upper_names = [n.upper() for n in ch_names]
    for left, right in asym_pairs:
        lu, ru = left.upper(), right.upper()
        if lu in upper_names and ru in upper_names:
            li, ri = upper_names.index(lu), upper_names.index(ru)
            asym = np.log(alpha_bp[ri] + 1e-10) - np.log(alpha_bp[li] + 1e-10)
            features.append(np.array([asym]))

    return np.concatenate(features)


def build_models(seed: int):
    return {
        "LogReg_L2": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, penalty="l2", max_iter=1000,
                                       random_state=seed,
                                       class_weight="balanced")),
        ]),
        "LogReg_L1": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, penalty="l1", solver="saga",
                                       max_iter=1000, random_state=seed,
                                       class_weight="balanced")),
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                        class_weight="balanced", random_state=seed)),
        ]),
        "RF": RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=3,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        ),
        "XGBoost": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=3, random_state=seed,
        ),
    }


def run(dataset_name: str, folds: int = 5, seeds=(42, 123, 2024)):
    print(f"{'='*60}")
    print(f"Classical ML × {dataset_name}  (seeds {list(seeds)})")
    print(f"{'='*60}")

    t0 = time.time()
    epochs_list, labels, pids, ch_names, asym_pairs = load_dataset(dataset_name)
    print(f"Loaded {len(epochs_list)} recordings, {len(np.unique(pids))} subjects "
          f"(pos={int(labels.sum())}, neg={int((labels == 0).sum())}) "
          f"in {time.time()-t0:.1f}s")

    print("Extracting features...")
    t0 = time.time()
    X = np.stack([extract_features(e, ch_names, asym_pairs) for e in epochs_list])
    print(f"X shape = {X.shape}  ({time.time()-t0:.1f}s)")

    out_dir = REPO / f"results/studies/exp02_classical_dass/{dataset_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed = {}
    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        models = build_models(seed)
        seed_results = {}
        for name, model in models.items():
            y_true_all, y_pred_all = [], []
            fold_rows = []
            for f, (tr, te) in enumerate(cv.split(X, labels, groups=pids)):
                model.fit(X[tr], labels[tr])
                pred = model.predict(X[te])
                ba = balanced_accuracy_score(labels[te], pred)
                acc = accuracy_score(labels[te], pred)
                fold_rows.append({"fold": f + 1, "acc": float(acc), "bal_acc": float(ba)})
                y_true_all.extend(labels[te].tolist())
                y_pred_all.extend(pred.tolist())
            gba = balanced_accuracy_score(y_true_all, y_pred_all)
            gacc = accuracy_score(y_true_all, y_pred_all)
            gf1 = f1_score(y_true_all, y_pred_all, average="weighted")
            gk = cohen_kappa_score(y_true_all, y_pred_all)
            seed_results[name] = {
                "acc": float(gacc), "bal_acc": float(gba),
                "f1": float(gf1), "kappa": float(gk),
                "folds": fold_rows,
            }
        per_seed[str(seed)] = seed_results
        print(f"  seed {seed}: " + ", ".join(
            f"{n}={seed_results[n]['bal_acc']:.3f}" for n in seed_results))

    # aggregate across seeds
    model_names = list(next(iter(per_seed.values())).keys())
    aggregated = {}
    for name in model_names:
        bas = [per_seed[str(s)][name]["bal_acc"] for s in seeds]
        aggregated[name] = {
            "mean_bal_acc": float(np.mean(bas)),
            "std_bal_acc": float(np.std(bas, ddof=1)),
            "bal_acc_per_seed": {str(s): float(per_seed[str(s)][name]["bal_acc"])
                                 for s in seeds},
        }
        print(f"  {name:12s}  BA = {aggregated[name]['mean_bal_acc']:.3f} "
              f"± {aggregated[name]['std_bal_acc']:.3f}")

    summary = {
        "dataset": dataset_name,
        "n_features": int(X.shape[1]),
        "n_recordings": int(X.shape[0]),
        "n_subjects": int(len(np.unique(pids))),
        "folds": folds,
        "seeds": list(seeds),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "aggregated": aggregated,
        "per_seed": per_seed,
    }
    out = out_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"→ {out.relative_to(REPO)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["stress", "eegmat", "sleepdep", "adftd"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    a = p.parse_args()
    run(a.dataset, folds=a.folds, seeds=tuple(a.seeds))


if __name__ == "__main__":
    main()
