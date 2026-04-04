"""Within-Subject Longitudinal Stress Evaluation.

Tests whether REVE features capture within-subject stress variation using
leave-one-out per subject. No cross-subject generalization required.

For each subject (n>=3 recordings):
  For each recording held out:
    A. Centroid: cosine similarity to "high" vs "low" calibration centroids
    B. 1-NN: nearest calibration recording by embedding distance
    C. Linear: fit small logistic regression on n-1, predict held-out

Label: above/below subject's personal median stress score.

Usage:
    python train_longitudinal.py --extractor reve --norm none --device cuda:4
    python train_longitudinal.py --extractor mock_fm --device cuda:4
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.linear_model import LogisticRegression

import baseline.mock_fm  # noqa: F401
import baseline.reve  # noqa: F401
from baseline.abstract import create_extractor
from pipeline.dataset import StressEEGDataset, stress_collate_fn
from src.model import DecoupledStressModel
from torch.utils.data import DataLoader, Subset

CSV_PATH = "data/comprehensive_labels.csv"
DATA_ROOT = "data"
EMBED_DIM = 512
MIN_RECORDINGS = 3


def parse_args():
    p = argparse.ArgumentParser(description="Within-Subject Longitudinal Stress Evaluation")
    p.add_argument("--extractor", default="mock_fm")
    p.add_argument("--device", default="cuda:4")
    p.add_argument("--norm", choices=["zscore", "none"], default="none")
    p.add_argument("--no-bf16", action="store_true")
    return p.parse_args()


def extract_all_features(model, dataset, device, use_amp):
    """Extract pooled embeddings for all recordings."""
    model.eval()
    loader = DataLoader(dataset, batch_size=4, shuffle=False,
                        collate_fn=stress_collate_fn, num_workers=0)
    all_feats = []
    with torch.no_grad():
        for epochs_batch, _labels, _scores, mask, _pids in loader:
            epochs_batch = epochs_batch.to(device)
            mask = mask.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pooled = model.extract_pooled(epochs_batch, mask)
            all_feats.append(pooled.float().cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def method_centroid(feats_cal, labels_cal, feat_test):
    """Classify by cosine similarity to high/low centroids."""
    high_mask = labels_cal == 1
    low_mask = labels_cal == 0
    if high_mask.sum() == 0 or low_mask.sum() == 0:
        return -1  # can't classify
    high_centroid = feats_cal[high_mask].mean(axis=0)
    low_centroid = feats_cal[low_mask].mean(axis=0)
    sim_high = cosine_sim(feat_test, high_centroid)
    sim_low = cosine_sim(feat_test, low_centroid)
    return 1 if sim_high > sim_low else 0


def method_1nn(feats_cal, labels_cal, feat_test):
    """Classify by nearest neighbor in L2-normalized embedding space."""
    # Normalize to remove magnitude bias
    cal_norm = feats_cal / (np.linalg.norm(feats_cal, axis=1, keepdims=True) + 1e-8)
    test_norm = feat_test / (np.linalg.norm(feat_test) + 1e-8)
    dists = np.linalg.norm(cal_norm - test_norm, axis=1)
    return int(labels_cal[np.argmin(dists)])


def method_linear(feats_cal, labels_cal, feat_test):
    """Fit logistic regression on calibration, predict test.
    Very strong regularization (C=0.001) since n << 512 features.
    """
    if len(np.unique(labels_cal)) < 2:
        return -1
    clf = LogisticRegression(max_iter=1000, C=0.001, solver="lbfgs")
    clf.fit(feats_cal, labels_cal)
    return int(clf.predict(feat_test.reshape(1, -1))[0])


def main():
    args = parse_args()
    device = torch.device(args.device)
    use_amp = not args.no_bf16 and device.type == "cuda"

    print(f"Extractor: {args.extractor} | Norm: {args.norm} | Device: {device}")

    # Load dataset and extract features
    dataset = StressEEGDataset(CSV_PATH, DATA_ROOT, norm=args.norm)
    extractor = create_extractor(args.extractor)
    embed_dim = extractor.embed_dim
    model = DecoupledStressModel(extractor, embed_dim=embed_dim).to(device)
    model.freeze_backbone()

    print("Extracting features...", end=" ", flush=True)
    feats = extract_all_features(model, dataset, device, use_amp)
    print(f"done ({feats.shape})")

    # Build per-subject data
    records = dataset.records
    patient_ids = np.array([r["patient_id"] for r in records])
    stress_scores = np.array([r["stress_score"] * 100 for r in records])  # back to 0-100

    results = {name: [] for name in ["centroid", "1nn", "linear"]}
    methods = {"centroid": method_centroid, "1nn": method_1nn, "linear": method_linear}

    print(f"\n{'='*70}")
    print("Within-Subject Leave-One-Out Evaluation")
    print(f"{'='*70}")

    per_subject_results = []

    for pid in sorted(set(patient_ids)):
        idx = np.where(patient_ids == pid)[0]
        if len(idx) < MIN_RECORDINGS:
            continue

        subj_feats = feats[idx]
        subj_scores = stress_scores[idx]
        median = np.median(subj_scores)

        # Binary label: above/below personal median
        # -1 for exactly at median (excluded from evaluation but used in calibration)
        subj_labels = np.where(
            subj_scores > median, 1,
            np.where(subj_scores < median, 0, -1)
        )

        # Need at least 1 above AND 1 below (excluding ties)
        n_above = (subj_labels == 1).sum()
        n_below = (subj_labels == 0).sum()
        if n_above < 1 or n_below < 1:
            continue

        subj_result = {"patient_id": int(pid), "n": len(idx), "median": float(median),
                        "n_above": int(n_above), "n_below": int(n_below),
                        "n_tied": int((subj_labels == -1).sum())}

        for method_name, method_fn in methods.items():
            preds = []
            test_indices = []  # which LOO iterations are testable
            for i in range(len(idx)):
                # Skip testing on recordings exactly at median (ambiguous)
                if subj_labels[i] == -1:
                    continue

                # Calibration: all others (including median recordings as context)
                cal_mask = np.ones(len(idx), dtype=bool)
                cal_mask[i] = False
                cal_feats = subj_feats[cal_mask]
                cal_labels_raw = subj_labels[cal_mask]

                # Only use non-tied calibration samples for label-based methods
                cal_valid = cal_labels_raw >= 0
                if cal_valid.sum() < 2 or len(np.unique(cal_labels_raw[cal_valid])) < 2:
                    preds.append(-1)
                    test_indices.append(i)
                    continue

                pred = method_fn(cal_feats[cal_valid], cal_labels_raw[cal_valid], subj_feats[i])
                preds.append(pred)
                test_indices.append(i)

            valid = [j for j, p in enumerate(preds) if p >= 0]
            if len(valid) < 2:
                continue
            y_true = np.array([subj_labels[test_indices[j]] for j in valid])
            y_pred = np.array([preds[j] for j in valid])

            results[method_name].append((y_true, y_pred))
            subj_result[f"{method_name}_acc"] = round(accuracy_score(y_true, y_pred), 4)

        per_subject_results.append(subj_result)

        scores_str = ", ".join([
            f"{s:.0f}{'+'if l==1 else '-' if l==0 else '='}"
            for s, l in zip(subj_scores, subj_labels)])
        accs = "  ".join([f"{m}={subj_result.get(f'{m}_acc', 'N/A')}"
                          for m in methods])
        print(f"  P{pid:>2d} (n={len(idx)}, med={median:.0f}): [{scores_str}]  {accs}")

    # Global metrics
    print(f"\n{'='*70}")
    print("Global Within-Subject Results")
    print(f"{'='*70}")

    summary = {"extractor": args.extractor, "norm": args.norm}

    for method_name in methods:
        if not results[method_name]:
            continue
        all_true = np.concatenate([r[0] for r in results[method_name]])
        all_pred = np.concatenate([r[1] for r in results[method_name]])

        acc = accuracy_score(all_true, all_pred)
        bal = balanced_accuracy_score(all_true, all_pred)
        kappa = cohen_kappa_score(all_true, all_pred)

        print(f"  {method_name:>10s}: acc={acc:.4f}  bal_acc={bal:.4f}  kappa={kappa:.4f}  (n={len(all_true)})")
        summary[method_name] = {
            "acc": round(acc, 4), "bal_acc": round(bal, 4),
            "kappa": round(kappa, 4), "n": len(all_true),
        }

    summary["per_subject"] = per_subject_results

    # Save
    run_id = f"{datetime.now():%Y%m%d_%H%M}_longitudinal_{args.extractor}"
    results_dir = os.path.join("results", run_id)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
