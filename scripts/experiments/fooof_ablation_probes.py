"""Run subject-ID and state-label probes on FOOOF-ablated FM features.

Loads per-window features for four conditions (original, aperiodic_removed,
periodic_removed, both_removed) and reports:
  - subject-ID linear probe BA (multi-class, session-level held-out)
  - state-label linear probe BA (binary, subject-level held-out, only
    meaningful on EEGMAT — DASS on Stress is already known unlearnable)

Output: JSON with per-condition × per-FM × per-probe numbers.

Usage:
    PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
    $PY scripts/experiments/fooof_ablation_probes.py --dataset stress
    $PY scripts/experiments/fooof_ablation_probes.py --dataset eegmat
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


SEEDS = [42, 123, 2024, 7, 0, 1, 99, 31337]


def clip_and_scale(X_tr, X_te):
    lo = np.percentile(X_tr, 1, axis=0)
    hi = np.percentile(X_tr, 99, axis=0)
    X_tr_c = np.clip(X_tr, lo, hi)
    X_te_c = np.clip(X_te, lo, hi)
    sc = StandardScaler().fit(X_tr_c)
    return sc.transform(X_tr_c), sc.transform(X_te_c)


def state_probe(window_feats, window_rec_idx, rec_labels, rec_pids, seed, n_splits=5):
    """Binary state-label probe, subject-level CV, prediction pooling."""
    n_rec = len(rec_labels)
    rec_indices = np.arange(n_rec)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rec_pred = np.zeros(n_rec, dtype=int)
    for train_rec, test_rec in cv.split(rec_indices, rec_labels, groups=rec_pids):
        train_mask = np.isin(window_rec_idx, train_rec)
        test_mask = np.isin(window_rec_idx, test_rec)
        X_tr = window_feats[train_mask]
        y_tr = rec_labels[window_rec_idx[train_mask]]
        X_te = window_feats[test_mask]
        te_rec = window_rec_idx[test_mask]
        X_tr_s, X_te_s = clip_and_scale(X_tr, X_te)
        clf = LogisticRegression(
            max_iter=5000, class_weight="balanced", C=1.0,
            solver="liblinear", tol=1e-3,
        )
        clf.fit(X_tr_s, y_tr)
        prob_te = clf.predict_proba(X_te_s)[:, 1]
        for r in test_rec:
            m = te_rec == r
            if m.any():
                rec_pred[r] = int(prob_te[m].mean() >= 0.5)
    return float(balanced_accuracy_score(rec_labels, rec_pred))


def subject_probe(window_feats, window_rec_idx, rec_labels, rec_pids, seed):
    """Multi-class subject-ID probe, SESSION-level held-out within subject.

    Each subject's recordings are split into train/test halves (if ≥ 2 recs).
    Subjects with only 1 recording are used for training only. Window-level
    training with prediction pooling to recording level.
    """
    n_rec = len(rec_labels)
    unique_pids = np.unique(rec_pids)
    # Require at least 2 recordings per subject for split
    rng = np.random.RandomState(seed)

    train_rec_mask = np.zeros(n_rec, dtype=bool)
    test_rec_mask = np.zeros(n_rec, dtype=bool)
    eligible_subjects = []

    for pid in unique_pids:
        rec_of_pid = np.where(rec_pids == pid)[0]
        if len(rec_of_pid) < 2:
            train_rec_mask[rec_of_pid] = True
            continue
        shuffled = rng.permutation(rec_of_pid)
        half = len(shuffled) // 2
        train_rec_mask[shuffled[:half]] = True
        test_rec_mask[shuffled[half:]] = True
        eligible_subjects.append(pid)

    if len(eligible_subjects) < 2:
        return float("nan")

    # Window-level train/test
    train_mask = np.isin(window_rec_idx, np.where(train_rec_mask)[0])
    test_mask = np.isin(window_rec_idx, np.where(test_rec_mask)[0])

    X_tr = window_feats[train_mask]
    y_tr = rec_pids[window_rec_idx[train_mask]]
    X_te = window_feats[test_mask]
    te_rec = window_rec_idx[test_mask]
    y_te_true = rec_pids

    X_tr_s, X_te_s = clip_and_scale(X_tr, X_te)
    # Multi-class via OneVsRest + liblinear: liblinear doesn't support
    # multinomial directly, but OvR gives per-subject binary probes which
    # aggregate into a multi-class probe.
    clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=5000, class_weight="balanced", C=1.0,
            solver="liblinear", tol=1e-3,
        ),
        n_jobs=-1,
    )
    clf.fit(X_tr_s, y_tr)
    proba = clf.predict_proba(X_te_s)  # (N_te_windows, K)
    classes = clf.classes_

    # Pool probs per test recording, then argmax
    rec_pred_pid = {}
    for r in np.unique(te_rec):
        m = te_rec == r
        mean_prob = proba[m].mean(axis=0)
        rec_pred_pid[int(r)] = int(classes[np.argmax(mean_prob)])

    y_true = np.array([y_te_true[r] for r in rec_pred_pid.keys()])
    y_pred = np.array(list(rec_pred_pid.values()))
    return float(balanced_accuracy_score(y_true, y_pred))


def load_features(dataset: str, model: str, condition: str):
    """Load per-window features + meta for given condition.

    condition in {'original', 'aperiodic_removed', 'periodic_removed', 'both_removed'}.
    """
    if condition == "original":
        p = f"results/features_cache/frozen_{model}_{dataset}_perwindow.npz"
    else:
        p = f"results/features_cache/fooof_ablation/feat_{model}_{dataset}_{condition}.npz"
    d = np.load(p)
    return {
        "features": d["features"],
        "window_rec_idx": d["window_rec_idx"],
        "rec_labels": d["rec_labels"],
        "rec_pids": d["rec_pids"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["stress", "eegmat", "sleepdep", "adftd"])
    p.add_argument("--models", nargs="+", default=["labram", "cbramod", "reve"])
    p.add_argument("--out-path", default=None)
    args = p.parse_args()

    out_path = Path(args.out_path or
                    f"results/studies/fooof_ablation/{args.dataset}_probes.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conditions = ["original", "aperiodic_removed", "periodic_removed", "both_removed"]
    # State probe runs on all 3 datasets. Stress DASS is trait-noisy — expected to be
    # near chance by design, serves as explicit null control for the three-regime taxonomy.
    do_state = args.dataset in ("eegmat", "sleepdep", "stress", "adftd")

    results = {"dataset": args.dataset, "results": {}}
    for model in args.models:
        results["results"][model] = {}
        for cond in conditions:
            print(f"[{model} | {cond}]")
            try:
                d = load_features(args.dataset, model, cond)
            except FileNotFoundError as e:
                print(f"  MISSING: {e}")
                continue
            X = d["features"]
            rec_idx = d["window_rec_idx"]
            rec_lbl = d["rec_labels"]
            rec_pid = d["rec_pids"]

            subj_bas = []
            state_bas = []
            for s in SEEDS:
                subj_ba = subject_probe(X, rec_idx, rec_lbl, rec_pid, s)
                subj_bas.append(subj_ba)
                if do_state:
                    st_ba = state_probe(X, rec_idx, rec_lbl, rec_pid, s)
                    state_bas.append(st_ba)

            subj_arr = np.array(subj_bas)
            entry = {
                "subject_probe_mean": float(subj_arr.mean()),
                "subject_probe_std": float(subj_arr.std(ddof=1)),
                "subject_probe_bas": subj_bas,
            }
            if do_state:
                st_arr = np.array(state_bas)
                entry.update({
                    "state_probe_mean": float(st_arr.mean()),
                    "state_probe_std": float(st_arr.std(ddof=1)),
                    "state_probe_bas": state_bas,
                })
            print(f"  subject_probe: {entry['subject_probe_mean']:.4f} "
                  f"± {entry['subject_probe_std']:.4f}")
            if do_state:
                print(f"  state_probe:   {entry['state_probe_mean']:.4f} "
                      f"± {entry['state_probe_std']:.4f}")
            results["results"][model][cond] = entry

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
