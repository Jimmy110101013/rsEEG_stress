#!/usr/bin/env python
"""Window-level variance decomposition for all 4 cells × 3 FMs.

Drop-in successor to `run_variance_analysis.py` that operates on per-window
frozen features instead of pooled per-recording features. Enables variance
decomposition on single-session trait cohorts (ADFTD split1) where the
recording-level decomposition is degenerate (n_rec = n_subj, 0 within-subject
df).

Uses a **crossed** SS decomposition (same formula as
`compute_sleepdep_variance_rsa.py::crossed_ss_fractions`) applied uniformly
across 4 cells: `SS_total ≈ SS_label + SS_subject + SS_residual_clipped`.

For within-subject-label cells (EEGMAT, SleepDep) the label and subject
factors are orthogonal by design, so frac_label + frac_subject < 1.
For subject-label-trait cells (ADFTD, Stress pure-label) subject variance
contains label variance by construction, so frac_subject + frac_label may
exceed 1 and residual is clipped at 0. This is documented in §3.6.1.

Reads `results/features_cache/frozen_{model}_{cell}_perwindow.npz` — expects
fields: features (N_windows, D), window_rec_idx (N_windows,),
rec_labels (n_rec,), rec_pids (n_rec,).

Writes `results/final/source_tables/variance_analysis_window_level.json`
— one entry per (model, cell) pair.

Usage:
    /raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python \\
        scripts/analysis/run_variance_window_level.py
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

CELLS = ["eegmat", "sleepdep", "stress", "adftd"]
MODELS = ["labram", "cbramod", "reve"]

# Stress (DASS per-rec): 3 subjects with recordings in both classes are
# dropped so within-subject label contrast is unambiguous (none) for the
# 14 pure-label subjects.
DROP_MIXED_SUBJECTS = {"stress"}


def crossed_ss_fractions(features: np.ndarray, subject: np.ndarray, label: np.ndarray) -> dict:
    """Two-factor SS fractions (crossed design).

    SS_label    = sum_l n_l * ||mean_l - grand||^2
    SS_subject  = sum_s n_s * ||mean_s - grand||^2
    SS_total    = sum_i ||x_i - grand||^2
    frac_residual = max(1 - frac_label - frac_subject, 0)

    Norm-squared is summed over feature dims, i.e. multivariate SS.
    """
    f = np.asarray(features, dtype=np.float64)
    s = np.asarray(subject)
    y = np.asarray(label)
    grand = f.mean(axis=0, keepdims=True)
    diff = f - grand
    ss_total = float((diff * diff).sum())

    ss_label = 0.0
    for lab in np.unique(y):
        m = y == lab
        if m.sum() == 0:
            continue
        d = f[m].mean(0) - grand.squeeze()
        ss_label += float(m.sum()) * float((d * d).sum())

    ss_subject = 0.0
    for sid in np.unique(s):
        m = s == sid
        if m.sum() == 0:
            continue
        d = f[m].mean(0) - grand.squeeze()
        ss_subject += float(m.sum()) * float((d * d).sum())

    t = max(ss_total, 1e-18)
    frac_label = ss_label / t
    frac_subject = ss_subject / t
    frac_residual = max(1.0 - frac_label - frac_subject, 0.0)
    return {
        "SS_total": ss_total,
        "SS_label": ss_label,
        "SS_subject": ss_subject,
        "label_frac": frac_label,
        "subject_frac": frac_subject,
        "residual_frac": frac_residual,
    }


def find_mixed_subjects(pids: np.ndarray, labels: np.ndarray) -> set:
    """Subjects whose recordings span >1 label."""
    mixed = set()
    for p in np.unique(pids):
        if len(np.unique(labels[pids == p])) > 1:
            mixed.add(int(p))
    return mixed


def _load_frozen(cell: str, model: str):
    npz_path = f"results/features_cache/frozen_{model}_{cell}_perwindow.npz"
    if not os.path.isfile(npz_path):
        return None
    d = np.load(npz_path)
    return {
        "features": d["features"],
        "w_rec": d["window_rec_idx"].astype(np.int64),
        "rec_lbl": d["rec_labels"],
        "rec_pid": d["rec_pids"],
    }


def _load_ft_winfeat(cell: str, model: str, seed: int = 42):
    """Concatenate per-window FT features across all folds.

    Expects `results/final_winfeat/{cell}/{model}/seed{N}/fold*_features.npz`
    where each npz contains `window_features`, `window_rec_idx`, `labels`,
    `patient_ids` (the latter two at recording level, one per test recording
    in that fold).

    Since each fold's test set is disjoint (from the StratifiedGroupKFold CV),
    we can concatenate all fold features into a single (N_all_windows, D)
    matrix and build synthetic recording indices by offsetting each fold's
    recording indices by the cumulative count.
    """
    root = f"results/final_winfeat/{cell}/{model}/seed{seed}"
    fold_paths = sorted([
        os.path.join(root, f)
        for f in os.listdir(root)
        if f.startswith("fold") and f.endswith("_features.npz")
    ]) if os.path.isdir(root) else []
    if not fold_paths:
        return None

    all_win_feats = []
    all_w_rec = []
    all_rec_lbl = []
    all_rec_pid = []
    rec_offset = 0
    for p in fold_paths:
        d = np.load(p)
        if "window_features" not in d.files:
            return None  # old-format FT features without per-window data
        wf = d["window_features"]
        w_rec = d["window_rec_idx"].astype(np.int64) + rec_offset
        rec_lbl_f = d["labels"]
        rec_pid_f = d["patient_ids"]
        all_win_feats.append(wf)
        all_w_rec.append(w_rec)
        all_rec_lbl.append(rec_lbl_f)
        all_rec_pid.append(rec_pid_f)
        rec_offset += len(rec_lbl_f)

    return {
        "features": np.concatenate(all_win_feats, axis=0),
        "w_rec": np.concatenate(all_w_rec),
        "rec_lbl": np.concatenate(all_rec_lbl),
        "rec_pid": np.concatenate(all_rec_pid),
    }


def _compute_one_side(data: dict, cell: str) -> dict:
    """Run crossed-SS decomposition on one (frozen OR FT) feature matrix.

    Applies Stress mixed-label filter if applicable.
    """
    feats = data["features"]
    w_rec = data["w_rec"]
    rec_lbl = data["rec_lbl"]
    rec_pid = data["rec_pid"]

    dropped = []
    if cell in DROP_MIXED_SUBJECTS:
        mixed = find_mixed_subjects(rec_pid, rec_lbl)
        if mixed:
            keep_rec_mask = np.array([p not in mixed for p in rec_pid])
            kept_rec_idx = np.where(keep_rec_mask)[0]
            keep_win_mask = np.isin(w_rec, kept_rec_idx)
            feats = feats[keep_win_mask]
            w_rec = w_rec[keep_win_mask]
            old_to_new = {int(o): int(n) for n, o in enumerate(kept_rec_idx)}
            w_rec = np.array([old_to_new[int(o)] for o in w_rec])
            rec_lbl = rec_lbl[keep_rec_mask]
            rec_pid = rec_pid[keep_rec_mask]
            dropped = sorted(int(x) for x in mixed)

    # Determine label design AFTER mixed-subject filter
    per_subject_label_counts = {}
    for p, l in zip(rec_pid, rec_lbl):
        per_subject_label_counts.setdefault(int(p), set()).add(int(l))
    multi_label_pids = [p for p, ls in per_subject_label_counts.items() if len(ls) > 1]
    label_design = "within_subject" if len(multi_label_pids) > 0 else "between_subject"

    win_lab = rec_lbl[w_rec]
    win_subj = rec_pid[w_rec]
    ss = crossed_ss_fractions(feats, win_subj, win_lab)
    ratio = ss["label_frac"] / ss["subject_frac"] if ss["subject_frac"] > 0 else float("nan")
    return {
        "label_design": label_design,
        "n_windows": int(feats.shape[0]),
        "n_recordings": int(len(rec_lbl)),
        "n_subjects": int(len(np.unique(rec_pid))),
        "n_feat_dims": int(feats.shape[1]),
        "dropped_mixed_subjects": dropped,
        "label_frac": ss["label_frac"],
        "subject_frac": ss["subject_frac"],
        "residual_frac": ss["residual_frac"],
        "label_subject_ratio": ratio,
        "SS_total": ss["SS_total"],
        "SS_label": ss["SS_label"],
        "SS_subject": ss["SS_subject"],
    }


def decompose_one(cell: str, model: str, ft_seed: int = 42) -> dict:
    t0 = time.time()
    frozen_data = _load_frozen(cell, model)
    if frozen_data is None:
        return {"cell": cell, "model": model, "error": "missing frozen npz"}

    frozen_side = _compute_one_side(frozen_data, cell)

    ft_side = None
    ft_data = _load_ft_winfeat(cell, model, seed=ft_seed)
    if ft_data is not None:
        ft_side = _compute_one_side(ft_data, cell)

    entry = {
        "cell": cell,
        "model": model,
        "unit": "window",
        "frozen": frozen_side,
        "ft_seed": ft_seed,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    if ft_side is not None:
        entry["ft"] = ft_side
        entry["delta_label_frac"] = ft_side["label_frac"] - frozen_side["label_frac"]
        entry["delta_subject_frac"] = ft_side["subject_frac"] - frozen_side["subject_frac"]
    return entry


# Legacy-compat: keep the single-side worker callable for anyone importing it.
def _legacy_decompose_one(cell: str, model: str) -> dict:
    frozen = _load_frozen(cell, model)
    if frozen is None:
        return {"cell": cell, "model": model, "error": "missing frozen npz"}
    side = _compute_one_side(frozen, cell)
    return {"cell": cell, "model": model, "unit": "window", **side}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=CELLS, choices=CELLS)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--ft-seed", type=int, default=42,
                    help="Which seed's FT run under results/final_winfeat/ to load "
                         "(if available). Frozen side does not depend on this.")
    ap.add_argument(
        "--out",
        default="results/final/source_tables/variance_analysis_window_level.json",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    for cell in args.cells:
        for model in args.models:
            key = f"{model}_{cell}"
            print(f"\n=== {key} ===")
            entry = decompose_one(cell, model, ft_seed=args.ft_seed)
            if "error" in entry:
                print(f"  ERROR: {entry['error']}")
            else:
                fz = entry["frozen"]
                print(f"  [frozen] design={fz['label_design']} "
                      f"n_win={fz['n_windows']} n_rec={fz['n_recordings']} "
                      f"n_subj={fz['n_subjects']} dims={fz['n_feat_dims']}")
                print(f"  [frozen] label={fz['label_frac']*100:.3f}% "
                      f"subject={fz['subject_frac']*100:.3f}% "
                      f"residual={fz['residual_frac']*100:.3f}% "
                      f"ratio={fz['label_subject_ratio']:.4f}")
                if fz["dropped_mixed_subjects"]:
                    print(f"  dropped: {fz['dropped_mixed_subjects']}")
                if "ft" in entry:
                    ft = entry["ft"]
                    print(f"  [FT    ] n_win={ft['n_windows']} "
                          f"label={ft['label_frac']*100:.3f}% "
                          f"subject={ft['subject_frac']*100:.3f}% "
                          f"residual={ft['residual_frac']*100:.3f}% "
                          f"ratio={ft['label_subject_ratio']:.4f}")
                    print(f"  Δlabel_frac={entry['delta_label_frac']*100:+.3f} pp  "
                          f"Δsubject_frac={entry['delta_subject_frac']*100:+.3f} pp")
                else:
                    print(f"  [FT    ] not found (results/final_winfeat/{cell}/{model}/seed{args.ft_seed}/)")
                print(f"  elapsed={entry['elapsed_sec']}s")
            results[key] = entry

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
