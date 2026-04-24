"""Crossed PERMANOVA (Anderson 2001) on cosine distance for variance triangulation.

Robustness check for the trace-ANOVA variance decomposition. Operates on the
window-level cosine distance matrix and partitions pseudo-SS into label,
subject (nested within label for subject-trait cells, crossed with label for
within-subject cells), and residual. Reports pseudo-F and pseudo-R^2 with a
999-permutation p-value under a nested permutation scheme (permute within
each label's subject pool).

Works uniformly on all 4 cells: subject-trait cells (ADFTD, Stress) get
"subject nested in label" SS; within-subject cells (EEGMAT, SleepDep) get
"subject crossed with label" SS. The script auto-detects design from the
label:subject joint table.

Frozen features from results/features_cache/frozen_{fm}_{cell}_perwindow.npz;
FT features from results/final_winfeat/{cell}/{fm}/seed42/fold*_features.npz
(concatenated across folds, per-window).

Output: results/studies/exp32_variance_triangulation/permanova_cosine.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

N_PERM = 999
SEED = 42


def cosine_distance_matrix(X):
    """Pairwise cosine distance (0 to 2). X is (N, d)."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sim = Xn @ Xn.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def permanova_crossed_ss(D2, y_label, y_subject):
    """Crossed-design SS partition on squared pairwise distance matrix D2.

    Uses Anderson 2001 formulation:
        SS_T  = sum_{i<j} D2_ij / N
        SS_A  = sum over label groups g_A of (sum_{i<j in g_A} D2_ij / n_gA) - SS_T
                  (technically a between/within decomposition)

    For efficiency and to avoid loop over all pairs, we use the identity
        SS_within(group g) = (1/n_g) * sum_{i<j in g} D2_ij
        SS_total = (1/N) * sum_{i<j} D2_ij

    The crossed two-way partition follows Anderson & Legendre (1999):
        SS_total = SS_label + SS_subject|label + SS_residual
      where subject|label means subject variability after accounting for label.
    We compute:
        SS_label    = SS_total - SS_within(label)
        SS_subject|label = SS_within(label) - SS_within(label x subject)
        SS_residual      = SS_within(label x subject)

    Returns dict with ss_total, ss_label, ss_subject, ss_residual,
    label_frac, subject_frac, residual_frac.
    """
    N = D2.shape[0]
    # SS_total = sum over all pairs / N
    tri = np.triu_indices(N, k=1)
    ss_total = D2[tri].sum() / N

    def ss_within_by(groups):
        ss = 0.0
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            if len(idx) < 2:
                continue
            sub = D2[np.ix_(idx, idx)]
            ti = np.triu_indices(len(idx), k=1)
            ss += sub[ti].sum() / len(idx)
        return ss

    ss_within_label = ss_within_by(y_label)
    # Composite label x subject cell
    joint = y_label.astype(np.int64) * (y_subject.max() + 1) + y_subject.astype(np.int64)
    ss_within_joint = ss_within_by(joint)

    ss_label = ss_total - ss_within_label
    ss_subject = ss_within_label - ss_within_joint
    ss_residual = ss_within_joint

    return {
        "ss_total": float(ss_total),
        "ss_label": float(ss_label),
        "ss_subject": float(ss_subject),
        "ss_residual": float(ss_residual),
        "label_frac": float(ss_label / ss_total) if ss_total > 0 else 0.0,
        "subject_frac": float(ss_subject / ss_total) if ss_total > 0 else 0.0,
        "residual_frac": float(ss_residual / ss_total) if ss_total > 0 else 0.0,
    }


def detect_design(y_label, y_subject):
    """Return 'subject_trait' if each subject has a unique label,
    else 'within_subject'."""
    for s in np.unique(y_subject):
        lbls = np.unique(y_label[y_subject == s])
        if len(lbls) > 1:
            return "within_subject"
    return "subject_trait"


def permanova_pvalue(D2, y_label, y_subject, design, n_perm=N_PERM, rng=None):
    """Permutation p-value for the label effect under the correct exchangeability.

    subject_trait cells: permute labels AMONG SUBJECTS (not windows), then
        propagate to windows. Preserves subject structure.
    within_subject cells: permute labels WITHIN SUBJECT (pairwise swap of
        state labels per subject), preserving subject identity.

    Returns p = (1 + #{SS_label_perm >= SS_label_obs}) / (1 + n_perm).
    """
    if rng is None:
        rng = np.random.RandomState(SEED)
    obs = permanova_crossed_ss(D2, y_label, y_subject)
    ss_label_obs = obs["ss_label"]

    count_ge = 0
    for _ in range(n_perm):
        if design == "subject_trait":
            # Permute label at subject level
            unique_subj = np.unique(y_subject)
            subj_label = {}
            for s in unique_subj:
                subj_label[s] = y_label[y_subject == s][0]
            shuffled_labels = np.array(list(subj_label.values()))
            rng.shuffle(shuffled_labels)
            subj_new_label = dict(zip(unique_subj, shuffled_labels))
            y_perm = np.array([subj_new_label[s] for s in y_subject])
        else:
            # Permute label within subject (window-level)
            y_perm = y_label.copy()
            for s in np.unique(y_subject):
                m = y_subject == s
                vals = y_perm[m].copy()
                rng.shuffle(vals)
                y_perm[m] = vals
        p = permanova_crossed_ss(D2, y_perm, y_subject)
        if p["ss_label"] >= ss_label_obs:
            count_ge += 1
    pval = (1 + count_ge) / (1 + n_perm)
    return obs, float(pval)


def subsample(X, y_label, y_subject, n_max, rng):
    """Subsample windows to cap distance-matrix memory, preserving proportions."""
    if len(X) <= n_max:
        return X, y_label, y_subject
    # Stratified by (label, subject) to keep design
    joint = y_label.astype(np.int64) * (y_subject.max() + 1) + y_subject.astype(np.int64)
    frac = n_max / len(X)
    keep = []
    for g in np.unique(joint):
        idx = np.where(joint == g)[0]
        n_take = max(2, int(round(len(idx) * frac)))
        n_take = min(n_take, len(idx))
        keep.append(rng.choice(idx, size=n_take, replace=False))
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return X[keep], y_label[keep], y_subject[keep]


def load_frozen(dataset, model):
    p = REPO / f"results/features_cache/frozen_{model}_{dataset}_perwindow.npz"
    d = np.load(p)
    X = d["features"]
    rec_idx = d["window_rec_idx"].astype(np.int64)
    rec_labels = d["rec_labels"].astype(np.int64)
    rec_pids = d["rec_pids"].astype(np.int64)
    return X, rec_labels[rec_idx], rec_pids[rec_idx]


def load_ft(dataset, model, seed=42):
    """Concat all folds of FT per-window features."""
    fold_dir = REPO / f"results/final_winfeat/{dataset}/{model}/seed{seed}"
    fold_files = sorted(fold_dir.glob("fold*_features.npz"))
    if not fold_files:
        return None
    # Load rec meta from frozen cache (consistent subject/label mapping)
    _, rec_labels, rec_pids = None, None, None
    p_frozen = REPO / f"results/features_cache/frozen_{model}_{dataset}_perwindow.npz"
    d_frozen = np.load(p_frozen)
    rec_labels = d_frozen["rec_labels"].astype(np.int64)
    rec_pids = d_frozen["rec_pids"].astype(np.int64)

    X_all, y_lbl, y_sub = [], [], []
    for f in fold_files:
        d = np.load(f)
        Xw = d["window_features"]
        wri = d["window_rec_idx"].astype(np.int64)  # local to test_idx
        test_idx = d["test_idx"].astype(np.int64)   # global rec indices
        global_rec = test_idx[wri]
        X_all.append(Xw)
        y_lbl.append(rec_labels[global_rec])
        y_sub.append(rec_pids[global_rec])
    return np.concatenate(X_all), np.concatenate(y_lbl), np.concatenate(y_sub)


def apply_stress_filter(X, y_label, y_subject):
    mask = ~np.isin(y_subject, [3, 11, 17])
    return X[mask], y_label[mask], y_subject[mask]


def run_one(X, y_label, y_subject, n_max=4000, n_perm=N_PERM, seed=SEED):
    rng = np.random.RandomState(seed)
    X_s, yl_s, ys_s = subsample(X, y_label, y_subject, n_max, rng)
    design = detect_design(yl_s, ys_s)
    t0 = time.time()
    D = cosine_distance_matrix(X_s)
    D2 = D ** 2  # PERMANOVA uses squared distance
    obs, pval = permanova_pvalue(D2, yl_s, ys_s, design, n_perm=n_perm, rng=rng)
    elapsed = time.time() - t0
    obs["n_windows_used"] = int(len(X_s))
    obs["n_windows_total"] = int(len(X))
    obs["design"] = design
    obs["p_label"] = pval
    obs["elapsed_s"] = round(elapsed, 1)
    return obs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True,
                   choices=["stress", "eegmat", "sleepdep", "adftd"])
    p.add_argument("--models", nargs="+", default=["labram", "cbramod", "reve"])
    p.add_argument("--n-max", type=int, default=4000,
                   help="Cap windows for distance matrix to stay within RAM")
    p.add_argument("--n-perm", type=int, default=N_PERM)
    p.add_argument("--out-dir", default="results/studies/exp32_variance_triangulation")
    args = p.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_permanova.json"

    out = {
        "dataset": args.dataset,
        "protocol": "crossed_PERMANOVA_cosine",
        "n_perm": args.n_perm,
        "n_max_subsample": args.n_max,
        "seed": SEED,
        "results": {},
    }
    for model in args.models:
        out["results"][model] = {}
        for source in ("frozen", "ft"):
            print(f"[{args.dataset} | {model} | {source}]", flush=True)
            if source == "frozen":
                X, yl, ys = load_frozen(args.dataset, model)
            else:
                got = load_ft(args.dataset, model)
                if got is None:
                    print("  no FT features, skip")
                    continue
                X, yl, ys = got
            if args.dataset == "stress":
                X, yl, ys = apply_stress_filter(X, yl, ys)
            print(f"  n_windows={len(X)} d={X.shape[1]} n_subj={len(np.unique(ys))}",
                  flush=True)
            r = run_one(X, yl, ys, n_max=args.n_max, n_perm=args.n_perm)
            print(f"  label_frac={r['label_frac']:.4f} "
                  f"subject_frac={r['subject_frac']:.4f} "
                  f"residual_frac={r['residual_frac']:.4f} "
                  f"design={r['design']} p={r['p_label']:.4f} "
                  f"t={r['elapsed_s']}s", flush=True)
            out["results"][model][source] = r

        # Delta (FT - frozen) in percentage points
        if "frozen" in out["results"][model] and "ft" in out["results"][model]:
            frz = out["results"][model]["frozen"]
            ft = out["results"][model]["ft"]
            out["results"][model]["delta"] = {
                "delta_label_frac_pp": (ft["label_frac"] - frz["label_frac"]) * 100,
                "delta_subject_frac_pp": (ft["subject_frac"] - frz["subject_frac"]) * 100,
            }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
