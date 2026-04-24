"""Linear CKA of FM features against label / subject indicator Grams.

Third leg of the variance triangulation. Centered Kernel Alignment
(Kornblith et al. 2019) between the feature Gram K_Z and one-hot Grams for
label (K_y) and subject (K_s). Reports:

  CKA(K_Z, K_y)  -- how aligned is the representation to task label
  CKA(K_Z, K_s)  -- how aligned is the representation to subject identity

Linear-kernel CKA is scale- and rotation-invariant, insensitive to high
dimensionality, and a standard representation-learning metric. Unlike
variance partition (ANOVA, PERMANOVA) it gives one bounded number in [0, 1]
per factor but does NOT partition -- CKA(label) + CKA(subject) need not
equal anything meaningful.

Output: results/studies/exp32_variance_triangulation/{cell}_cka.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def center_gram(K):
    """Center a Gram matrix (Gretton 2005): H K H where H = I - (1/n) 11^T."""
    n = K.shape[0]
    unit = np.ones((n, n)) / n
    Kc = K - unit @ K - K @ unit + unit @ K @ unit
    return Kc


def linear_cka(X, Y):
    """Linear CKA between feature-ish matrices X (n,dx) and Y (n,dy).

    Using the inner-product formula:
       CKA = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    after centering rows. This avoids forming (n,n) Gram matrices when
    d << n (which is our case -- d=200/512 vs n up to ~5k).
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    # ||Xc^T Yc||_F^2 = tr((Xc^T Yc)(Yc^T Xc)) = tr(Yc^T Xc Xc^T Yc)
    num = np.linalg.norm(Xc.T @ Yc, ord="fro") ** 2
    denom = np.linalg.norm(Xc.T @ Xc, ord="fro") * np.linalg.norm(Yc.T @ Yc, ord="fro")
    if denom <= 0:
        return 0.0
    return float(num / denom)


def one_hot(y):
    vals = np.unique(y)
    H = np.zeros((len(y), len(vals)), dtype=np.float32)
    for i, v in enumerate(vals):
        H[y == v, i] = 1.0
    return H


def load_frozen(dataset, model):
    p = REPO / f"results/features_cache/frozen_{model}_{dataset}_perwindow.npz"
    d = np.load(p)
    X = d["features"].astype(np.float32)
    rec_idx = d["window_rec_idx"].astype(np.int64)
    rec_labels = d["rec_labels"].astype(np.int64)
    rec_pids = d["rec_pids"].astype(np.int64)
    return X, rec_labels[rec_idx], rec_pids[rec_idx]


def load_ft(dataset, model, seed=42):
    fold_dir = REPO / f"results/final_winfeat/{dataset}/{model}/seed{seed}"
    fold_files = sorted(fold_dir.glob("fold*_features.npz"))
    if not fold_files:
        return None
    p_frozen = REPO / f"results/features_cache/frozen_{model}_{dataset}_perwindow.npz"
    d_frozen = np.load(p_frozen)
    rec_labels = d_frozen["rec_labels"].astype(np.int64)
    rec_pids = d_frozen["rec_pids"].astype(np.int64)
    X_all, y_lbl, y_sub = [], [], []
    for f in fold_files:
        d = np.load(f)
        Xw = d["window_features"].astype(np.float32)
        wri = d["window_rec_idx"].astype(np.int64)
        test_idx = d["test_idx"].astype(np.int64)
        global_rec = test_idx[wri]
        X_all.append(Xw)
        y_lbl.append(rec_labels[global_rec])
        y_sub.append(rec_pids[global_rec])
    return np.concatenate(X_all), np.concatenate(y_lbl), np.concatenate(y_sub)


def apply_stress_filter(X, y_label, y_subject):
    mask = ~np.isin(y_subject, [3, 11, 17])
    return X[mask], y_label[mask], y_subject[mask]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True,
                   choices=["stress", "eegmat", "sleepdep", "adftd"])
    p.add_argument("--models", nargs="+", default=["labram", "cbramod", "reve"])
    p.add_argument("--out-dir", default="results/studies/exp32_variance_triangulation")
    args = p.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_cka.json"

    out = {
        "dataset": args.dataset,
        "protocol": "linear_CKA_label_subject",
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

            H_lbl = one_hot(yl)
            H_sub = one_hot(ys)
            cka_lbl = linear_cka(X, H_lbl)
            cka_sub = linear_cka(X, H_sub)
            print(f"  CKA(label)={cka_lbl:.4f}  CKA(subject)={cka_sub:.4f}",
                  flush=True)
            out["results"][model][source] = {
                "cka_label": cka_lbl,
                "cka_subject": cka_sub,
                "n_windows": int(len(X)),
                "feature_dim": int(X.shape[1]),
                "n_labels": int(H_lbl.shape[1]),
                "n_subjects": int(H_sub.shape[1]),
            }

        # Delta
        if "frozen" in out["results"][model] and "ft" in out["results"][model]:
            frz = out["results"][model]["frozen"]
            ft = out["results"][model]["ft"]
            out["results"][model]["delta"] = {
                "delta_cka_label": ft["cka_label"] - frz["cka_label"],
                "delta_cka_subject": ft["cka_subject"] - frz["cka_subject"],
            }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
