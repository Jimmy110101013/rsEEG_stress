"""Compute variance decomposition + RSA for SleepDep (3 FMs).
Writes results to results/final/source_tables/sleepdep_variance_rsa.json
to be consumed by Fig 2 build script."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr

REPO = Path("/raid/jupyter-linjimmy1003.md10/UCSD_stress")
sys.path.insert(0, str(REPO))


def crossed_ss_fractions(features, subject, label):
    """Two-factor SS fractions for CROSSED designs (within-subject paired).
    Matches scripts/analysis/exp30_run_pipeline.py._crossed_ss_fractions."""
    f = np.asarray(features, dtype=np.float64)
    s, y = np.asarray(subject), np.asarray(label)
    grand = f.mean(axis=0, keepdims=True)
    ss_total = ((f - grand) ** 2).sum()
    ss_label = sum(
        (y == lab).sum() * float(((f[y == lab].mean(0) - grand.squeeze()) ** 2).sum())
        for lab in np.unique(y)
    )
    ss_subject = sum(
        (s == sid).sum() * float(((f[s == sid].mean(0) - grand.squeeze()) ** 2).sum())
        for sid in np.unique(s)
    )
    t = max(ss_total, 1e-18)
    return {
        "frac_label":    ss_label / t * 100.0,
        "frac_subject":  ss_subject / t * 100.0,
        "frac_residual": max(1.0 - ss_label/t - ss_subject/t, 0.0) * 100.0,
    }

CACHE = REPO / "results/features_cache"
FT_BASE = REPO / "results/studies/exp_newdata"
OUT = REPO / "results/final/source_tables/sleepdep_variance_rsa.json"


def compute_rsa(features, labels, patient_ids):
    feat_rdm = squareform(pdist(features, metric="cosine"))
    n = len(features)
    label_rdm = (labels[:, None] != labels[None, :]).astype(float)
    subj_rdm = (patient_ids[:, None] != patient_ids[None, :]).astype(float)
    tri = np.triu_indices(n, k=1)
    r_l, _ = spearmanr(feat_rdm[tri], label_rdm[tri])
    r_s, _ = spearmanr(feat_rdm[tri], subj_rdm[tri])
    return float(r_l), float(r_s)


def load_ft_folds(fm: str, seed: int = 42):
    d = FT_BASE / f"sleepdep_ft_{fm}_s{seed}"
    folds = []
    for k in range(1, 6):
        npz = d / f"fold{k}_features.npz"
        if not npz.exists():
            return None
        f = np.load(npz, allow_pickle=True)
        folds.append((f["features"], f["patient_ids"], f["labels"]))
    return folds


def main():
    out = {}
    for fm in ["labram", "cbramod", "reve"]:
        ch = 19
        frozen = np.load(CACHE / f"frozen_{fm}_sleepdep_{ch}ch.npz", allow_pickle=True)
        X, pids, y = frozen["features"], frozen["patient_ids"], frozen["labels"]

        rz = crossed_ss_fractions(X, pids, y)
        fr_label, fr_subj = rz["frac_label"], rz["frac_subject"]
        rsa_l, rsa_s = compute_rsa(X, y.astype(int), pids.astype(int))

        folds = load_ft_folds(fm)
        if folds is not None:
            pf = np.concatenate([t[0] for t in folds])
            ps = np.concatenate([t[1] for t in folds])
            py = np.concatenate([t[2] for t in folds])
            rf = crossed_ss_fractions(pf, ps, py)
            ft_label, ft_subj = rf["frac_label"], rf["frac_subject"]
            ft_rsa_l, ft_rsa_s = compute_rsa(pf, py.astype(int), ps.astype(int))
        else:
            ft_label = ft_subj = ft_rsa_l = ft_rsa_s = None

        key = f"{fm}_sleepdep"
        out[key] = {
            "model": fm, "dataset": "sleepdep",
            "frozen_label_frac": round(fr_label, 4),
            "frozen_subject_frac": round(fr_subj, 4),
            "ft_label_frac": round(ft_label, 4) if ft_label is not None else None,
            "ft_subject_frac": round(ft_subj, 4) if ft_subj is not None else None,
            "rsa_label_r": round(rsa_l, 4),
            "rsa_subject_r": round(rsa_s, 4),
            "ft_rsa_label_r": round(ft_rsa_l, 4) if ft_rsa_l is not None else None,
            "ft_rsa_subject_r": round(ft_rsa_s, 4) if ft_rsa_s is not None else None,
            "n_rec": int(len(X)),
            "n_subj": int(len(np.unique(pids))),
        }
        print(f"{key}  frozen: label={fr_label:.2f}%  subj={fr_subj:.2f}%  "
              f"RSA(label={rsa_l:+.3f}, subj={rsa_s:+.3f})  "
              f"FT: label={ft_label}  subj={ft_subj}")

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
