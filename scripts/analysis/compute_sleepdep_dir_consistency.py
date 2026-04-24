"""Compute SleepDep within-subject direction-consistency for Fig 4 trajectory row.

Matches EEGMAT definition (scripts/figures/build_within_subject_supplementary.py):
direction per subject = (x_sleep_deprived - x_rested) / ||.||, then mean pairwise
cosine similarity across subjects = `dir_consistency`.

Writes into paper/figures/_historical/source_tables/sleepdep_within_subject.json.
Consumed by Fig 4 build script.
"""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cosine

REPO = Path("/raid/jupyter-linjimmy1003.md10/UCSD_stress")
CACHE = REPO / "results/features_cache"
FT_BASE = REPO / "results/studies/exp_newdata"
OUT = REPO / "paper/figures/_historical/source_tables/sleepdep_within_subject.json"


def dir_consistency(X, pids, labels):
    unique_pids = np.unique(pids)
    directions = []
    for pid in unique_pids:
        m = pids == pid
        x0 = X[m & (labels == 0)]
        x1 = X[m & (labels == 1)]
        if len(x0) == 0 or len(x1) == 0:
            continue
        diff = x1[0] - x0[0]
        n = np.linalg.norm(diff)
        if n > 0:
            directions.append(diff / n)
    sims = [1 - cosine(directions[i], directions[j])
            for i, j in combinations(range(len(directions)), 2)]
    return float(np.mean(sims)), int(len(directions))


def load_frozen(fm):
    d = np.load(CACHE / f"frozen_{fm}_sleepdep_19ch.npz", allow_pickle=True)
    return d["features"], d["patient_ids"], d["labels"].astype(int)


def load_ft(fm, seed=42):
    d = FT_BASE / f"sleepdep_ft_{fm}_s{seed}"
    feats, pids, labs, idxs = [], [], [], []
    for k in range(1, 6):
        p = d / f"fold{k}_features.npz"
        if not p.exists():
            return None
        f = np.load(p, allow_pickle=True)
        feats.append(f["features"])
        pids.append(f["patient_ids"])
        labs.append(f["labels"])
        idxs.append(f["test_idx"])
    X = np.concatenate(feats); P = np.concatenate(pids)
    L = np.concatenate(labs).astype(int); I = np.concatenate(idxs)
    order = np.argsort(I)
    return X[order], P[order], L[order]


def main():
    out = {"frozen": {"sleepdep": {}}, "ft": {"sleepdep": {}}}
    for fm in ["labram", "cbramod", "reve"]:
        Xf, Pf, Lf = load_frozen(fm)
        dc_f, n_f = dir_consistency(Xf, Pf, Lf)
        out["frozen"]["sleepdep"][fm] = {"dir_consistency": dc_f, "n_subj": n_f}
        print(f"frozen {fm}  dir={dc_f:+.3f}  (n_subj={n_f})")
        ft = load_ft(fm)
        if ft is not None:
            X, P, L = ft
            dc_t, n_t = dir_consistency(X, P, L)
            out["ft"]["sleepdep"][fm] = {"dir_consistency": dc_t, "n_subj": n_t}
            print(f"    FT {fm}  dir={dc_t:+.3f}  (n_subj={n_t})")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
