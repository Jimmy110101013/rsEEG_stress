"""SleepDep within-subject pairwise discrimination on frozen LP features.

Each subject has exactly 2 recordings (NS, SD). The within-subject test is:
  for each held-out subject S:
    train LR on all OTHER subjects' (recording-avg) features → labels
    predict P(SD) on S's NS and S's SD recordings
    correctly_ranked = P(SD | S's SD recording) > P(SD | S's NS recording)
  pairwise_BA = mean(correctly_ranked) across subjects

Comparable to exp11_longitudinal_dss EEGMAT LOO BA (same logic, 2 recs/subj).

Output: results/studies/exp11_longitudinal_dss/sleepdep_within_subject.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

REPO = Path("/raid/jupyter-linjimmy1003.md10/UCSD_stress")
CACHE = REPO / "results/features_cache"
OUT = REPO / "results/studies/exp11_longitudinal_dss/sleepdep_within_subject.json"


def load(fm: str):
    p = CACHE / f"frozen_{fm}_sleepdep_19ch.npz"
    d = np.load(p, allow_pickle=True)
    return d["features"], d["labels"], d["patient_ids"]


def pairwise_within_subject(X, y, p, seed=42):
    """For each subject S, hold out their pair, train on rest, check if SD prob > NS."""
    correct = []
    for s in np.unique(p):
        s_mask = (p == s)
        # require exactly 2 distinct labels (NS+SD pair)
        if len(np.unique(y[s_mask])) != 2 or s_mask.sum() != 2:
            continue
        X_te = X[s_mask]
        y_te = y[s_mask]
        X_tr = X[~s_mask]
        y_tr = y[~s_mask]
        clf = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced",
                                      C=1.0, solver="liblinear",
                                      random_state=seed)),
        ])
        clf.fit(X_tr, y_tr)
        prob_sd = clf.predict_proba(X_te)[:, 1]
        # SD recording (y_te==1) should have higher prob than NS (y_te==0)
        sd_idx = np.where(y_te == 1)[0][0]
        ns_idx = np.where(y_te == 0)[0][0]
        correct.append(int(prob_sd[sd_idx] > prob_sd[ns_idx]))
    return np.array(correct)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = {"protocol": "Within-subject pairwise discrimination, LOO over subjects, "
                       "SD prob > NS prob test on the held-out subject's pair.",
           "results": {}}
    for fm in ["labram", "cbramod", "reve"]:
        try:
            X, y, p = load(fm)
            corrects = []
            for seed in [42, 123, 2024]:
                c = pairwise_within_subject(X, y, p, seed=seed)
                corrects.append(c.mean())
            arr = np.array(corrects)
            out["results"][fm] = {
                "pairwise_ba_per_seed": [round(x, 4) for x in corrects],
                "pairwise_ba_mean": round(float(arr.mean()), 4),
                "pairwise_ba_std": round(float(arr.std(ddof=1)), 4),
                "n_subjects_used": int(len([s for s in np.unique(p)
                                            if (p == s).sum() == 2
                                            and len(np.unique(y[p == s])) == 2])),
            }
            print(f"  {fm}: pairwise BA = {arr.mean():.4f} ± {arr.std(ddof=1):.4f} "
                  f"(3 seeds, n_used={out['results'][fm]['n_subjects_used']})")
        except Exception as e:
            print(f"  FAIL {fm}: {e}")
            out["results"][fm] = {"error": str(e)}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
