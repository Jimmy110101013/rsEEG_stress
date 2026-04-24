"""LP vs FT representation drift analysis on Stress.

For each FM, compare frozen LP features (recording-avg cache) vs FT features
(concatenated across 5 folds, out-of-fold predictions).

Metrics:
  - Variance decomposition (label_frac, subject_frac, residual_frac via nested SS)
  - LogME (Bayesian transferability for classification)
  - H-score (transferability)
  - Linear CKA between LP and FT representations
  - kNN BA on representations (k=5 LOO)

Mechanistic interpretation for §4.6 CBraMod/REVE rescue question:
  - rescue=real label signal     → FT label_frac↑, subject_frac↓
  - rescue=subject shortcut      → FT subject_frac↑, label_frac↓ or ≈
  - rescue=generic noise reduce  → FT residual_frac↓, both ≈

Output: results/studies/representation_drift/lp_vs_ft_stress.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/raid/jupyter-linjimmy1003.md10/UCSD_stress")
sys.path.insert(0, str(REPO))

from src.variance_analysis import nested_ss  # noqa: E402
from scripts.analysis.analyze_eegmat import crossed_decomposition  # noqa: E402
from scripts.analysis.compute_fm_task_fitness import (  # noqa: E402
    compute_logme, compute_hscore, compute_cka, compute_knn,
)

CACHE = REPO / "results/features_cache"
OUT = REPO / "results/studies/representation_drift/lp_vs_ft_stress.json"


def variance_fracs(features, labels, patient_ids, design="nested"):
    """Return (label_frac_pct, subject_frac_pct, residual_frac_pct).
    design='nested' for between-subject (Stress); 'crossed' for within-subject
    (EEGMAT/SleepDep) where every subject has both labels."""
    y = np.asarray(labels); p = np.asarray(patient_ids)
    if design == "crossed":
        cd = crossed_decomposition(features, p, y)
        pf = cd["pooled_fractions"]
        return 100.0 * pf["label"], 100.0 * pf["subject"], 100.0 * pf["residual"]
    ss = nested_ss(features, p, y)
    total = float(ss["label"].sum() + ss["subject_within_label"].sum()
                  + ss["residual"].sum())
    if total == 0:
        return 0.0, 0.0, 0.0
    return (
        100.0 * float(ss["label"].sum()) / total,
        100.0 * float(ss["subject_within_label"].sum()) / total,
        100.0 * float(ss["residual"].sum()) / total,
    )


def stress_mixed_filter(y, p):
    """Drop subjects with recordings in both DASS classes — same convention as
    fig_variance_atlas (nested SS requires pure-label subjects)."""
    y = np.asarray(y); p = np.asarray(p)
    mixed = {pid for pid in np.unique(p) if len(np.unique(y[p == pid])) > 1}
    keep = np.array([pid not in mixed for pid in p])
    return keep


def load_lp(fm: str, dataset: str):
    """Frozen recording-avg cache."""
    ch = 30 if dataset == "stress" else 19
    p = CACHE / f"frozen_{fm}_{dataset}_{ch}ch.npz"
    d = np.load(p, allow_pickle=True)
    f, y, pp = d["features"], d["labels"], d["patient_ids"]
    if dataset == "stress":
        keep = stress_mixed_filter(y, pp)
        f, y, pp = f[keep], y[keep], pp[keep]
    return f, y, pp


def load_ft(fm: str, dataset: str):
    """Concatenate test-fold features across 5 folds (out-of-fold pooled).
    Prefers `_canonical` suffix when present (matches per-FM canonical recipe)."""
    candidates = [f"ft_{fm}_{dataset}_canonical", f"ft_{fm}_{dataset}"]
    base = None
    for c in candidates:
        if (CACHE / c / "fold1_features.npz").exists():
            base = CACHE / c
            break
    if base is None:
        raise FileNotFoundError(f"no fold features under any of {candidates}")
    feats, labels, pids = [], [], []
    for fold in range(1, 6):
        p = base / f"fold{fold}_features.npz"
        if not p.exists():
            raise FileNotFoundError(p)
        d = np.load(p)
        feats.append(d["features"])
        labels.append(d["labels"])
        pids.append(d["patient_ids"])
    print(f"  (FT source: {base.name})")
    f = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0)
    pp = np.concatenate(pids, axis=0)
    if dataset == "stress":
        keep = stress_mixed_filter(y, pp)
        f, y, pp = f[keep], y[keep], pp[keep]
    return f, y, pp


def analyze_one(fm: str, dataset: str):
    print(f"\n=== {fm} × {dataset} ===")
    f_lp, y_lp, p_lp = load_lp(fm, dataset)
    f_ft, y_ft, p_ft = load_ft(fm, dataset)
    # align FT features back to recording order matching LP
    # (LP cache is in original order; FT comes from CV folds — we use FT's own order)
    print(f" LP shape={f_lp.shape}  FT shape={f_ft.shape}")

    out = {"fm": fm,
           "n_recordings_lp": int(len(f_lp)),
           "n_recordings_ft": int(len(f_ft))}

    design = "crossed" if dataset in ("eegmat", "sleepdep") else "nested"
    for tag, (f, y, p) in [("lp", (f_lp, y_lp, p_lp)),
                            ("ft", (f_ft, y_ft, p_ft))]:
        lab_pct, subj_pct, res_pct = variance_fracs(f, y, p, design=design)
        knn = compute_knn(f, y, p, k=5)
        try:
            logme = compute_logme(f, y)
        except Exception as e:
            logme = {"logme": float("nan"), "err": str(e)}
        try:
            hsc = compute_hscore(f, y)
        except Exception as e:
            hsc = {"hscore": float("nan"), "err": str(e)}
        out[tag] = {
            "label_frac_pct": round(lab_pct, 4),
            "subject_frac_pct": round(subj_pct, 4),
            "residual_frac_pct": round(res_pct, 4),
            "subject_to_label_ratio": round(subj_pct / max(lab_pct, 1e-6), 3),
            "knn_ba": knn,
            "logme": logme,
            "hscore": hsc,
        }
        print(f" {tag.upper():3s}: label={lab_pct:5.2f}%  subj={subj_pct:5.2f}%  "
              f"res={res_pct:5.2f}%  knn={knn}  logme={logme.get('logme','?'):.3f}  "
              f"hscore={hsc.get('hscore','?'):.3f}")

    # CKA between LP and FT representations
    # Need same N — match by recording (FT is in CV-fold order, LP is original)
    # Sort both by (patient_id, label) to align
    def sort_key(y, p):
        return np.lexsort((y, p))
    idx_lp = sort_key(y_lp, p_lp)
    idx_ft = sort_key(y_ft, p_ft)
    if (y_lp[idx_lp] == y_ft[idx_ft]).all() and (p_lp[idx_lp] == p_ft[idx_ft]).all():
        cka = compute_cka(f_lp[idx_lp], f_ft[idx_ft])
    else:
        # FT fold subset may be different (e.g. mixed-label filter); align by intersection
        common = np.intersect1d(p_lp, p_ft)
        keep_lp = np.isin(p_lp, common)
        keep_ft = np.isin(p_ft, common)
        f_lp2, y_lp2, p_lp2 = f_lp[keep_lp], y_lp[keep_lp], p_lp[keep_lp]
        f_ft2, y_ft2, p_ft2 = f_ft[keep_ft], y_ft[keep_ft], p_ft[keep_ft]
        i_lp = sort_key(y_lp2, p_lp2); i_ft = sort_key(y_ft2, p_ft2)
        cka = compute_cka(f_lp2[i_lp], f_ft2[i_ft])
        print(f"  (CKA computed on {len(f_lp2)} common recordings after subject intersect)")

    out["cka_lp_vs_ft"] = cka
    print(f" CKA(LP, FT) = {cka}")

    # Mechanism delta
    dlab = out["ft"]["label_frac_pct"] - out["lp"]["label_frac_pct"]
    dsubj = out["ft"]["subject_frac_pct"] - out["lp"]["subject_frac_pct"]
    out["delta"] = {
        "label_frac_pct": round(dlab, 4),
        "subject_frac_pct": round(dsubj, 4),
        "logme": (out["ft"]["logme"].get("logme", float("nan"))
                  - out["lp"]["logme"].get("logme", float("nan"))),
        "interpretation": _interpret(dlab, dsubj),
    }
    print(f" Δ: label={dlab:+.2f}pp  subj={dsubj:+.2f}pp  → {out['delta']['interpretation']}")
    return out


def _interpret(dlab, dsubj):
    if dlab > 1.0 and dsubj <= 0:
        return "rescue_consistent_with_label_signal"
    if dsubj > 5.0 and dlab <= 1.0:
        return "rescue_consistent_with_subject_shortcut"
    if abs(dlab) < 1.0 and abs(dsubj) < 5.0:
        return "no_meaningful_drift"
    return "ambiguous"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = {"protocol": {
                  "lp_features": "frozen_<fm>_<dataset>_<ch>ch.npz (recording-avg)",
                  "ft_features": "ft_<fm>_<dataset>/fold[1-5]_features.npz "
                                 "(concat 5-fold OOF)",
                  "stress_filter": "drops mixed-label subjects (same as fig_variance_atlas)",
                  "ft_caveat_labram_stress": "lr=1e-4 (HP-sweep); canonical lr=1e-5 "
                                             "has no cached features",
                  "metrics": "variance frac (nested SS), LogME, H-score, CKA, kNN-LOO",
              }, "results": {}}
    for dataset in ["stress", "eegmat"]:
        results["results"][dataset] = {}
        for fm in ["labram", "cbramod", "reve"]:
            try:
                results["results"][dataset][fm] = analyze_one(fm, dataset)
            except Exception as e:
                print(f" FAIL {fm} × {dataset}: {e}")
                results["results"][dataset][fm] = {"error": str(e)}
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
