"""EEGMAT-specific variance analysis (within-subject positive control).

The standard `nested_ss` from src/variance_analysis.py requires pure-label
subjects (subject_id → unique label), which is true for Stress, ADFTD, and
TDBRAIN. EEGMAT is the deliberate counterexample: every subject contributes
both labels (rest=0 and arithmetic=1), so `nested_ss` raises.

The pooled label fraction $SS_{\text{label}} / SS_{\text{total}}$ is still
well-defined and computed exactly the same way — only the *interpretation*
differs (within-subject contrast vs between-subject diagnosis).

For EEGMAT we additionally use a *crossed* (not nested) decomposition:

    SS_total = SS_label + SS_subject + SS_label*subject_residual

The mixed-effects model `feat ~ label + (1|subject)` is the cleanest
parametric tool for the within-subject case — it gives the variance of the
label fixed effect after partialling out the subject random effect — and
runs the same way it does for the other three datasets.

Run under the new `stress` conda env (which has working scipy + statsmodels):

    /raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python \
        scripts/analyze_eegmat.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src import variance_analysis as va  # noqa: E402

FROZEN_NPZ = "results/features_cache/frozen_labram_eegmat_19ch.npz"
FT_DIR = "results/features_cache/ft_labram/eegmat_2026-04-08"
OUT_JSON = "paper/figures/variance_analysis_eegmat.json"


def pooled_label_fraction(features: np.ndarray, label: np.ndarray) -> dict:
    """Compute SS_label / SS_total summed over feature dims.

    This is the same number `nested_ss` would return for the label component,
    but works for any label structure (subject-level, within-subject, mixed).
    """
    f = np.asarray(features, dtype=np.float64)
    y = np.asarray(label)
    grand_mean = f.mean(axis=0)  # (D,)
    ss_total = ((f - grand_mean) ** 2).sum(axis=0)  # (D,)
    ss_label = np.zeros_like(ss_total)
    for lab in np.unique(y):
        mask = y == lab
        n = mask.sum()
        ss_label += n * (f[mask].mean(axis=0) - grand_mean) ** 2
    return {
        "ss_label_per_dim": ss_label,
        "ss_total_per_dim": ss_total,
        "pooled_label_fraction": float(ss_label.sum() / max(ss_total.sum(), 1e-18)),
        "n_recordings": int(f.shape[0]),
        "n_dims": int(f.shape[1]),
    }


def crossed_decomposition(
    features: np.ndarray, subject: np.ndarray, label: np.ndarray
) -> dict:
    """Crossed (not nested) sum-of-squares decomposition for within-subject
    labels.

    Each subject contributes recordings of multiple labels, so the design is
    crossed: subject × label. We report:
        SS_total          — total variance
        SS_label          — between-label variance (the within-subject contrast)
        SS_subject        — between-subject variance (collapsed over labels)
        SS_label_subject_resid — interaction + residual

    Pooled fractions are reported on the same SS_label / SS_total denominator
    as the other three datasets, so the headline number is directly comparable.
    """
    f = np.asarray(features, dtype=np.float64)
    s = np.asarray(subject)
    y = np.asarray(label)

    grand = f.mean(axis=0)
    ss_total = ((f - grand) ** 2).sum(axis=0)

    ss_label = np.zeros_like(ss_total)
    for lab in np.unique(y):
        mask = y == lab
        ss_label += mask.sum() * (f[mask].mean(axis=0) - grand) ** 2

    ss_subject = np.zeros_like(ss_total)
    for sub in np.unique(s):
        mask = s == sub
        ss_subject += mask.sum() * (f[mask].mean(axis=0) - grand) ** 2

    ss_resid = ss_total - ss_label - ss_subject  # may be negative for crossed
    # Clamp residual at 0 if negative due to non-orthogonality of factors;
    # report the raw value too for transparency.
    return {
        "n_recordings": int(f.shape[0]),
        "n_subjects": int(len(np.unique(s))),
        "n_labels": int(len(np.unique(y))),
        "pooled_fractions": {
            "label": float(ss_label.sum() / max(ss_total.sum(), 1e-18)),
            "subject": float(ss_subject.sum() / max(ss_total.sum(), 1e-18)),
            "residual": float(max(ss_resid.sum(), 0.0) / max(ss_total.sum(), 1e-18)),
            "raw_residual_signed": float(
                ss_resid.sum() / max(ss_total.sum(), 1e-18)
            ),
        },
        "ratio_subject_to_label": float(
            ss_subject.sum() / max(ss_label.sum(), 1e-18)
        ),
    }


def main():
    print("EEGMAT within-subject variance analysis")
    print("=" * 60)

    # Frozen features
    fz = np.load(FROZEN_NPZ)
    f_frozen = fz["features"]
    y_frozen = fz["labels"]
    s_frozen = fz["patient_ids"]
    print(f"Frozen: {f_frozen.shape}  subjects={len(np.unique(s_frozen))}  "
          f"label_balance={(y_frozen == 1).sum()}/{(y_frozen == 0).sum()}")

    # Fine-tuned (pooled OOF)
    ft_per_fold = va.load_ft_features_per_fold(FT_DIR)
    f_ft = np.concatenate([t[0] for t in ft_per_fold])
    y_ft = np.concatenate([t[1] for t in ft_per_fold])
    s_ft = np.concatenate([t[2] for t in ft_per_fold])
    print(f"FT (pooled): {f_ft.shape}  subjects={len(np.unique(s_ft))}  "
          f"label_balance={(y_ft == 1).sum()}/{(y_ft == 0).sum()}")

    print("\n--- Pooled label fraction ($SS_{label} / SS_{total}$) ---")
    plf_frozen = pooled_label_fraction(f_frozen, y_frozen)
    plf_ft = pooled_label_fraction(f_ft, y_ft)
    print(f"Frozen:    {plf_frozen['pooled_label_fraction'] * 100:6.2f}%")
    print(f"FT pooled: {plf_ft['pooled_label_fraction'] * 100:6.2f}%")
    if plf_frozen["pooled_label_fraction"] > 0:
        mult = plf_ft["pooled_label_fraction"] / plf_frozen["pooled_label_fraction"]
        print(f"Change:    ×{mult:.2f}")

    print("\n--- Crossed decomposition (subject × label) ---")
    cd_frozen = crossed_decomposition(f_frozen, s_frozen, y_frozen)
    cd_ft = crossed_decomposition(f_ft, s_ft, y_ft)
    print(f"FROZEN  label_frac={cd_frozen['pooled_fractions']['label']*100:5.2f}%  "
          f"subject_frac={cd_frozen['pooled_fractions']['subject']*100:5.2f}%  "
          f"ratio S/L={cd_frozen['ratio_subject_to_label']:.2f}")
    print(f"FT POOL label_frac={cd_ft['pooled_fractions']['label']*100:5.2f}%  "
          f"subject_frac={cd_ft['pooled_fractions']['subject']*100:5.2f}%  "
          f"ratio S/L={cd_ft['ratio_subject_to_label']:.2f}")

    print("\n--- Mixed-effects: feat_d ~ label + (1|subject) ---")
    me_frozen = va.mixed_effects_variance(
        f_frozen, s_frozen, y_frozen, max_dims=None, seed=0
    )
    me_ft = va.mixed_effects_variance(f_ft, s_ft, y_ft, max_dims=None, seed=0)
    if "error" not in me_frozen:
        print(f"FROZEN  ICC={me_frozen['icc_subject_mean']:.3f}  "
              f"frac_label={me_frozen['frac_label_mean']:.4f}  "
              f"converged={me_frozen['n_converged']}/{me_frozen['n_dims_tried']}")
    else:
        print(f"FROZEN mixed-effects error: {me_frozen['error']}")
    if "error" not in me_ft:
        print(f"FT POOL ICC={me_ft['icc_subject_mean']:.3f}  "
              f"frac_label={me_ft['frac_label_mean']:.4f}  "
              f"converged={me_ft['n_converged']}/{me_ft['n_dims_tried']}")
    else:
        print(f"FT POOL mixed-effects error: {me_ft['error']}")

    print("\n--- Subject-level PERMANOVA ---")
    # For within-subject design, "subject-level permutation" doesn't make sense
    # the same way (label is constant within subject). Permute labels
    # *within each subject* to test whether the rest/task contrast survives
    # over and above subject identity.
    print("(skipped: subject-level permutation is degenerate when labels are "
          "within-subject; rely on mixed-effects p-values instead.)")

    # Save output JSON
    out = {
        "dataset": "EEGMAT",
        "ba": 0.7361,
        "design": "within_subject_crossed",
        "frozen": {
            "n_recordings": plf_frozen["n_recordings"],
            "pooled_label_fraction": plf_frozen["pooled_label_fraction"],
            "crossed": cd_frozen,
            "mixed_effects": {
                "icc_subject_mean": me_frozen.get("icc_subject_mean"),
                "frac_label_mean": me_frozen.get("frac_label_mean"),
                "n_converged": me_frozen.get("n_converged"),
                "n_dims_tried": me_frozen.get("n_dims_tried"),
            } if "error" not in me_frozen else {"error": me_frozen["error"]},
        },
        "ft_pooled": {
            "n_recordings": plf_ft["n_recordings"],
            "pooled_label_fraction": plf_ft["pooled_label_fraction"],
            "crossed": cd_ft,
            "mixed_effects": {
                "icc_subject_mean": me_ft.get("icc_subject_mean"),
                "frac_label_mean": me_ft.get("frac_label_mean"),
                "n_converged": me_ft.get("n_converged"),
                "n_dims_tried": me_ft.get("n_dims_tried"),
            } if "error" not in me_ft else {"error": me_ft["error"]},
        },
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
