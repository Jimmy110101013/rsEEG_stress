"""Variance decomposition analysis for EEG foundation-model representations.

Single source of truth for the subject-vs-label variance analysis used in the
cross-dataset signal-strength figure (paper §4.3) and the FM diagnosis
notebook. Replaces the three duplicated `eta_squared` implementations
previously scattered across `scripts/build_cross_dataset_figure.py`,
`notebooks/FM_Representation_Diagnosis.ipynb`, and
`notebooks/Cross_Dataset_Signal_Strength.ipynb`.

# Math summary

In all our datasets, *label is constant within subject* (every recording from
subject s has the same diagnosis / stress label). This means the naive
one-way η² ratio η²_subject / η²_label is structurally biased — subject SS
contains label SS by construction. We therefore use:

1. **Nested decomposition** (`nested_ss`):
       SS_total = SS_label + SS_subject|label + SS_residual
   where SS_subject|label measures between-subject variance *within* each
   label group.

2. **df-corrected ω²** (`omega_squared_from_ss`): unbiased version of η²,
   making cross-dataset comparison fair when subject counts differ.

3. **Mixed-effects model** (`mixed_effects_variance`, lazy-imports
   statsmodels): fits feat_d ~ label + (1 | subject) per dim and returns
   variance components σ²_subject / σ²_label / σ²_residual; ICC is
   σ²_subject / (σ²_subject + σ²_residual). This is the headline number we
   report in the paper.

4. **Cluster bootstrap** (`cluster_bootstrap`): all CIs resample at the
   *subject* level (with replacement), never at the recording level —
   recordings within a subject are not independent, so a recording-level
   bootstrap produces anti-conservative intervals.

5. **PERMANOVA** (`subject_level_permanova`): multivariate non-parametric
   robustness check. Permutes labels at the *subject* level (every recording
   of subject s gets the new label together), so exchangeability holds.

# Environment

This module is numpy-only at top level — safe to import from `timm_eeg`
notebooks. `mixed_effects_variance` lazy-imports `statsmodels`, which is
broken in `timm_eeg` (scipy ABI issue) — call it only from `stats_env`.
The CLI `scripts/run_variance_analysis.py` should be invoked via
`conda run -n stats_env python scripts/run_variance_analysis.py`.

# Reviewer-defensible quantities

Every quantity reported by `analyze_dataset` is one of:
- A nested ω² (df-corrected, not the naive per-dim η² average).
- A mixed-effects variance fraction (REML, label as fixed effect).
- An ICC computed from the mixed-effects components.
- A cluster-bootstrap mean ± 95% CI on any of the above.
- A subject-level permutation p-value from PERMANOVA.
"""
from __future__ import annotations

import json
import os
from glob import glob
from typing import Callable, Sequence

import numpy as np


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------
def load_ft_features(run_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate test-fold features across all CV folds in a run directory.

    Returns (features, labels, patient_ids) with one row per recording.
    Each row is the embedding produced by the fold-model in which that
    recording was held out.
    """
    feats, labels, pids = [], [], []
    for npz in sorted(glob(f"{run_dir}/fold*_features.npz")):
        d = np.load(npz)
        feats.append(d["features"])
        labels.append(d["labels"])
        pids.append(d["patient_ids"])
    if not feats:
        raise FileNotFoundError(f"No fold*_features.npz in {run_dir}")
    return (
        np.concatenate(feats),
        np.concatenate(labels),
        np.concatenate(pids),
    )


def load_ft_features_per_fold(
    run_dir: str,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Like load_ft_features but preserves fold separation.

    Returns a list of (feats, labels, pids) tuples — one per CV fold.
    Used for per-fold variance decomposition so the fold-mixing confound
    (5 different fine-tuned models pooled into one matrix) is avoided.
    """
    out = []
    for npz in sorted(glob(f"{run_dir}/fold*_features.npz")):
        d = np.load(npz)
        out.append((d["features"], d["labels"], d["patient_ids"]))
    if not out:
        raise FileNotFoundError(f"No fold*_features.npz in {run_dir}")
    return out


def load_frozen_features(
    npz_path: str, ft_run_dir: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load frozen-FM features and recover labels/pids if missing.

    If labels/pids are stored in the npz, use them directly. Otherwise
    derive them from the FT-feature dir's `test_idx` field — concatenate
    all folds, sort by test_idx to recover the original dataset order,
    and align to the frozen-feature row order.
    """
    d = np.load(npz_path)
    feats = d["features"]
    if "labels" in d.files and "patient_ids" in d.files:
        return feats, d["labels"], d["patient_ids"]

    if ft_run_dir is None:
        raise ValueError(
            f"{npz_path} has no labels/pids and no FT fallback dir provided"
        )

    test_idxs, labels, pids = [], [], []
    for npz in sorted(glob(f"{ft_run_dir}/fold*_features.npz")):
        d2 = np.load(npz)
        test_idxs.append(d2["test_idx"])
        labels.append(d2["labels"])
        pids.append(d2["patient_ids"])
    test_idxs = np.concatenate(test_idxs)
    labels = np.concatenate(labels)
    pids = np.concatenate(pids)

    order = np.argsort(test_idxs)
    labels_sorted = labels[order]
    pids_sorted = pids[order]
    idx_sorted = test_idxs[order]

    n = feats.shape[0]
    if not np.array_equal(idx_sorted, np.arange(n)):
        # Some recordings dropped during training — match by index.
        out_labels = np.full(n, -1, dtype=labels_sorted.dtype)
        out_pids = np.full(n, -1, dtype=pids_sorted.dtype)
        out_labels[idx_sorted] = labels_sorted
        out_pids[idx_sorted] = pids_sorted
        valid = out_labels >= 0
        return feats[valid], out_labels[valid], out_pids[valid]

    return feats, labels_sorted, pids_sorted


# ----------------------------------------------------------------------
# Nested sum-of-squares decomposition
# ----------------------------------------------------------------------
def nested_decomposition_is_identifiable(
    subject: np.ndarray, label: np.ndarray, min_subjects_per_label: int = 2
) -> tuple[bool, str]:
    """Check whether a nested subject-within-label decomposition can be
    meaningfully computed for this data.

    When a label class has fewer than 2 subjects, SS_subject_within_label
    for that class is structurally 0 — all within-class variation is
    attributed to residual, and the decomposition is degenerate. This
    happens in e.g. Stress per-fold analysis where every fold has only
    1 subject in the "increase" class.

    Returns (is_ok, reason). Callers should warn or skip if is_ok is False.
    """
    s = np.asarray(subject)
    y = np.asarray(label)
    for lab in np.unique(y):
        mask = y == lab
        n_subj = len(np.unique(s[mask]))
        if n_subj < min_subjects_per_label:
            return False, (
                f"label={lab} has only {n_subj} subject(s); need >= "
                f"{min_subjects_per_label} for nested decomposition to be "
                f"identifiable"
            )
    return True, "ok"


def nested_ss(
    features: np.ndarray, subject: np.ndarray, label: np.ndarray
) -> dict[str, np.ndarray]:
    """Nested ANOVA sum-of-squares decomposition: subject nested in label.

    Each subject must be pure-label (all recordings from one subject share
    the same label) — this is true for all our datasets because diagnosis
    is a subject-level property.

    Decomposition (per feature dim, vectorized):
        SS_total = SS_label + SS_subject_within_label + SS_residual

    where:
        SS_label             = sum_l n_l * (mean_l - grand)^2
        SS_subject|label     = sum_l sum_{s in l} n_{s,l} * (mean_{s,l} - mean_l)^2
        SS_residual          = sum_i (x_i - mean_{s(i),l(i)})^2

    Returns dict with keys 'label', 'subject_within_label', 'residual',
    'total' — each value is a (D,) array of per-dim sums of squares plus
    df counts in 'df_label', 'df_subject_within_label', 'df_residual'
    (scalars).
    """
    f = np.asarray(features, dtype=np.float64)
    s = np.asarray(subject)
    y = np.asarray(label)
    N, D = f.shape

    grand = f.mean(axis=0, keepdims=True)
    ss_total = ((f - grand) ** 2).sum(axis=0)

    # Verify nested structure: each subject must have a unique label.
    label_per_subject: dict = {}
    for sid, lab in zip(s, y):
        if sid in label_per_subject and label_per_subject[sid] != lab:
            raise ValueError(
                f"Subject {sid} has multiple labels — nested decomposition "
                "requires pure-label subjects."
            )
        label_per_subject[sid] = lab

    unique_labels = np.unique(y)
    ss_label = np.zeros(D)
    ss_subject_within_label = np.zeros(D)
    df_label = len(unique_labels) - 1
    df_subject_within_label = 0

    # Per-recording subject-mean storage for residual computation.
    subj_mean_per_record = np.zeros_like(f)

    for lab in unique_labels:
        lab_mask = y == lab
        n_l = lab_mask.sum()
        if n_l < 1:
            continue
        mean_l = f[lab_mask].mean(axis=0)
        ss_label += n_l * (mean_l - grand.squeeze()) ** 2

        subjects_in_label = np.unique(s[lab_mask])
        df_subject_within_label += max(len(subjects_in_label) - 1, 0)

        for sid in subjects_in_label:
            sub_mask = (s == sid) & lab_mask
            n_sl = sub_mask.sum()
            if n_sl < 1:
                continue
            mean_sl = f[sub_mask].mean(axis=0)
            ss_subject_within_label += n_sl * (mean_sl - mean_l) ** 2
            subj_mean_per_record[sub_mask] = mean_sl

    ss_residual = ((f - subj_mean_per_record) ** 2).sum(axis=0)
    df_residual = N - sum(len(np.unique(s[y == lab])) for lab in unique_labels)

    return {
        "label": ss_label,
        "subject_within_label": ss_subject_within_label,
        "residual": ss_residual,
        "total": ss_total,
        "df_label": int(df_label),
        "df_subject_within_label": int(df_subject_within_label),
        "df_residual": int(df_residual),
        "n_recordings": int(N),
        "n_subjects": int(len(np.unique(s))),
    }


def omega_squared_from_ss(ss: dict) -> dict[str, float]:
    """Convert nested SS dict to df-corrected ω² (per-dim averaged scalars).

    ω² is the unbiased (downward-corrected) version of η². For an effect
    with df_effect degrees of freedom and mean-square-error MS_resid:
        ω² = (SS_effect - df_effect * MS_resid) / (SS_total + MS_resid)

    We use the *residual* MS as the error term (not subject MS) — this is
    the standard convention for nested designs when the higher-level factor
    (label) is fixed.

    Returns dict with 'omega2_label', 'omega2_subject_within_label',
    plus the convenient ratio 'subject_to_label_omega2' and
    'frac_label', 'frac_subject_within_label', 'frac_residual'
    (per-dim averaged variance fractions).
    """
    df_resid = ss["df_residual"]
    if df_resid <= 0:
        raise ValueError("df_residual <= 0; not enough recordings per subject")
    ms_resid = ss["residual"] / df_resid

    def _omega2(ss_effect: np.ndarray, df_effect: int) -> float:
        num = ss_effect - df_effect * ms_resid
        den = ss["total"] + ms_resid
        # Per-dim, then average. Negative values clipped to 0 (small-sample
        # noise — ω² can go slightly negative in finite samples).
        per_dim = np.clip(num / np.maximum(den, 1e-12), 0.0, None)
        return float(per_dim.mean())

    # Variance fractions (uncorrected, easier to interpret).
    total_safe = np.maximum(ss["total"], 1e-12)
    frac_label = float((ss["label"] / total_safe).mean())
    frac_swl = float((ss["subject_within_label"] / total_safe).mean())
    frac_resid = float((ss["residual"] / total_safe).mean())

    omega2_label = _omega2(ss["label"], ss["df_label"])
    omega2_swl = _omega2(
        ss["subject_within_label"], ss["df_subject_within_label"]
    )

    return {
        "omega2_label": omega2_label,
        "omega2_subject_within_label": omega2_swl,
        "subject_to_label_omega2": (
            omega2_swl / max(omega2_label, 1e-9)
        ),
        "frac_label": frac_label,
        "frac_subject_within_label": frac_swl,
        "frac_residual": frac_resid,
    }


# ----------------------------------------------------------------------
# Mixed-effects model (statsmodels — must run in stats_env)
# ----------------------------------------------------------------------
def mixed_effects_variance(
    features: np.ndarray,
    subject: np.ndarray,
    label: np.ndarray,
    max_dims: int | None = None,
    seed: int = 42,
) -> dict:
    """Per-dim mixed-effects variance decomposition: feat_d ~ label + (1|subject).

    Uses REML via statsmodels.MixedLM. For each feature dim:
        σ²_subject  = result.cov_re.iloc[0, 0]   (random intercept variance)
        σ²_residual = result.scale                (residual variance)
        σ²_label    = β_label² · Var(label)       (variance explained by fixed effect)

    ICC_subject = σ²_subject / (σ²_subject + σ²_residual) — the standard
    "fraction of residual variance attributable to subject clustering after
    accounting for the fixed label effect."

    Caller must pass arrays where each subject has exactly one label
    (we don't enforce this here; nested_ss does).

    `max_dims` subsamples feature dims for speed (default: all dims).

    Returns dict with mean variance fractions, mean ICC, and convergence
    counts. Returns {'error': '...'} if statsmodels can't import or no
    dim converges.
    """
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except Exception as exc:  # pragma: no cover
        return {
            "error": f"statsmodels unavailable ({type(exc).__name__}: {exc}). "
            "Run from stats_env: conda run -n stats_env python ..."
        }

    f = np.asarray(features, dtype=np.float64)
    s = np.asarray(subject)
    y = np.asarray(label, dtype=np.float64)
    N, D = f.shape

    rng = np.random.RandomState(seed)
    if max_dims is None or max_dims >= D:
        dims = np.arange(D)
    else:
        dims = rng.choice(D, size=max_dims, replace=False)

    var_subj_list, var_lab_list, var_resid_list = [], [], []
    icc_list = []
    converged = 0

    label_var = float(np.var(y, ddof=0))

    for d in dims:
        df = pd.DataFrame({"feat": f[:, d], "label": y, "subject": s})
        try:
            model = smf.mixedlm("feat ~ label", df, groups=df["subject"])
            result = model.fit(method="lbfgs", disp=False, reml=True, maxiter=200)
        except Exception:
            continue
        if not getattr(result, "converged", False):
            continue

        var_subj = float(result.cov_re.iloc[0, 0])
        var_resid = float(result.scale)
        beta_label = float(result.params.get("label", 0.0))
        var_label = (beta_label ** 2) * label_var

        denom = var_subj + var_resid
        if denom <= 1e-12:
            continue
        icc = var_subj / denom

        var_subj_list.append(var_subj)
        var_lab_list.append(var_label)
        var_resid_list.append(var_resid)
        icc_list.append(icc)
        converged += 1

    if converged == 0:
        return {"error": "no feature dim converged"}

    var_subj_arr = np.array(var_subj_list)
    var_lab_arr = np.array(var_lab_list)
    var_resid_arr = np.array(var_resid_list)
    total_per_dim = var_subj_arr + var_lab_arr + var_resid_arr

    return {
        "icc_subject_mean": float(np.mean(icc_list)),
        "icc_subject_median": float(np.median(icc_list)),
        "var_subject_mean": float(var_subj_arr.mean()),
        "var_label_mean": float(var_lab_arr.mean()),
        "var_residual_mean": float(var_resid_arr.mean()),
        "frac_subject_mean": float(np.mean(var_subj_arr / np.maximum(total_per_dim, 1e-12))),
        "frac_label_mean":   float(np.mean(var_lab_arr  / np.maximum(total_per_dim, 1e-12))),
        "frac_residual_mean": float(np.mean(var_resid_arr / np.maximum(total_per_dim, 1e-12))),
        "subject_to_label_var_ratio": float(var_subj_arr.mean() / max(var_lab_arr.mean(), 1e-12)),
        "n_converged": converged,
        "n_dims_tried": int(len(dims)),
    }


# ----------------------------------------------------------------------
# Cluster bootstrap
# ----------------------------------------------------------------------
def cluster_bootstrap(
    features: np.ndarray,
    subject: np.ndarray,
    label: np.ndarray,
    statistic_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 0,
    log_transform: bool = False,
) -> dict:
    """Cluster bootstrap that resamples subjects (not recordings) with replacement.

    Recordings within a subject are not independent (shared brain, shared
    label, shared session), so a naive recording-level bootstrap produces
    anti-conservative confidence intervals. We resample whole subjects and
    rebuild the per-recording arrays.

    `statistic_fn(features, subject, label) -> float` is called once per
    bootstrap iteration on the resampled data.

    `log_transform=True` is recommended when the statistic is a ratio
    (heavy-tailed bootstrap distribution); we exponentiate the percentile
    bounds at the end.

    Returns dict with point estimate (statistic on the original data),
    bootstrap mean, 2.5/97.5 percentile CI, and the raw bootstrap samples
    (in case downstream wants a different summary).
    """
    rng = np.random.RandomState(seed)
    s = np.asarray(subject)
    f = np.asarray(features)
    y = np.asarray(label)

    # Build per-subject record indices once.
    unique_subj = np.unique(s)
    sub_to_rows = {sid: np.where(s == sid)[0] for sid in unique_subj}
    n_subj = len(unique_subj)

    # Point estimate on original data.
    point = float(statistic_fn(f, s, y))

    samples = []
    for _ in range(n_boot):
        chosen = rng.choice(unique_subj, size=n_subj, replace=True)
        rows = np.concatenate([sub_to_rows[sid] for sid in chosen])
        # Rebuild a synthetic per-recording array. Note: when the same
        # subject is drawn twice we duplicate its rows; we also rename
        # duplicated subjects to keep them distinct in the resampled data.
        new_subj = np.concatenate([
            np.full(len(sub_to_rows[sid]), f"{sid}_{i}", dtype=object)
            for i, sid in enumerate(chosen)
        ])
        try:
            stat = float(statistic_fn(f[rows], new_subj, y[rows]))
        except Exception:
            continue
        if not np.isfinite(stat):
            continue
        samples.append(stat)

    samples = np.array(samples)
    if len(samples) == 0:
        return {"point": point, "mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_valid": 0}

    if log_transform:
        positive = samples[samples > 0]
        if len(positive) >= 10:
            log_samples = np.log(positive)
            mean_log = log_samples.mean()
            lo_log, hi_log = np.percentile(log_samples, [2.5, 97.5])
            return {
                "point": point,
                "mean": float(np.exp(mean_log)),
                "ci_low": float(np.exp(lo_log)),
                "ci_high": float(np.exp(hi_log)),
                "n_valid": int(len(samples)),
            }

    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {
        "point": point,
        "mean": float(samples.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_valid": int(len(samples)),
    }


# ----------------------------------------------------------------------
# PERMANOVA (subject-level permutation)
# ----------------------------------------------------------------------
def _pseudo_f(dist_sq: np.ndarray, group: np.ndarray) -> float:
    """Anderson (2001) PERMANOVA pseudo-F statistic from a squared-distance
    matrix and group assignments.

        SS_total  = sum_{i<j} d²_ij / N
        SS_within = sum_g [ sum_{i<j in g} d²_ij / n_g ]
        SS_among  = SS_total - SS_within
        F = (SS_among / (a-1)) / (SS_within / (N-a))
    """
    n = dist_sq.shape[0]
    iu = np.triu_indices(n, k=1)
    total_sum = dist_sq[iu].sum()
    ss_total = total_sum / n

    ss_within = 0.0
    groups = np.unique(group)
    a = len(groups)
    for g in groups:
        idx = np.where(group == g)[0]
        n_g = len(idx)
        if n_g < 2:
            continue
        sub = dist_sq[np.ix_(idx, idx)]
        ss_within += sub[np.triu_indices(n_g, k=1)].sum() / n_g

    ss_among = ss_total - ss_within
    df_among = max(a - 1, 1)
    df_within = max(n - a, 1)
    if ss_within <= 0:
        return float("inf")
    return float((ss_among / df_among) / (ss_within / df_within))


def subject_level_permanova(
    features: np.ndarray,
    subject: np.ndarray,
    label: np.ndarray,
    n_perm: int = 999,
    seed: int = 0,
) -> dict:
    """Multivariate PERMANOVA on Euclidean distances with subject-level
    permutation.

    Permutes labels at the *subject* level — every recording of a subject
    receives the new label together — preserving exchangeability under
    the null of "no label effect at the subject level". Recording-level
    permutation would inflate F (pseudoreplication) and is rejected.

    Returns dict with pseudo-F, permutation p-value, and R² = SS_among/SS_total.
    """
    rng = np.random.RandomState(seed)
    f = np.asarray(features, dtype=np.float64)
    s = np.asarray(subject)
    y = np.asarray(label)

    # Standardize features so distances are not dominated by high-variance dims.
    f_std = (f - f.mean(0)) / np.maximum(f.std(0), 1e-12)

    # Squared Euclidean distance matrix.
    sq = np.sum(f_std ** 2, axis=1)
    dist_sq = sq[:, None] + sq[None, :] - 2 * f_std @ f_std.T
    np.fill_diagonal(dist_sq, 0.0)
    dist_sq = np.maximum(dist_sq, 0.0)

    f_obs = _pseudo_f(dist_sq, y)

    # Subject-level permutation: shuffle the subject→label mapping.
    unique_subj = np.unique(s)
    subj_to_label = np.array([y[s == sid][0] for sid in unique_subj])

    perm_count = 0
    for _ in range(n_perm):
        perm_subj_label = rng.permutation(subj_to_label)
        sid_to_lab = dict(zip(unique_subj, perm_subj_label))
        y_perm = np.array([sid_to_lab[sid] for sid in s])
        f_perm = _pseudo_f(dist_sq, y_perm)
        if f_perm >= f_obs:
            perm_count += 1

    p = (perm_count + 1) / (n_perm + 1)

    # R²: SS_among / SS_total
    n = len(s)
    iu = np.triu_indices(n, k=1)
    ss_total = dist_sq[iu].sum() / n
    ss_within = 0.0
    for g in np.unique(y):
        idx = np.where(y == g)[0]
        n_g = len(idx)
        if n_g < 2:
            continue
        sub = dist_sq[np.ix_(idx, idx)]
        ss_within += sub[np.triu_indices(n_g, k=1)].sum() / n_g
    r2 = (ss_total - ss_within) / max(ss_total, 1e-12)

    return {"pseudo_F": float(f_obs), "p_value": float(p), "R2": float(r2),
            "n_perm": int(n_perm)}


# ----------------------------------------------------------------------
# Subsampling (matched-N comparison)
# ----------------------------------------------------------------------
def subsample_matched(
    features: np.ndarray,
    subject: np.ndarray,
    label: np.ndarray,
    n_per_label: dict[int, int],
    n_draws: int = 10,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Draw `n_draws` subsamples matching a target subject count per label.

    Generalizes the ADFTD-to-17-subjects matching code. Subjects are
    sampled *without replacement* per label class. Returns a list of
    (feats, subject, label) tuples that downstream code can feed to
    nested_ss / mixed_effects_variance / etc.
    """
    f = np.asarray(features)
    s = np.asarray(subject)
    y = np.asarray(label)

    pid_to_label = {}
    for sid, lab in zip(s, y):
        pid_to_label[sid] = lab
    unique_subj = np.array(list(pid_to_label.keys()))
    subj_label = np.array([pid_to_label[sid] for sid in unique_subj])

    rng_master = np.random.RandomState(seed)
    out = []
    for draw in range(n_draws):
        rng = np.random.RandomState(rng_master.randint(0, 2**31 - 1))
        chosen = []
        for lab, n_take in n_per_label.items():
            pool = unique_subj[subj_label == lab]
            if len(pool) < n_take:
                raise ValueError(
                    f"Label {lab} has only {len(pool)} subjects, need {n_take}"
                )
            picked = rng.choice(pool, size=n_take, replace=False)
            chosen.extend(picked.tolist())
        chosen = np.array(chosen)
        mask = np.isin(s, chosen)
        out.append((f[mask], s[mask], y[mask]))
    return out


# ----------------------------------------------------------------------
# Label-subspace concentration analysis
# ----------------------------------------------------------------------
def _label_ss_per_axis(features: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Per-axis between-label sum-of-squares (one value per column).

    Vectorized version of `sum_l n_l * (mean_l - grand)^2`.
    """
    f = np.asarray(features, dtype=np.float64)
    y = np.asarray(label)
    grand = f.mean(axis=0)
    label_ss = np.zeros(f.shape[1])
    for lab in np.unique(y):
        mask = y == lab
        n_l = mask.sum()
        if n_l < 1:
            continue
        label_ss += n_l * (f[mask].mean(axis=0) - grand) ** 2
    return label_ss


def label_subspace_analysis(
    features: np.ndarray,
    label: np.ndarray,
    n_pcs_report: int = 20,
) -> dict:
    """Characterize how label variance is distributed across feature axes.

    Tests the "sparse label subspace" hypothesis: after fine-tuning with
    small N, does label variance concentrate in a few dimensions while
    the complementary subspace gets noisier? If yes, per-dim averaged
    ω² sees a clean improvement (dominated by the label-loud minority)
    but multivariate Euclidean PERMANOVA does not (dominated by the
    noisy majority).

    Two parallel analyses:

    **Raw-dim basis** (the original 200 LaBraM dims):
        - Per-dim label SS, sorted descending.
        - `dims_for_50pct / 80pct / 95pct`: how many top-sorted dims are
          needed to capture that fraction of *total* label SS.
        - `top_k_label_fraction[k]`: cumulative fraction at rank k (k <= n_pcs_report).
        - `participation_ratio_label`: effective dimensionality of the
          normalized per-dim label-variance distribution. Equals total
          dims if label SS is uniform, 1 if fully concentrated in one dim.
        - `participation_ratio_total`: same for total per-dim variance
          (the "noise budget" shape). Subject-dominant features tend to
          have high PR_total (variance spread broadly).

    **PCA basis** (features projected onto principal axes):
        - Same concentration metrics computed on PCs.
        - Additionally `pc_variance_explained[k]` (fraction of total
          variance in PC k) and `pc_label_over_variance[k]` (per-PC
          η²_label — how label-aligned is PC k?). Sorted by PC variance
          (standard PCA order), NOT by label loading.

    The smoking-gun comparison for the sparse-label-subspace hypothesis:

        frozen.raw.dims_for_80pct   vs   ft.raw.dims_for_80pct
        frozen.pc.dims_for_80pct    vs   ft.pc.dims_for_80pct

    If FT values are substantially smaller than frozen, the label signal
    became concentrated. If they're similar, the signal is spread and
    the ω²↓ / PERMANOVA→ tension needs a different explanation.
    """
    f = np.asarray(features, dtype=np.float64)
    y = np.asarray(label)
    N, D = f.shape

    def _concentration_metrics(ss_vec: np.ndarray, total_var: np.ndarray) -> dict:
        sorted_ss = np.sort(ss_vec)[::-1]
        total = sorted_ss.sum()
        if total <= 0:
            return {
                "dims_for_50pct": -1, "dims_for_80pct": -1, "dims_for_95pct": -1,
                "top_k_label_fraction": [0.0] * n_pcs_report,
                "participation_ratio_label": float("nan"),
                "participation_ratio_total": float("nan"),
            }
        cum = np.cumsum(sorted_ss) / total
        def _rank_at(th: float) -> int:
            idx = np.searchsorted(cum, th)
            return int(idx + 1) if idx < len(cum) else int(len(cum))
        k_max = min(n_pcs_report, len(sorted_ss))
        top_k = cum[:k_max].tolist()
        # Participation ratio: (sum w)^2 / sum(w^2)
        pr_label = float(total ** 2 / max(float((sorted_ss ** 2).sum()), 1e-18))
        tv = np.asarray(total_var, dtype=np.float64)
        tv_total = tv.sum()
        if tv_total > 0:
            pr_total = float(tv_total ** 2 / max(float((tv ** 2).sum()), 1e-18))
        else:
            pr_total = float("nan")
        return {
            "dims_for_50pct": _rank_at(0.50),
            "dims_for_80pct": _rank_at(0.80),
            "dims_for_95pct": _rank_at(0.95),
            "top_k_label_fraction": top_k,
            "participation_ratio_label": pr_label,
            "participation_ratio_total": pr_total,
        }

    # ----- Raw-dim basis -----
    raw_label_ss = _label_ss_per_axis(f, y)
    raw_total_var = f.var(axis=0, ddof=0) * N  # per-dim SS_total
    raw_metrics = _concentration_metrics(raw_label_ss, raw_total_var)

    # Per-dim η²_label for top dims (sorted by label SS)
    raw_eta2 = raw_label_ss / np.maximum(raw_total_var, 1e-18)
    raw_eta2_sorted = np.sort(raw_eta2)[::-1][:n_pcs_report].tolist()
    raw_metrics["top_k_eta2_label"] = raw_eta2_sorted

    # ----- PCA basis -----
    # Center. SVD of centered features: F = U S V^T.
    centered = f - f.mean(axis=0, keepdims=True)
    # Use economy SVD; V has shape (min(N,D), D) → rows are PC axes.
    u, sv, vt = np.linalg.svd(centered, full_matrices=False)
    # PC scores: U * S, shape (N, K) where K = min(N, D).
    pc_scores = u * sv
    K = pc_scores.shape[1]
    pc_label_ss = _label_ss_per_axis(pc_scores, y)
    pc_total_var = (sv ** 2)  # per-PC total SS
    pc_metrics = _concentration_metrics(pc_label_ss, pc_total_var)
    # Variance-explained per PC (sorted by variance, standard PCA order):
    total_var_sum = float(pc_total_var.sum())
    var_explained = (
        (pc_total_var / max(total_var_sum, 1e-18)).tolist()[:n_pcs_report]
    )
    # Per-PC η²_label in PCA order (NOT sorted by label).
    pc_eta2_label_in_order = (
        pc_label_ss / np.maximum(pc_total_var, 1e-18)
    ).tolist()[:n_pcs_report]
    pc_metrics["variance_explained_top_k"] = var_explained
    pc_metrics["eta2_label_per_pc_top_k"] = pc_eta2_label_in_order

    return {
        "n_recordings": int(N),
        "n_dims": int(D),
        "n_pcs": int(K),
        "raw": raw_metrics,
        "pc": pc_metrics,
    }


# ----------------------------------------------------------------------
# Unified per-dataset analysis
# ----------------------------------------------------------------------
def _nested_omega2_subject_label_ratio(
    f: np.ndarray, s: np.ndarray, y: np.ndarray
) -> float:
    """Statistic helper used by cluster_bootstrap."""
    ss = nested_ss(f, s, y)
    omega = omega_squared_from_ss(ss)
    return omega["subject_to_label_omega2"]


def _nested_omega2_subject(f: np.ndarray, s: np.ndarray, y: np.ndarray) -> float:
    ss = nested_ss(f, s, y)
    return omega_squared_from_ss(ss)["omega2_subject_within_label"]


def _nested_omega2_label(f: np.ndarray, s: np.ndarray, y: np.ndarray) -> float:
    ss = nested_ss(f, s, y)
    return omega_squared_from_ss(ss)["omega2_label"]


def analyze_regime(
    f: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 1000,
    n_perm: int = 999,
    mixed_effects_max_dims: int | None = None,
    do_permanova: bool = True,
    do_mixed_effects: bool = True,
    seed: int = 0,
) -> dict:
    """Run the full reporting suite on a single (features, subject, label)
    tuple. Returns the dict that downstream consumers (figure script,
    paper) read."""
    identifiable, why = nested_decomposition_is_identifiable(s, y)
    ss = nested_ss(f, s, y)
    omega = omega_squared_from_ss(ss)

    boot_subject = cluster_bootstrap(
        f, s, y, _nested_omega2_subject, n_boot=n_boot, seed=seed
    )
    boot_label = cluster_bootstrap(
        f, s, y, _nested_omega2_label, n_boot=n_boot, seed=seed + 1
    )
    boot_ratio = cluster_bootstrap(
        f, s, y, _nested_omega2_subject_label_ratio,
        n_boot=n_boot, seed=seed + 2, log_transform=True,
    )

    out = {
        "n_recordings": ss["n_recordings"],
        "n_subjects": ss["n_subjects"],
        "nested_identifiable": identifiable,
        "nested_identifiable_reason": why,
        "df": {
            "label": ss["df_label"],
            "subject_within_label": ss["df_subject_within_label"],
            "residual": ss["df_residual"],
        },
        "nested_omega2": {
            "label": omega["omega2_label"],
            "subject_within_label": omega["omega2_subject_within_label"],
            "ratio_subject_to_label": omega["subject_to_label_omega2"],
        },
        "variance_fractions_uncorrected": {
            "label": omega["frac_label"],
            "subject_within_label": omega["frac_subject_within_label"],
            "residual": omega["frac_residual"],
        },
        # Pooled label/subject fraction: sum(SS) ratios rather than
        # per-dim averaged ratios. This is the reviewer-ready multivariate
        # metric — directly comparable to PERMANOVA R² and to what you
        # compute by hand on the raw feature matrix. Primary metric in
        # the paper.
        "pooled_fractions": {
            "label": float(ss["label"].sum() / max(ss["total"].sum(), 1e-18)),
            "subject_within_label": float(
                ss["subject_within_label"].sum() / max(ss["total"].sum(), 1e-18)
            ),
            "residual": float(ss["residual"].sum() / max(ss["total"].sum(), 1e-18)),
        },
        "bootstrap": {
            "omega2_subject": boot_subject,
            "omega2_label": boot_label,
            "ratio_subject_to_label": boot_ratio,
        },
    }

    if do_mixed_effects:
        me = mixed_effects_variance(
            f, s, y, max_dims=mixed_effects_max_dims, seed=seed
        )
        out["mixed_effects"] = me

    if do_permanova:
        out["permanova"] = subject_level_permanova(
            f, s, y, n_perm=n_perm, seed=seed
        )

    out["label_subspace"] = label_subspace_analysis(f, y)

    return out


def analyze_dataset(
    frozen: tuple[np.ndarray, np.ndarray, np.ndarray],
    ft_per_fold: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    n_boot: int = 1000,
    n_perm: int = 999,
    mixed_effects_max_dims: int | None = None,
    do_permanova: bool = True,
    do_mixed_effects: bool = True,
    seed: int = 0,
) -> dict:
    """Frozen + per-fold FT analysis with reviewer-defensible math.

    Frozen path: one analyze_regime call on the whole frozen feature matrix.

    FT path: TWO complementary views, both reported, so reviewers can pick:
      - `ft_pooled`: concatenate all 5 folds and analyze as one matrix
        (full N, but mixes 5 different fine-tuned models — methods caveat).
      - `ft_per_fold` + `ft_aggregated`: per-fold analysis, aggregated to
        mean ± percentile across folds. Clean subject independence but
        small N per fold (especially for Stress: 9-20 records / fold).

    Returns dict with 'frozen', 'ft_pooled', 'ft_per_fold' (list),
    'ft_aggregated'.
    """
    out = {"frozen": analyze_regime(
        *frozen,
        n_boot=n_boot, n_perm=n_perm,
        mixed_effects_max_dims=mixed_effects_max_dims,
        do_permanova=do_permanova,
        do_mixed_effects=do_mixed_effects,
        seed=seed,
    )}

    # Pooled FT: concat all folds, run the same analysis as frozen.
    pooled_f = np.concatenate([t[0] for t in ft_per_fold])
    pooled_s = np.concatenate([t[1] for t in ft_per_fold])
    pooled_y = np.concatenate([t[2] for t in ft_per_fold])
    out["ft_pooled"] = analyze_regime(
        pooled_f, pooled_s, pooled_y,
        n_boot=n_boot, n_perm=n_perm,
        mixed_effects_max_dims=mixed_effects_max_dims,
        do_permanova=do_permanova,
        do_mixed_effects=do_mixed_effects,
        seed=seed + 50,
    )

    fold_results = []
    for k, (f, s, y) in enumerate(ft_per_fold):
        fold_results.append(analyze_regime(
            f, s, y,
            n_boot=n_boot, n_perm=n_perm,
            mixed_effects_max_dims=mixed_effects_max_dims,
            do_permanova=do_permanova,
            do_mixed_effects=do_mixed_effects,
            seed=seed + 100 + k,
        ))
    out["ft_per_fold"] = fold_results

    # Aggregate per-fold scalars to mean + percentile.
    def _agg(values: list[float]) -> dict:
        arr = np.array([v for v in values if v is not None and np.isfinite(v)])
        if len(arr) == 0:
            return {"mean": float("nan"), "ci_low": float("nan"),
                    "ci_high": float("nan"), "n_folds": 0}
        return {
            "mean": float(arr.mean()),
            "ci_low": float(np.percentile(arr, 2.5)) if len(arr) >= 2 else float(arr.min()),
            "ci_high": float(np.percentile(arr, 97.5)) if len(arr) >= 2 else float(arr.max()),
            "n_folds": int(len(arr)),
        }

    omega_subj = [r["nested_omega2"]["subject_within_label"] for r in fold_results]
    omega_lab = [r["nested_omega2"]["label"] for r in fold_results]
    ratio = [r["nested_omega2"]["ratio_subject_to_label"] for r in fold_results]

    aggregated = {
        "omega2_subject_within_label": _agg(omega_subj),
        "omega2_label": _agg(omega_lab),
        "ratio_subject_to_label": _agg(ratio),
    }

    if do_mixed_effects:
        icc_vals = [
            r["mixed_effects"].get("icc_subject_mean")
            for r in fold_results
            if "error" not in r["mixed_effects"]
        ]
        frac_lab = [
            r["mixed_effects"].get("frac_label_mean")
            for r in fold_results
            if "error" not in r["mixed_effects"]
        ]
        aggregated["icc_subject"] = _agg([v for v in icc_vals if v is not None])
        aggregated["frac_label"] = _agg([v for v in frac_lab if v is not None])

    if do_permanova:
        f_vals = [r["permanova"]["pseudo_F"] for r in fold_results]
        r2_vals = [r["permanova"]["R2"] for r in fold_results]
        p_vals = [r["permanova"]["p_value"] for r in fold_results]
        aggregated["permanova_pseudo_F"] = _agg(f_vals)
        aggregated["permanova_R2"] = _agg(r2_vals)
        aggregated["permanova_p_value"] = _agg(p_vals)

    out["ft_aggregated"] = aggregated
    return out
