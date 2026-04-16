"""WSCI — Within-Subject Contrast Index.

Statistical reduction from per-subject holospectra (one per epoch) to:
  (a) per-bin Hedges' g effect-size map for that subject's two-condition
      contrast,
  (b) cluster-mass permutation null on the 2D (carrier × AM) plane to
      identify surviving clusters,
  (c) per-subject WSCI scalar (signed surviving-cluster mass / total
      spectrum energy),
  (d) dataset-level summary (median + subject-bootstrap CI).

Design rationale: docs/wsci_design.md §6.

Notable deviation from design doc: the doc described Hedges' g_z (paired);
actual EEG data has unequal epoch counts per (subject, condition), so we
implement unpaired Hedges' g with pooled SD. Pairing at the *epoch* level
is not defined in our recordings — pairing only makes sense at subject-
level (one mean per condition per subject), but cluster permutation needs
within-subject epoch-level resampling, where unpaired g is the correct
statistic.

Status: experimental. Not yet validated on real EEG.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as cc_label


# Default parameters (per docs/wsci_design.md ratification)
CLUSTER_THRESHOLD = 0.5      # |g| threshold for cluster definition
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 1000
EPS = 1e-12


@dataclass
class SubjectWSCI:
    """Per-subject WSCI + diagnostics."""
    wsci: float                       # signed surviving cluster mass / total energy
    g_map: np.ndarray                 # (n_fc, n_fa) effect size
    surviving_cluster_mask: np.ndarray  # bool (n_fc, n_fa); True = bin in surviving cluster
    n_surviving_clusters: int
    null_max_mass_p95: float
    n_epochs_cond0: int
    n_epochs_cond1: int


@dataclass
class DatasetWSCI:
    """Dataset-level summary across subjects."""
    median: float
    ci_lower: float
    ci_upper: float
    per_subject: np.ndarray           # (n_subject,) WSCI values
    n_subjects: int


def hedges_g_unpaired(H0: np.ndarray, H1: np.ndarray) -> np.ndarray:
    """Unpaired Hedges' g per bin: (mean(H1) - mean(H0)) / SD_pool * J(N).

    Parameters
    ----------
    H0, H1 : ndarray, shape (n_epoch_c, n_fc, n_fa)
        Holospectra for condition 0 and 1, channel-aggregated.

    Returns
    -------
    g : ndarray, shape (n_fc, n_fa)
    """
    n0, n1 = H0.shape[0], H1.shape[0]
    if n0 < 2 or n1 < 2:
        # Need at least 2 epochs per condition for variance estimate
        return np.zeros(H0.shape[1:], dtype=np.float64)
    m0, m1 = H0.mean(axis=0), H1.mean(axis=0)
    v0 = H0.var(axis=0, ddof=1)
    v1 = H1.var(axis=0, ddof=1)
    sd_pool = np.sqrt(((n0 - 1) * v0 + (n1 - 1) * v1) / (n0 + n1 - 2))
    g_raw = np.where(sd_pool > EPS, (m1 - m0) / (sd_pool + EPS), 0.0)
    # Hedges small-sample correction
    N = n0 + n1
    J = 1.0 - 3.0 / (4.0 * N - 9.0) if N > 9 / 4 else 1.0
    return g_raw * J


def _cluster_mass(g_map: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Connected-components on |g| > threshold; returns (labels, per-label-mass).

    Mass per cluster = sum of |g| within the cluster.
    """
    over = np.abs(g_map) > threshold
    labels, n = cc_label(over)
    if n == 0:
        return labels, np.array([], dtype=np.float64)
    # bincount over labels (skipping background label 0)
    masses = np.bincount(labels.ravel(), weights=np.abs(g_map).ravel())[1:]
    return labels, masses


def cluster_permutation(
    H0: np.ndarray,
    H1: np.ndarray,
    *,
    n_perm: int = N_PERMUTATIONS,
    cluster_threshold: float = CLUSTER_THRESHOLD,
    seed: int = 0,
) -> dict:
    """2D cluster-mass permutation test.

    Pools epochs from both conditions, randomly relabels into n0/n1 splits,
    recomputes g, records max cluster mass per permutation.

    Returns
    -------
    dict with keys:
        g_map_real : (n_fc, n_fa)
        labels_real : (n_fc, n_fa) integer labels
        masses_real : (n_real_clusters,)
        null_max_dist : (n_perm,) max cluster mass per permutation
        surviving_cluster_idx : list[int] (1-indexed cluster labels)
    """
    n0, n1 = H0.shape[0], H1.shape[0]
    pooled = np.concatenate([H0, H1], axis=0)
    rng = np.random.default_rng(seed)

    g_real = hedges_g_unpaired(H0, H1)
    labels_real, masses_real = _cluster_mass(g_real, cluster_threshold)

    null_max = np.zeros(n_perm, dtype=np.float64)
    for p in range(n_perm):
        perm = rng.permutation(n0 + n1)
        H0_p = pooled[perm[:n0]]
        H1_p = pooled[perm[n0:]]
        g_p = hedges_g_unpaired(H0_p, H1_p)
        _, masses_p = _cluster_mass(g_p, cluster_threshold)
        null_max[p] = masses_p.max() if masses_p.size > 0 else 0.0

    null_p95 = np.percentile(null_max, 95)
    surviving = [
        i + 1 for i, m in enumerate(masses_real) if m > null_p95
    ]
    return {
        "g_map_real": g_real,
        "labels_real": labels_real,
        "masses_real": masses_real,
        "null_max_dist": null_max,
        "null_p95": float(null_p95),
        "surviving_cluster_idx": surviving,
    }


def wsci_subject(
    H_cond0: np.ndarray,
    H_cond1: np.ndarray,
    *,
    n_perm: int = N_PERMUTATIONS,
    cluster_threshold: float = CLUSTER_THRESHOLD,
    seed: int = 0,
) -> SubjectWSCI:
    """Compute WSCI for a single subject.

    Parameters
    ----------
    H_cond0, H_cond1 : ndarray, shape (n_epoch_c, n_fc, n_fa)
        Channel-aggregated holospectra for the two conditions.

    Returns
    -------
    SubjectWSCI
    """
    n_fc, n_fa = H_cond0.shape[1], H_cond0.shape[2]
    n0, n1 = H_cond0.shape[0], H_cond1.shape[0]

    if n0 < 2 or n1 < 2:
        return SubjectWSCI(
            wsci=0.0,
            g_map=np.zeros((n_fc, n_fa)),
            surviving_cluster_mask=np.zeros((n_fc, n_fa), dtype=bool),
            n_surviving_clusters=0,
            null_max_mass_p95=0.0,
            n_epochs_cond0=n0, n_epochs_cond1=n1,
        )

    out = cluster_permutation(
        H_cond0, H_cond1,
        n_perm=n_perm, cluster_threshold=cluster_threshold, seed=seed,
    )

    # Build surviving-cluster mask + signed mass
    surv_mask = np.zeros((n_fc, n_fa), dtype=bool)
    signed_mass = 0.0
    for cid in out["surviving_cluster_idx"]:
        bin_mask = (out["labels_real"] == cid)
        surv_mask |= bin_mask
        # Cluster sign: average sign of g within the cluster (typically all same sign)
        sign = np.sign(out["g_map_real"][bin_mask].mean())
        mass = np.abs(out["g_map_real"][bin_mask]).sum()
        signed_mass += sign * mass

    # Normalise by total energy in the spectrum (mean of two conditions)
    total_energy = (H_cond0.mean(axis=0).sum() + H_cond1.mean(axis=0).sum()) / 2.0
    wsci = signed_mass / (total_energy + EPS)

    return SubjectWSCI(
        wsci=float(wsci),
        g_map=out["g_map_real"],
        surviving_cluster_mask=surv_mask,
        n_surviving_clusters=len(out["surviving_cluster_idx"]),
        null_max_mass_p95=out["null_p95"],
        n_epochs_cond0=n0, n_epochs_cond1=n1,
    )


def wsci_dataset(
    wsci_per_subject: np.ndarray,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 0,
) -> DatasetWSCI:
    """Dataset-level summary: subject-median + bootstrap 95% CI.

    Parameters
    ----------
    wsci_per_subject : ndarray, shape (n_subject,)
        Per-subject WSCI values (NaN-free).

    Returns
    -------
    DatasetWSCI
    """
    w = np.asarray(wsci_per_subject, dtype=np.float64)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return DatasetWSCI(median=np.nan, ci_lower=np.nan, ci_upper=np.nan,
                           per_subject=w, n_subjects=0)

    rng = np.random.default_rng(seed)
    boot = np.array([
        np.median(rng.choice(w, size=w.size, replace=True))
        for _ in range(n_bootstrap)
    ])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return DatasetWSCI(
        median=float(np.median(w)),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        per_subject=w,
        n_subjects=int(w.size),
    )
