"""Holo-Hilbert Spectral Analysis (HHSA) pipeline for EEG.

Implements the data-adaptive cross-frequency representation of
Huang et al. (2016, Phil. Trans. R. Soc. A 374: 20150206) using the
`emd` package (Quinn et al. 2021).

Design decisions documented in docs/wsci_design.md.

Status: experimental, prototype phase. Not validated on real EEG yet.

Pipeline:
    EEG epoch (1D array, samples)
        ↓ CEEMDAN (Torres et al. 2011)
    Layer-1 IMFs (samples, n_imf1)
        ↓ Normalised Hilbert Transform
    {a_k(t), f_k(t)} per IMF
        ↓ CEEMDAN on each a_k
    Layer-2 AM-IMFs per carrier
        ↓ NHT
    {b_{k,j}(t), g_{k,j}(t)}
        ↓ Holospectrum binning
    H(omega_carrier, omega_AM)  [56 × 64 log-spaced grid]
"""
from __future__ import annotations

from dataclasses import dataclass

import emd
import numpy as np

# ----- HHSA parameters (locked per docs/wsci_design.md ratification) -----
ENSEMBLE_NOISE = 0.1          # σ relative to std(x); EEG-HHSA literature median
N_ENSEMBLES_L1 = 100          # CEEMDAN ensemble size, layer 1
N_ENSEMBLES_L2 = 24           # CEEMDAN ensemble size, layer 2 (AM)
MAX_IMF_L1 = 8                # Carrier IMF count cap
MAX_IMF_L2 = 5                # AM IMF count cap per carrier
SLOW_IMF_THRESHOLD_HZ = 2.0   # Skip layer-2 sift on IMFs with median IF < this
BOUNDARY_TRUST_S = 1.0        # Drop this many seconds from each end (Hilbert circular leakage)

# Holospectrum frequency grid (log-spaced, ratified)
CARRIER_EDGES = np.geomspace(0.5, 80.0, 57)   # 56 bins
AM_EDGES = np.geomspace(0.05, 25.0, 65)       # 64 bins


@dataclass
class HHSAResult:
    """Holospectrum + intermediate diagnostics for a single 1D signal."""
    holospectrum: np.ndarray         # (n_carrier_bins, n_am_bins)
    carrier_centers: np.ndarray      # (n_carrier_bins,)
    am_centers: np.ndarray           # (n_am_bins,)
    n_imf_l1: int                    # actual layer-1 IMF count
    n_imf_l1_used: int               # those passed to layer-2 (after slow-IMF filter)
    layer1_median_if: np.ndarray     # (n_imf_l1,) — median instantaneous freq per IMF
    skipped_imfs: list[int]          # indices of layer-1 IMFs not sifted at layer 2
    total_energy: float
    failure_reason: str | None = None  # populated when input fails validation


def _truncate_imfs(imfs: np.ndarray, max_imfs: int) -> np.ndarray:
    """`emd`'s max_imfs is a target, not a hard cap. Enforce it."""
    return np.ascontiguousarray(imfs[:, :max_imfs]) if imfs.shape[1] > max_imfs else imfs


def _empty_result(reason: str, carrier_edges: np.ndarray, am_edges: np.ndarray) -> "HHSAResult":
    """Return a zero holospectrum tagged with the failure reason."""
    fc = (carrier_edges[:-1] + carrier_edges[1:]) / 2
    fa = (am_edges[:-1] + am_edges[1:]) / 2
    return HHSAResult(
        holospectrum=np.zeros((len(fc), len(fa)), dtype=np.float64),
        carrier_centers=fc,
        am_centers=fa,
        n_imf_l1=0,
        n_imf_l1_used=0,
        layer1_median_if=np.array([]),
        skipped_imfs=[],
        total_energy=0.0,
        failure_reason=reason,
    )


def _safe_ceemdan(
    x: np.ndarray, nensembles: int, noise_seed: int
) -> np.ndarray | None:
    """Run CEEMDAN with fallback for emd 0.8.1 IndexError bug.

    emd's complete_ensemble_sift crashes when the signal decomposes into
    more layers than the noise template. Strategy: first try without
    max_imfs, then retry with decreasing max_imfs caps.
    """
    for max_imf in [None, 6, 5, 4]:
        try:
            kwargs = dict(
                ensemble_noise=ENSEMBLE_NOISE,
                nensembles=nensembles,
                noise_seed=noise_seed,
            )
            if max_imf is not None:
                kwargs["max_imfs"] = max_imf
            return emd.sift.complete_ensemble_sift(x, **kwargs)
        except (IndexError, ValueError):
            continue
    return None


def _validate_input(x: np.ndarray, sample_rate: float) -> str | None:
    """Return failure reason string, or None if input is OK to sift."""
    if x.size < 200:  # < 1s @ 200Hz: too short for meaningful HHSA
        return f"too_short_{x.size}_samples"
    if not np.all(np.isfinite(x)):
        return "contains_nan_or_inf"
    if x.std() < 1e-10:
        return "constant_or_zero_signal"
    return None


def _sanitize_nht(IF: np.ndarray, IA: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sanitize NHT outputs.

    Strategy: when IF is non-finite OR negative, force IA at that sample to 0.
    This excludes the sample from holospectrum energy accumulation regardless
    of which bin its (possibly garbage) IF would have indexed.
    """
    bad = ~np.isfinite(IF) | (IF < 0) | ~np.isfinite(IA)
    IF_clean = np.where(bad, 0.0, IF)
    IA_clean = np.where(bad, 0.0, IA)
    return IF_clean, IA_clean


def compute_holospectrum(
    x: np.ndarray,
    sample_rate: float,
    *,
    noise_seed: int = 0,
    carrier_edges: np.ndarray = CARRIER_EDGES,
    am_edges: np.ndarray = AM_EDGES,
    boundary_trust_s: float = BOUNDARY_TRUST_S,
) -> HHSAResult:
    """Compute the 2D holo-Hilbert spectrum for a single 1D signal.

    Parameters
    ----------
    x : ndarray, shape (n_samples,)
        Single-channel single-epoch EEG signal.
    sample_rate : float
        Sampling rate in Hz.
    noise_seed : int
        Master seed; layer-1 uses `noise_seed`, layer-2 IMF k uses `noise_seed + 1000 + k`.

    Returns
    -------
    HHSAResult
    """
    if x.ndim != 1:
        raise ValueError(f"compute_holospectrum expects 1D signal, got shape {x.shape}")

    fail = _validate_input(x, sample_rate)
    if fail is not None:
        return _empty_result(fail, carrier_edges, am_edges)

    # ----- Layer 1: CEEMDAN on signal -----
    # emd 0.8.1 bug: complete_ensemble_sift can IndexError internally when
    # the signal decomposes into more layers than the pre-computed noise
    # template. Fallback: retry with fewer max_imfs, then give up.
    imfs_l1 = _safe_ceemdan(x, N_ENSEMBLES_L1, noise_seed)
    if imfs_l1 is None:
        return _empty_result("ceemdan_layer1_failed", carrier_edges, am_edges)
    imfs_l1 = _truncate_imfs(imfs_l1, MAX_IMF_L1)
    n_samp, n_imf1 = imfs_l1.shape

    # Layer-1 NHT
    _, IF1, IA1 = emd.spectra.frequency_transform(imfs_l1, sample_rate, "nht")
    IF1, IA1 = _sanitize_nht(IF1, IA1)

    layer1_median_if = np.array([np.nanmedian(IF1[:, k]) for k in range(n_imf1)])

    # ----- Layer 2: CEEMDAN on each layer-1 amplitude -----
    IA2 = np.zeros((n_samp, n_imf1, MAX_IMF_L2))
    IF2 = np.zeros((n_samp, n_imf1, MAX_IMF_L2))
    skipped: list[int] = []
    for k in range(n_imf1):
        if layer1_median_if[k] < SLOW_IMF_THRESHOLD_HZ:
            skipped.append(k)
            continue
        if not np.all(np.isfinite(IA1[:, k])) or IA1[:, k].std() < 1e-10:
            skipped.append(k)
            continue
        try:
            am_imfs = _safe_ceemdan(IA1[:, k], N_ENSEMBLES_L2, noise_seed + 1000 + k)
            if am_imfs is None:
                skipped.append(k)
                continue
            am_imfs = _truncate_imfs(am_imfs, MAX_IMF_L2)
            n_used = am_imfs.shape[1]
            if not np.all(np.isfinite(am_imfs)):
                skipped.append(k)
                continue
            _, IF2_k, IA2_k = emd.spectra.frequency_transform(am_imfs, sample_rate, "nht")
            IF2_k, IA2_k = _sanitize_nht(IF2_k, IA2_k)
            IA2[:, k, :n_used] = IA2_k
            IF2[:, k, :n_used] = IF2_k
        except Exception:
            # Broad except is intentional in prototype: emd internal failures
            # should not abort a full recording. Log via skipped list for QC.
            skipped.append(k)
            continue

    # Boundary masking: zero out trust-window margins on layer-2 amplitudes
    # (NOT on layer-1 IA, which would create a discontinuity that the layer-2
    # sift would mistake for an extremum and contaminate the AM-IMFs).
    if boundary_trust_s > 0:
        n_drop = int(round(boundary_trust_s * sample_rate))
        IA2[:n_drop, :, :] = 0.0
        IA2[-n_drop:, :, :] = 0.0

    # ----- Holospectrum -----
    fc, fa, H = emd.spectra.holospectrum(
        IF1, IF2, IA2,
        edges=carrier_edges,
        edges2=am_edges,
        sample_rate=sample_rate,
    )

    return HHSAResult(
        holospectrum=H,
        carrier_centers=fc,
        am_centers=fa,
        n_imf_l1=n_imf1,
        n_imf_l1_used=n_imf1 - len(skipped),
        layer1_median_if=layer1_median_if,
        skipped_imfs=skipped,
        total_energy=float(H.sum()),
    )


def compute_holospectrum_recording(
    eeg: np.ndarray,
    sample_rate: float,
    *,
    base_seed: int = 0,
) -> np.ndarray:
    """Compute holospectra for all (channel, epoch) pairs in a recording.

    Parameters
    ----------
    eeg : ndarray, shape (n_epochs, n_channels, n_samples)
        EEG epochs.
    sample_rate : float
    base_seed : int
        Master seed; per-call seed is derived to be unique across (epoch,
        channel) pairs and across the inner layer-2 IMF index, with no
        collisions for n_chan ≤ 9_999 and MAX_IMF_L1 ≤ 9_999_999.

    Returns
    -------
    H_all : ndarray, shape (n_epochs, n_channels, n_carrier_bins, n_am_bins)
    """
    if eeg.ndim != 3:
        raise ValueError(f"expect (n_epochs, n_channels, n_samples), got {eeg.shape}")
    n_epoch, n_chan, _ = eeg.shape
    n_fc = len(CARRIER_EDGES) - 1
    n_fa = len(AM_EDGES) - 1
    H_all = np.zeros((n_epoch, n_chan, n_fc, n_fa), dtype=np.float32)
    # Seed scheme: outer = base_seed + (ei * n_chan + ci) * SEED_STRIDE
    # SEED_STRIDE must exceed (1000 + MAX_IMF_L1) used inside compute_holospectrum
    # for layer-2 perturbation, to avoid collisions with the next (ei, ci)'s seed range.
    SEED_STRIDE = 10_000
    for ei in range(n_epoch):
        for ci in range(n_chan):
            res = compute_holospectrum(
                eeg[ei, ci, :].astype(np.float64),
                sample_rate,
                noise_seed=base_seed + (ei * n_chan + ci) * SEED_STRIDE,
            )
            H_all[ei, ci, :, :] = res.holospectrum.astype(np.float32)
    return H_all


def compute_holospectrum_from_l1(
    IF_1d: np.ndarray,
    IA_1d: np.ndarray,
    sample_rate: float,
    *,
    noise_seed: int = 0,
    carrier_edges: np.ndarray = CARRIER_EDGES,
    am_edges: np.ndarray = AM_EDGES,
    boundary_trust_s: float = BOUNDARY_TRUST_S,
) -> HHSAResult:
    """Compute holospectrum from cached Layer-1 NHT output (skip Layer 1 CEEMDAN).

    Parameters
    ----------
    IF_1d : (n_samp, n_imf) — Layer 1 instantaneous frequency
    IA_1d : (n_samp, n_imf) — Layer 1 instantaneous amplitude
    """
    n_samp, n_imf1 = IF_1d.shape
    if n_samp < 200:
        return _empty_result("too_short", carrier_edges, am_edges)

    layer1_median_if = np.array([np.nanmedian(IF_1d[:, k]) for k in range(n_imf1)])

    # Layer 2: CEEMDAN on each carrier IA
    IA2 = np.zeros((n_samp, n_imf1, MAX_IMF_L2))
    IF2 = np.zeros((n_samp, n_imf1, MAX_IMF_L2))
    skipped: list[int] = []
    for k in range(n_imf1):
        if layer1_median_if[k] < SLOW_IMF_THRESHOLD_HZ:
            skipped.append(k)
            continue
        ak = IA_1d[:, k]
        if not np.all(np.isfinite(ak)) or ak.std() < 1e-10:
            skipped.append(k)
            continue
        try:
            am_imfs = _safe_ceemdan(ak, N_ENSEMBLES_L2, noise_seed + 1000 + k)
            if am_imfs is None:
                skipped.append(k)
                continue
            am_imfs = _truncate_imfs(am_imfs, MAX_IMF_L2)
            if not np.all(np.isfinite(am_imfs)):
                skipped.append(k)
                continue
            _, IF2_k, IA2_k = emd.spectra.frequency_transform(am_imfs, sample_rate, "nht")
            IF2_k, IA2_k = _sanitize_nht(IF2_k, IA2_k)
            n_used = am_imfs.shape[1]
            IA2[:, k, :n_used] = IA2_k
            IF2[:, k, :n_used] = IF2_k
        except Exception:
            skipped.append(k)
            continue

    if boundary_trust_s > 0:
        n_drop = int(round(boundary_trust_s * sample_rate))
        IA2[:n_drop, :, :] = 0.0
        IA2[-n_drop:, :, :] = 0.0

    fc, fa, H = emd.spectra.holospectrum(
        IF_1d, IF2, IA2,
        edges=carrier_edges, edges2=am_edges, sample_rate=sample_rate,
    )

    return HHSAResult(
        holospectrum=H,
        carrier_centers=fc, am_centers=fa,
        n_imf_l1=n_imf1, n_imf_l1_used=n_imf1 - len(skipped),
        layer1_median_if=layer1_median_if,
        skipped_imfs=skipped,
        total_energy=float(H.sum()),
    )


def aggregate_channels_geometric(H_chan_epoch: np.ndarray) -> np.ndarray:
    """Geometric-mean aggregate over channel axis.

    Parameters
    ----------
    H_chan_epoch : ndarray, shape (..., n_channels, n_fc, n_fa)
        Holospectrum array with channel as second-to-last axis.

    Returns
    -------
    H_agg : ndarray, shape (..., n_fc, n_fa)
    """
    eps = 1e-12
    return np.exp(np.mean(np.log(H_chan_epoch + eps), axis=-3))
