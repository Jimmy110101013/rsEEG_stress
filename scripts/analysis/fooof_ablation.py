"""FOOOF-based aperiodic/periodic ablation for EEG ablation study.

Per channel × per epoch:
  1. Welch PSD (1–45 Hz, nperseg=512)
  2. FOOOF fit (aperiodic mode='fixed'; peaks capped)
  3. Reconstruct three signal variants:
       - aperiodic-removed : X(f) / |aperiodic_amplitude(f)|
       - periodic-removed  : X(f) with peak-Gaussian power subtracted
       - both-removed      : apply both operations in sequence
  4. Save time-domain signals + per-epoch fit quality.

Inputs: cached epochs from pipeline/dataset.py (z-scored or raw per per-FM norm).
Outputs: npz with (aperiodic_removed, periodic_removed, both_removed) and quality JSON.

Usage:
    PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
    $PY scripts/analysis/fooof_ablation.py --dataset stress --out-suffix labram
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import signal as sig

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fooof import FOOOF  # noqa: E402

# Per-FM norm convention (keep consistent with extract_frozen_stress.py)
MODEL_NORM = {"labram": "zscore", "cbramod": "none", "reve": "none"}
# Per-FM canonical window (ADFTD refresh 2026-04-23: reve=10s, others=5s)
MODEL_WINDOW = {"labram": 5.0, "cbramod": 5.0, "reve": 10.0}

FS = 200.0  # Hz, target sample rate of cached epochs
FIT_RANGE = (1.0, 45.0)
WELCH_NPERSEG = 512
PEAK_WIDTH_LIMITS = (1.0, 12.0)
MAX_N_PEAKS = 6
MIN_PEAK_HEIGHT = 0.1


def fit_one_channel(x: np.ndarray, fs: float = FS):
    """Fit FOOOF on one channel × one epoch.

    Returns dict with (b, chi, peaks, R², fit_success) or None on failure.
    """
    freqs, psd = sig.welch(x, fs=fs, nperseg=WELCH_NPERSEG, noverlap=WELCH_NPERSEG // 2)
    mask = (freqs >= FIT_RANGE[0]) & (freqs <= FIT_RANGE[1])
    freqs_f = freqs[mask]
    psd_f = psd[mask]

    # Guard against non-positive PSD
    if np.any(psd_f <= 0) or not np.all(np.isfinite(psd_f)):
        return None

    fm = FOOOF(
        peak_width_limits=PEAK_WIDTH_LIMITS,
        max_n_peaks=MAX_N_PEAKS,
        min_peak_height=MIN_PEAK_HEIGHT,
        aperiodic_mode="fixed",
        verbose=False,
    )
    try:
        fm.fit(freqs_f, psd_f, FIT_RANGE)
    except Exception:
        return None

    if not fm.has_model:
        return None

    b = float(fm.aperiodic_params_[0])
    chi = float(fm.aperiodic_params_[1])
    peaks = fm.peak_params_.astype(np.float32) if fm.n_peaks_ > 0 else np.zeros((0, 3), np.float32)
    return {
        "b": b,
        "chi": chi,
        "peaks": peaks,  # rows of (center_freq, amp, bw)
        "r2": float(fm.r_squared_),
    }


def build_aperiodic_amp(n_fft: int, fs: float, b: float, chi: float) -> np.ndarray:
    """Return amplitude envelope matching FFT bin freqs for 1/f aperiodic model.

    FOOOF reports power as log10(P) = b - χ·log10(f). We convert to amplitude
    envelope (sqrt of linear power) to divide in frequency domain.

    Returns shape (n_fft // 2 + 1,) matching rfft output length.
    """
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    # Avoid freq=0 and freq < 1 Hz (FOOOF not fit there); use fit-range lower bound
    safe = np.maximum(freqs, FIT_RANGE[0])
    power_aper = 10.0 ** (b - chi * np.log10(safe))
    amp = np.sqrt(power_aper).astype(np.float32)
    return amp


def gaussian_power(freqs: np.ndarray, cf: float, amp: float, bw: float) -> np.ndarray:
    """Peak model: Gaussian in log power. Return linear power contribution.

    FOOOF peaks are parameterised on log10(P) surface. We reconstruct power
    difference of the form 10^(gaussian) - 1 on the log scale.
    """
    sigma = bw / 2.0
    log_peak = amp * np.exp(-((freqs - cf) ** 2) / (2.0 * sigma ** 2))
    return 10.0 ** log_peak - 1.0  # additive power above aperiodic baseline


def reconstruct(
    x: np.ndarray,
    fit: dict,
    fs: float = FS,
) -> dict:
    """Build three time-domain variants from a single channel × epoch signal."""
    X = np.fft.rfft(x)
    n_fft = len(x)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    amp_aper = build_aperiodic_amp(n_fft, fs, fit["b"], fit["chi"])
    amp_aper_safe = np.maximum(amp_aper, 1e-8)

    # aperiodic-removed: divide by aperiodic amplitude envelope (preserve phase)
    X_aper_removed = X / amp_aper_safe
    x_aper_removed = np.fft.irfft(X_aper_removed, n=n_fft).astype(np.float32)

    # periodic-removed: subtract gaussian-peak power contribution in power
    # domain, keep phase, reconstruct amplitude
    power_orig = np.abs(X) ** 2 / n_fft  # unnormalised power
    power_peaks = np.zeros_like(power_orig)
    for (cf, amp, bw) in fit["peaks"]:
        # amp on log scale; convert to linear power delta
        # Contribution is 10^(b - χ log f + gaussian) - 10^(b - χ log f)
        # = amp_aper_linear * (10^gaussian - 1)
        log_gauss = amp * np.exp(-((freqs - cf) ** 2) / (2.0 * (bw / 2.0) ** 2))
        delta = (amp_aper ** 2) * (10.0 ** log_gauss - 1.0)
        power_peaks = power_peaks + delta
    power_flat = np.maximum(power_orig - power_peaks, 1e-16)
    amp_flat = np.sqrt(power_flat * n_fft)
    # Preserve phase from original FFT
    phase = np.angle(X)
    X_per_removed = amp_flat * (np.cos(phase) + 1j * np.sin(phase))
    x_per_removed = np.fft.irfft(X_per_removed, n=n_fft).astype(np.float32)

    # both-removed: apply aperiodic division to periodic-removed
    X_both = X_per_removed / amp_aper_safe
    x_both = np.fft.irfft(X_both, n=n_fft).astype(np.float32)

    return {
        "aperiodic_removed": x_aper_removed,
        "periodic_removed": x_per_removed,
        "both_removed": x_both,
    }


def fit_channel_recording_level(all_epochs_one_ch: np.ndarray, fs: float = FS):
    """Fit FOOOF on PSD averaged across all epochs for one channel × one recording.

    Input: (M, T) epochs for one channel within one recording.
    Returns fit dict from fit_one_channel, using M-averaged PSD.
    """
    M, T = all_epochs_one_ch.shape
    psds = []
    for m in range(M):
        f_, p_ = sig.welch(all_epochs_one_ch[m].astype(np.float64),
                           fs=fs, nperseg=WELCH_NPERSEG,
                           noverlap=WELCH_NPERSEG // 2)
        psds.append(p_)
    psd_mean = np.mean(psds, axis=0)
    freqs = f_  # same for all epochs
    mask = (freqs >= FIT_RANGE[0]) & (freqs <= FIT_RANGE[1])
    freqs_f = freqs[mask]
    psd_f = psd_mean[mask]
    if np.any(psd_f <= 0) or not np.all(np.isfinite(psd_f)):
        return None
    fm = FOOOF(
        peak_width_limits=PEAK_WIDTH_LIMITS,
        max_n_peaks=MAX_N_PEAKS,
        min_peak_height=MIN_PEAK_HEIGHT,
        aperiodic_mode="fixed",
        verbose=False,
    )
    try:
        fm.fit(freqs_f, psd_f, FIT_RANGE)
    except Exception:
        return None
    if not fm.has_model:
        return None
    peaks = fm.peak_params_.astype(np.float32) if fm.n_peaks_ > 0 else np.zeros((0, 3), np.float32)
    return {
        "b": float(fm.aperiodic_params_[0]),
        "chi": float(fm.aperiodic_params_[1]),
        "peaks": peaks,
        "r2": float(fm.r_squared_),
    }


def process_epochs(epochs: np.ndarray) -> tuple[dict, dict]:
    """Apply FOOOF ablation to a (M, C, T) epoch tensor.

    Strategy (recording-level aperiodic, recording-level peaks):
      - For each channel, average PSD across all M epochs within this recording
        and fit one FOOOF model. Apply the same aperiodic/peaks ablation to
        every epoch of that channel in this recording.
      - Rationale: the 1/f aperiodic component is a physiologically trait-level
        property (skull thickness, E/I balance, montage) that should not vary
        across 5-s epochs of the same recording. Recording-level averaging
        produces smooth PSDs with higher FOOOF R² than per-epoch fits.

    Returns (signals_dict, quality_dict). Quality arrays are (C,) per channel
    for this recording (not per (M, C)).
    """
    M, C, T = epochs.shape
    ap = np.zeros_like(epochs, dtype=np.float32)
    pe = np.zeros_like(epochs, dtype=np.float32)
    bo = np.zeros_like(epochs, dtype=np.float32)

    r2 = np.full((C,), np.nan, dtype=np.float32)
    success = np.zeros((C,), dtype=bool)
    b_params = np.full((C,), np.nan, dtype=np.float32)
    chi_params = np.full((C,), np.nan, dtype=np.float32)
    n_peaks = np.zeros((C,), dtype=np.int16)

    for c in range(C):
        fit = fit_channel_recording_level(epochs[:, c, :])
        if fit is None:
            ap[:, c, :] = epochs[:, c, :].astype(np.float32)
            pe[:, c, :] = epochs[:, c, :].astype(np.float32)
            bo[:, c, :] = epochs[:, c, :].astype(np.float32)
            continue
        r2[c] = fit["r2"]
        success[c] = True
        b_params[c] = fit["b"]
        chi_params[c] = fit["chi"]
        n_peaks[c] = len(fit["peaks"])
        # Apply same aperiodic/peak params to every epoch of this channel
        for m in range(M):
            rec = reconstruct(epochs[m, c].astype(np.float64), fit)
            ap[m, c] = rec["aperiodic_removed"]
            pe[m, c] = rec["periodic_removed"]
            bo[m, c] = rec["both_removed"]

    signals = {
        "aperiodic_removed": ap,
        "periodic_removed": pe,
        "both_removed": bo,
    }
    quality = {
        "r2": r2,
        "fit_success": success,
        "b": b_params,
        "chi": chi_params,
        "n_peaks": n_peaks,
    }
    return signals, quality


# ------------------------------------------------------------------
# Dataset interface
# ------------------------------------------------------------------
def load_dataset_by_name(name: str, norm: str, window_sec: float = 5.0):
    """Load dataset for per-epoch processing. Returns (list of (epochs, meta), n_ch)."""
    from pipeline.dataset import StressEEGDataset
    if name == "stress":
        cache_dir = "data/cache" if norm == "zscore" else f"data/cache_n{norm}"
        ds = StressEEGDataset(
            "data/comprehensive_labels.csv", "data",
            target_sfreq=200.0, window_sec=window_sec, norm=norm,
            cache_dir=cache_dir,
        )
        return ds, 30, "stress"
    elif name == "eegmat":
        from pipeline.eegmat_dataset import EEGMATDataset
        cache_suffix = "" if norm == "zscore" else f"_n{norm}"
        ds = EEGMATDataset(
            "data/eegmat", target_sfreq=200.0, window_sec=window_sec,
            norm=norm, cache_dir=f"data/cache_eegmat{cache_suffix}",
        )
        return ds, 19, "other"
    elif name == "sleepdep":
        from pipeline.sleepdep_dataset import SleepDepDataset
        cache_suffix = "" if norm == "zscore" else f"_n{norm}"
        ds = SleepDepDataset(
            "data/sleep_deprivation", target_sfreq=200.0, window_sec=window_sec,
            norm=norm, cache_dir=f"data/cache_sleepdep{cache_suffix}",
        )
        # SleepDep returns 5-tuple (epochs, label, score, n_ep, pid) like Stress
        return ds, 19, "stress"
    elif name == "adftd":
        # ADFTD split1 binary (2026-04-23 refresh, G-F11/F12)
        from pipeline.adftd_dataset import ADFTDDataset
        cache_dir = "data/cache_adftd_split1" if norm == "zscore" else "data/cache_adftd_split1_nnone"
        ds = ADFTDDataset(
            "data/adftd", target_sfreq=200.0, window_sec=window_sec,
            norm=norm, binary=True,
            cache_dir=cache_dir, n_splits=1,
        )
        return ds, 19, "other"
    else:
        raise ValueError(f"Unknown dataset: {name}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["stress", "eegmat", "sleepdep", "adftd"])
    p.add_argument("--norm", default="zscore",
                   choices=["zscore", "none"],
                   help="Cached epochs norm; must match the FM norm used downstream")
    p.add_argument("--window-sec", type=float, default=5.0,
                   help="Epoch window seconds (ADFTD REVE uses 10; others 5)")
    p.add_argument("--pilot", type=int, default=0,
                   help="If >0, only process first N recordings (pilot mode)")
    p.add_argument("--out-dir", default="results/features_cache/fooof_ablation")
    args = p.parse_args()

    print(f"FOOOF ablation: dataset={args.dataset}, norm={args.norm}, window={args.window_sec}s, pilot={args.pilot}")
    ds, n_ch, ds_type = load_dataset_by_name(args.dataset, args.norm, args.window_sec)
    n_rec_total = len(ds)
    n_rec = args.pilot if args.pilot else n_rec_total
    print(f"  Processing {n_rec}/{n_rec_total} recordings × {n_ch} channels")

    rec_ap_list, rec_pe_list, rec_bo_list = [], [], []
    rec_rec_idx, rec_labels, rec_pids, rec_n_epochs = [], [], [], []
    all_quality_r2 = []
    all_quality_success = []
    all_b, all_chi, all_npk = [], [], []

    for i in range(n_rec):
        item = ds[i]
        if ds_type == "stress":
            epochs, label, _score, n_ep, pid = item
        else:
            epochs, label, n_ep, pid = item
        epochs_np = epochs.numpy().astype(np.float32)
        M = epochs_np.shape[0]
        print(f"  [{i+1}/{n_rec}] rec_pid={int(pid)}, label={int(label)}, M={M}")

        signals, quality = process_epochs(epochs_np)
        rec_ap_list.append(signals["aperiodic_removed"])
        rec_pe_list.append(signals["periodic_removed"])
        rec_bo_list.append(signals["both_removed"])
        rec_rec_idx.append(np.full(M, i, dtype=np.int32))
        rec_labels.append(int(label))
        rec_pids.append(int(pid))
        rec_n_epochs.append(M)
        all_quality_r2.append(quality["r2"])
        all_quality_success.append(quality["fit_success"])
        all_b.append(quality["b"])
        all_chi.append(quality["chi"])
        all_npk.append(quality["n_peaks"])

    # Stack: final arrays as concatenated (totalM, C, T)
    X_ap = np.concatenate(rec_ap_list, axis=0)
    X_pe = np.concatenate(rec_pe_list, axis=0)
    X_bo = np.concatenate(rec_bo_list, axis=0)
    rec_idx = np.concatenate(rec_rec_idx)
    # Quality arrays: one row per recording × C channels
    R2 = np.stack(all_quality_r2, axis=0)         # (n_rec, C)
    SUCC = np.stack(all_quality_success, axis=0)  # (n_rec, C)
    Bp = np.stack(all_b, axis=0)
    Chi = np.stack(all_chi, axis=0)
    Npk = np.stack(all_npk, axis=0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_pilot{args.pilot}" if args.pilot else ""
    # ADFTD uses per-FM window in the filename to disambiguate w5 vs w10 outputs;
    # other datasets retain the legacy single-file naming at w=5.
    if args.dataset == "adftd":
        win_tag = f"_w{int(args.window_sec)}"
    else:
        win_tag = ""
    out_path = out_dir / f"{args.dataset}_norm_{args.norm}{win_tag}{tag}.npz"
    np.savez_compressed(
        out_path,
        aperiodic_removed=X_ap,
        periodic_removed=X_pe,
        both_removed=X_bo,
        window_rec_idx=rec_idx,
        rec_labels=np.array(rec_labels, dtype=np.int32),
        rec_pids=np.array(rec_pids, dtype=np.int64),
        rec_n_epochs=np.array(rec_n_epochs, dtype=np.int32),
        quality_r2=R2,
        quality_success=SUCC,
        aperiodic_b=Bp,
        aperiodic_chi=Chi,
        n_peaks=Npk,
    )

    # Quality summary
    success_rate = float(SUCC.mean())
    r2_valid = R2[SUCC]
    summary = {
        "dataset": args.dataset,
        "norm": args.norm,
        "n_recordings": int(n_rec),
        "n_channels": int(n_ch),
        "total_channel_epochs": int(SUCC.size),
        "fit_success_rate": success_rate,
        "r2_mean": float(r2_valid.mean()) if len(r2_valid) else None,
        "r2_median": float(np.median(r2_valid)) if len(r2_valid) else None,
        "r2_p10": float(np.percentile(r2_valid, 10)) if len(r2_valid) else None,
        "r2_p90": float(np.percentile(r2_valid, 90)) if len(r2_valid) else None,
        "r2_ge_0.9_frac": float((r2_valid >= 0.9).mean()) if len(r2_valid) else None,
        "r2_ge_0.7_frac": float((r2_valid >= 0.7).mean()) if len(r2_valid) else None,
        "aperiodic_chi_mean": float(np.nanmean(Chi)),
        "aperiodic_chi_p10": float(np.nanpercentile(Chi, 10)),
        "aperiodic_chi_p90": float(np.nanpercentile(Chi, 90)),
        "mean_peaks_per_channel": float(Npk.mean()),
        "out_file": str(out_path),
    }
    summary_path = out_path.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Quality summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nSaved: {out_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
