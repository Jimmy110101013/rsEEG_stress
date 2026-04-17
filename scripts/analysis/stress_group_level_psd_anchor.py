"""Group-level cluster-permutation PSD anchor for UCSD Stress (pre-registered).

Tests whether a stable, group-level spectral contrast exists between
DASS-high (Group=='increase', n=14) and DASS-low (Group=='normal', n=56)
recordings across the 30-channel x (0.5-45 Hz) Welch PSD tensor, using
MNE's non-parametric cluster-level F test (1000 permutations).

Outputs:
    results/studies/exp26_stress_psd_anchor/summary.json
    paper/figures/supplementary/fig_stress_psd_anchor.pdf  (if clusters exist or mean topo requested)
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import signal

# repo root
ROOT = Path("/raid/jupyter-linjimmy1003.md10/UCSD_stress")
sys.path.insert(0, str(ROOT))

from pipeline.dataset import StressEEGDataset  # noqa: E402

import mne  # noqa: E402
from mne.stats import permutation_cluster_test  # noqa: E402

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

SFREQ = 200.0
WINDOW_SEC = 5.0
PSD_WINDOW_SEC = 2.0           # Welch segment length
PSD_OVERLAP = 0.5              # 50 %
FMIN, FMAX = 0.5, 45.0
N_PERM = 1000
CLUSTER_P = 0.05
SEED = 42

CH_NAMES = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ",
    "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ",
    "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2",
]

BANDS = {
    "delta":  (0.5, 4.0),
    "theta":  (4.0, 8.0),
    "alpha":  (8.0, 13.0),
    "beta":   (13.0, 30.0),
    "gamma":  (30.0, 45.0),
}

OUT_DIR = ROOT / "results/studies/exp26_stress_psd_anchor"
FIG_PATH = ROOT / "paper/figures/supplementary/fig_stress_psd_anchor.pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Load per-recording PSDs (channels x freqs, averaged over 5-s epochs)
# --------------------------------------------------------------------------- #

def compute_recording_psd(epochs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """epochs: (M, C, T) float, return (C, F) mean PSD and freqs vector."""
    # nperseg <= T; T = 5 s * 200 Hz = 1000, psd_window_sec = 2 s -> nperseg = 400
    nperseg = int(PSD_WINDOW_SEC * SFREQ)
    noverlap = int(nperseg * PSD_OVERLAP)
    freqs, psd = signal.welch(
        epochs, fs=SFREQ, nperseg=nperseg, noverlap=noverlap, axis=-1
    )
    # psd: (M, C, F).  Average over M epochs.
    psd_mean = psd.mean(axis=0)  # (C, F)
    # restrict to [FMIN, FMAX]
    mask = (freqs >= FMIN) & (freqs <= FMAX)
    return psd_mean[:, mask], freqs[mask]


def main() -> None:
    np.random.seed(SEED)

    csv_path = str(ROOT / "data/comprehensive_labels.csv")
    data_root = str(ROOT / "data")
    cache_dir = str(ROOT / "data/cache_nnone")

    ds = StressEEGDataset(
        csv_path=csv_path,
        data_root=data_root,
        target_sfreq=SFREQ,
        window_sec=WINDOW_SEC,
        stride_sec=None,
        norm="none",
        cache_dir=cache_dir,
    )
    labels = ds.get_labels()               # 1 = increase (DASS-high), 0 = normal (DASS-low)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    print(f"Cohort: {len(labels)} recordings | DASS-high={n_pos} | DASS-low={n_neg}")
    assert n_pos == 14 and n_neg == 56, "Cohort does not match 56/14 split"

    psd_list: list[np.ndarray] = []
    freqs_ref = None
    for i in range(len(ds)):
        epochs, _lab, _ss, _n, _pid = ds[i]
        epochs_np = epochs.numpy().astype(np.float64)    # (M, C, T) microvolts
        psd, freqs = compute_recording_psd(epochs_np)    # (C, F)
        if freqs_ref is None:
            freqs_ref = freqs
        psd_list.append(psd)
    psd_tensor = np.stack(psd_list, axis=0)              # (N, C, F)
    print(f"PSD tensor: shape={psd_tensor.shape} (N, C, F); freqs={freqs_ref[0]:.2f}-{freqs_ref[-1]:.2f} Hz, "
          f"{len(freqs_ref)} bins (df={freqs_ref[1]-freqs_ref[0]:.3f} Hz)")

    # log10 dB
    psd_db = 10.0 * np.log10(psd_tensor + 1e-20)

    group_high = psd_db[labels == 1]   # (14, C, F)
    group_low  = psd_db[labels == 0]   # (56, C, F)

    # MNE cluster permutation expects (n_obs, ...) per group. We pass (n, C, F).
    # adjacency=None -> treat all channel-freq cells as independent for clustering
    # along the feature axes (standard when no spatial adjacency provided).
    print(f"Running permutation_cluster_test with n_permutations={N_PERM} ...")
    T_obs, clusters, cluster_pvals, H0 = permutation_cluster_test(
        [group_high, group_low],
        n_permutations=N_PERM,
        threshold=None,           # default F-threshold at p=0.05
        tail=0,                   # two-tailed (positive F only, but test is symmetric via F)
        stat_fun=None,            # default F-test
        n_jobs=1,
        seed=SEED,
        out_type="mask",
        verbose=False,
    )
    print(f"Found {len(clusters)} candidate clusters; "
          f"{int(np.sum(cluster_pvals < CLUSTER_P))} significant at p<{CLUSTER_P}")

    # ------------------------------------------------------------------- #
    # Per-cluster summary
    # ------------------------------------------------------------------- #
    cluster_reports = []
    mean_high = group_high.mean(axis=0)   # (C, F)
    mean_low  = group_low.mean(axis=0)    # (C, F)
    diff      = mean_high - mean_low      # (C, F)

    for ci, (mask, p) in enumerate(zip(clusters, cluster_pvals)):
        # mask shape: (C, F) bool
        n_cells = int(mask.sum())
        # Frequency localization
        freq_any = mask.any(axis=0)
        ch_any = mask.any(axis=1)
        if freq_any.any():
            freq_idx = np.where(freq_any)[0]
            freq_range = [float(freqs_ref[freq_idx.min()]), float(freqs_ref[freq_idx.max()])]
        else:
            freq_range = [None, None]
        ch_indices = [int(i) for i in np.where(ch_any)[0]]
        ch_names_cluster = [CH_NAMES[i] for i in ch_indices]

        # Hedges' g on average power in cluster (avg over cluster cells per recording)
        def avg_in_cluster(tensor: np.ndarray) -> np.ndarray:
            # tensor: (n, C, F); returns (n,)
            m = mask.astype(bool)
            return tensor[:, m].mean(axis=1) if m.any() else np.zeros(tensor.shape[0])
        h_vals = avg_in_cluster(group_high)
        l_vals = avg_in_cluster(group_low)
        n1, n2 = len(h_vals), len(l_vals)
        s1, s2 = h_vals.var(ddof=1), l_vals.var(ddof=1)
        s_pool = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        cohens_d = (h_vals.mean() - l_vals.mean()) / (s_pool + 1e-12)
        # Hedges' g bias correction
        J = 1.0 - 3.0 / (4 * (n1 + n2) - 9)
        hedges_g = J * cohens_d

        cluster_reports.append(
            {
                "cluster_id": ci,
                "p_value": float(p),
                "n_cells": n_cells,
                "freq_range_hz": freq_range,
                "n_channels": len(ch_indices),
                "channels": ch_names_cluster,
                "hedges_g_on_mean_power": float(hedges_g),
                "mean_power_high_db": float(h_vals.mean()),
                "mean_power_low_db":  float(l_vals.mean()),
                "significant_at_p05": bool(p < CLUSTER_P),
            }
        )

    # ------------------------------------------------------------------- #
    # Band-wise descriptive contrasts (for paper text even if null)
    # ------------------------------------------------------------------- #
    band_reports = {}
    for bname, (flo, fhi) in BANDS.items():
        fmask = (freqs_ref >= flo) & (freqs_ref < fhi)
        if not fmask.any():
            continue
        h_band = group_high[..., fmask].mean(axis=(1, 2))  # (n_high,)
        l_band = group_low[...,  fmask].mean(axis=(1, 2))  # (n_low,)
        n1, n2 = len(h_band), len(l_band)
        s1, s2 = h_band.var(ddof=1), l_band.var(ddof=1)
        s_pool = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        d = (h_band.mean() - l_band.mean()) / (s_pool + 1e-12)
        J = 1.0 - 3.0 / (4 * (n1 + n2) - 9)
        band_reports[bname] = {
            "freq_range_hz": [flo, fhi],
            "mean_high_db": float(h_band.mean()),
            "mean_low_db":  float(l_band.mean()),
            "hedges_g":     float(J * d),
        }

    # ------------------------------------------------------------------- #
    # Save summary
    # ------------------------------------------------------------------- #
    summary = {
        "analysis": "group_level_cluster_permutation_PSD",
        "dataset": "UCSD Stress (70 recordings; DASS-high=14 / DASS-low=56)",
        "label_convention": "--label dass (Group=='increase' -> positive)",
        "welch": {
            "sfreq_hz": SFREQ,
            "nperseg_sec": PSD_WINDOW_SEC,
            "overlap": PSD_OVERLAP,
            "fmin_hz": FMIN,
            "fmax_hz": FMAX,
            "n_freq_bins": int(len(freqs_ref)),
            "df_hz": float(freqs_ref[1] - freqs_ref[0]),
        },
        "permutation_test": {
            "implementation": "mne.stats.permutation_cluster_test",
            "n_permutations": N_PERM,
            "threshold": "default (F at p=0.05)",
            "tail": 0,
            "adjacency": None,
            "seed": SEED,
        },
        "cohort": {"n_total": int(len(labels)), "n_high": n_pos, "n_low": n_neg},
        "n_candidate_clusters": int(len(clusters)),
        "n_significant_clusters_p05": int(np.sum(cluster_pvals < CLUSTER_P)),
        "min_cluster_p": float(cluster_pvals.min()) if len(cluster_pvals) else None,
        "cluster_pvalues_sorted": sorted(float(p) for p in cluster_pvals),
        "clusters": cluster_reports,
        "band_descriptive": band_reports,
        "channels": CH_NAMES,
        "freq_axis_hz": [float(x) for x in freqs_ref],
    }

    out_json = OUT_DIR / "summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote: {out_json}")

    # ------------------------------------------------------------------- #
    # Figure: topomap per band of group-mean dB difference + cluster outline
    # ------------------------------------------------------------------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Build minimal montage from standard_1005, matching our channel names
        montage = mne.channels.make_standard_montage("standard_1005")
        # Our names use 'T3/T4/T5/T6/FZ/CZ/PZ/OZ/FCZ/CPZ/FP1/FP2/FT7/FT8/TP7/TP8/FC3/FC4/CP3/CP4' style
        # standard_1005 uses 'T7/T8/P7/P8/Fz/Cz/Pz/Oz/FCz/CPz/Fp1/Fp2/...'
        rename = {
            "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
            "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "OZ": "Oz",
            "FCZ": "FCz", "CPZ": "CPz",
            "FP1": "Fp1", "FP2": "Fp2",
            "FT7": "FT7", "FT8": "FT8",
            "TP7": "TP7", "TP8": "TP8",
            "FC3": "FC3", "FC4": "FC4", "CP3": "CP3", "CP4": "CP4",
        }
        ch_names_std = [rename.get(c, c) for c in CH_NAMES]
        info = mne.create_info(ch_names=ch_names_std, sfreq=SFREQ, ch_types="eeg")
        info.set_montage(montage, match_case=False, on_missing="warn")

        fig, axes = plt.subplots(1, len(BANDS), figsize=(3.0 * len(BANDS), 3.2))
        vmax = 0.0
        band_diffs = {}
        for bname, (flo, fhi) in BANDS.items():
            fmask = (freqs_ref >= flo) & (freqs_ref < fhi)
            d = diff[:, fmask].mean(axis=1)                 # (C,)
            band_diffs[bname] = d
            vmax = max(vmax, np.max(np.abs(d)))

        # cluster channel mask (any frequency in any significant cluster)
        sig_ch_mask = np.zeros(len(CH_NAMES), dtype=bool)
        for mask, p in zip(clusters, cluster_pvals):
            if p < CLUSTER_P:
                sig_ch_mask |= mask.any(axis=1)

        for ax, (bname, (flo, fhi)) in zip(axes, BANDS.items()):
            d = band_diffs[bname]
            im, _ = mne.viz.plot_topomap(
                d, info, axes=ax, show=False,
                vlim=(-vmax, vmax), cmap="RdBu_r",
                mask=sig_ch_mask,
                mask_params=dict(marker="o", markerfacecolor="k",
                                 markeredgecolor="k", markersize=4),
            )
            ax.set_title(f"{bname}\n{flo:.1f}-{fhi:.1f} Hz", fontsize=9)
        cbar = fig.colorbar(im, ax=axes, shrink=0.7, location="right")
        cbar.set_label("High - Low (dB)")
        fig.suptitle(
            f"UCSD Stress group-level PSD (DASS-high n={n_pos} vs DASS-low n={n_neg})\n"
            f"Cluster permutation (N={N_PERM}): "
            f"{int(np.sum(cluster_pvals < CLUSTER_P))}/{len(clusters)} clusters p<{CLUSTER_P}; "
            f"min p = {float(cluster_pvals.min()) if len(cluster_pvals) else float('nan'):.3f}",
            fontsize=9,
        )
        fig.savefig(FIG_PATH, bbox_inches="tight", dpi=200)
        print(f"Wrote: {FIG_PATH}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Figure generation failed: {e}")

    # ------------------------------------------------------------------- #
    # Console summary
    # ------------------------------------------------------------------- #
    print("\n=== SUMMARY ===")
    print(f"candidate clusters: {len(clusters)}")
    print(f"significant (p<{CLUSTER_P}): {int(np.sum(cluster_pvals < CLUSTER_P))}")
    if len(cluster_pvals):
        print(f"min cluster p: {cluster_pvals.min():.4f}")
    for c in cluster_reports:
        if c["significant_at_p05"]:
            print(f"  * cluster {c['cluster_id']}: p={c['p_value']:.4f}, "
                  f"{c['freq_range_hz'][0]:.1f}-{c['freq_range_hz'][1]:.1f} Hz, "
                  f"{c['n_channels']} ch, g={c['hedges_g_on_mean_power']:.3f}")
    print("\nBand descriptive (Hedges' g on mean whole-scalp power, High - Low):")
    for b, r in band_reports.items():
        print(f"  {b:>6s}: g={r['hedges_g']:+.3f}  "
              f"(high={r['mean_high_db']:.2f} dB, low={r['mean_low_db']:.2f} dB)")


if __name__ == "__main__":
    main()
