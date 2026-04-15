"""Group-level cluster-permutation PSD anchor for TDBRAIN (MDD vs HC).

Parallel to exp26 stress_group_level_psd_anchor.py — same protocol,
1000 perms, default F-threshold. Tests whether a reproducible
group-level spectral contrast exists between MDD and HC recordings
across the 19-channel x (0.5-45 Hz) log-Welch PSD tensor.

Expected outcome (under SDL's intermediate-regime framing): a moderate
cluster around fronto-central alpha asymmetry that may or may not
survive p<0.05 after correction — published MDD EEG literature reports
group-level asymmetry but no within-subject per-recording correlate.

Outputs:
    results/studies/exp29_tdbrain_psd_anchor/summary.json
    paper/figures/supplementary/fig_tdbrain_psd_anchor.pdf
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path("/raid/jupyter-linjimmy1003.md10/UCSD_stress")
sys.path.insert(0, str(ROOT))

from pipeline.tdbrain_dataset import TDBRAINDataset  # noqa: E402

import mne  # noqa: E402
from mne.stats import permutation_cluster_test  # noqa: E402

warnings.filterwarnings("ignore")

SFREQ = 200.0
WINDOW_SEC = 5.0
PSD_WINDOW_SEC = 2.0
PSD_OVERLAP = 0.5
FMIN, FMAX = 0.5, 45.0
N_PERM = 1000
CLUSTER_P = 0.05
SEED = 42

CH_NAMES = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
            'T3', 'C3', 'CZ', 'C4', 'T4',
            'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2']

BANDS = {
    "delta":  (0.5, 4.0),
    "theta":  (4.0, 8.0),
    "alpha":  (8.0, 13.0),
    "beta":   (13.0, 30.0),
    "gamma":  (30.0, 45.0),
}

OUT_DIR = ROOT / "results/studies/exp29_tdbrain_psd_anchor"
FIG_PATH = ROOT / "paper/figures/supplementary/fig_tdbrain_psd_anchor.pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)


def compute_recording_psd(epochs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nperseg = int(PSD_WINDOW_SEC * SFREQ)
    noverlap = int(nperseg * PSD_OVERLAP)
    freqs, psd = signal.welch(
        epochs, fs=SFREQ, nperseg=nperseg, noverlap=noverlap, axis=-1
    )
    psd_mean = psd.mean(axis=0)
    mask = (freqs >= FMIN) & (freqs <= FMAX)
    return psd_mean[:, mask], freqs[mask]


def main() -> None:
    np.random.seed(SEED)

    ds = TDBRAINDataset(
        data_root=str(ROOT / "data/tdbrain"),
        target_sfreq=SFREQ,
        window_sec=WINDOW_SEC,
        norm="none",
        condition="both",
        target_dx="MDD",
        cache_dir=str(ROOT / "data/cache_tdbrain_nnone"),
    )
    labels = ds.get_labels()
    n_mdd = int((labels == 1).sum())
    n_hc = int((labels == 0).sum())
    print(f"Cohort: {len(labels)} recordings | MDD={n_mdd} | HC={n_hc}")

    psd_list, freqs_ref = [], None
    for i in range(len(ds)):
        epochs, _lab, _n, _pid = ds[i]
        if epochs.shape[0] == 0:
            print(f"  skip idx={i} (no valid epochs)")
            continue
        epochs_np = epochs.numpy().astype(np.float64)
        psd, freqs = compute_recording_psd(epochs_np)
        if freqs_ref is None:
            freqs_ref = freqs
        psd_list.append(psd)
    psd_tensor = np.stack(psd_list, axis=0)
    # Re-align labels with what we actually loaded
    valid_idx = []
    for i in range(len(ds)):
        ep, _lab, _n, _pid = ds[i]
        if ep.shape[0] > 0:
            valid_idx.append(i)
    labels_valid = labels[valid_idx]
    n_mdd = int((labels_valid == 1).sum())
    n_hc = int((labels_valid == 0).sum())
    print(f"PSD tensor: {psd_tensor.shape}; freqs {freqs_ref[0]:.2f}-{freqs_ref[-1]:.2f} Hz ({len(freqs_ref)} bins)")
    print(f"Valid cohort after epoch check: MDD={n_mdd} | HC={n_hc}")

    psd_db = 10.0 * np.log10(psd_tensor + 1e-20)
    group_mdd = psd_db[labels_valid == 1]
    group_hc  = psd_db[labels_valid == 0]

    print(f"Running permutation_cluster_test with n_permutations={N_PERM} ...")
    T_obs, clusters, cluster_pvals, H0 = permutation_cluster_test(
        [group_mdd, group_hc],
        n_permutations=N_PERM,
        threshold=None, tail=0, stat_fun=None, n_jobs=1,
        seed=SEED, out_type="mask", verbose=False,
    )
    print(f"Found {len(clusters)} candidate clusters; "
          f"{int(np.sum(cluster_pvals < CLUSTER_P))} significant at p<{CLUSTER_P}")

    cluster_reports = []
    mean_mdd = group_mdd.mean(axis=0)
    mean_hc  = group_hc.mean(axis=0)
    diff     = mean_mdd - mean_hc

    for ci, (mask, p) in enumerate(zip(clusters, cluster_pvals)):
        n_cells = int(mask.sum())
        freq_any = mask.any(axis=0)
        ch_any = mask.any(axis=1)
        if freq_any.any():
            freq_idx = np.where(freq_any)[0]
            freq_range = [float(freqs_ref[freq_idx.min()]), float(freqs_ref[freq_idx.max()])]
        else:
            freq_range = [None, None]
        ch_indices = [int(i) for i in np.where(ch_any)[0]]
        ch_names_cluster = [CH_NAMES[i] for i in ch_indices]

        def avg_in_cluster(tensor):
            m = mask.astype(bool)
            return tensor[:, m].mean(axis=1) if m.any() else np.zeros(tensor.shape[0])
        h_vals = avg_in_cluster(group_mdd)
        l_vals = avg_in_cluster(group_hc)
        n1, n2 = len(h_vals), len(l_vals)
        s1, s2 = h_vals.var(ddof=1), l_vals.var(ddof=1)
        s_pool = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        cohens_d = (h_vals.mean() - l_vals.mean()) / (s_pool + 1e-12)
        J = 1.0 - 3.0 / (4 * (n1 + n2) - 9)
        hedges_g = J * cohens_d

        cluster_reports.append({
            "cluster_id": ci,
            "p_value": float(p),
            "n_cells": n_cells,
            "freq_range_hz": freq_range,
            "n_channels": len(ch_indices),
            "channels": ch_names_cluster,
            "hedges_g_on_mean_power": float(hedges_g),
            "mean_power_mdd_db": float(h_vals.mean()),
            "mean_power_hc_db":  float(l_vals.mean()),
            "significant_at_p05": bool(p < CLUSTER_P),
        })

    band_reports = {}
    for bname, (flo, fhi) in BANDS.items():
        fmask = (freqs_ref >= flo) & (freqs_ref < fhi)
        if not fmask.any():
            continue
        h_band = group_mdd[..., fmask].mean(axis=(1, 2))
        l_band = group_hc[...,  fmask].mean(axis=(1, 2))
        n1, n2 = len(h_band), len(l_band)
        s1, s2 = h_band.var(ddof=1), l_band.var(ddof=1)
        s_pool = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        d = (h_band.mean() - l_band.mean()) / (s_pool + 1e-12)
        J = 1.0 - 3.0 / (4 * (n1 + n2) - 9)
        band_reports[bname] = {
            "freq_range_hz": [flo, fhi],
            "mean_mdd_db": float(h_band.mean()),
            "mean_hc_db":  float(l_band.mean()),
            "hedges_g":    float(J * d),
        }

    summary = {
        "analysis": "group_level_cluster_permutation_PSD",
        "dataset": f"TDBRAIN MDD vs HC ({len(labels_valid)} recordings; MDD={n_mdd} / HC={n_hc})",
        "label_convention": "label=1 MDD vs label=0 HC",
        "welch": {
            "sfreq_hz": SFREQ, "nperseg_sec": PSD_WINDOW_SEC, "overlap": PSD_OVERLAP,
            "fmin_hz": FMIN, "fmax_hz": FMAX,
            "n_freq_bins": int(len(freqs_ref)), "df_hz": float(freqs_ref[1] - freqs_ref[0]),
        },
        "permutation_test": {
            "implementation": "mne.stats.permutation_cluster_test",
            "n_permutations": N_PERM, "threshold": "default (F at p=0.05)",
            "tail": 0, "adjacency": None, "seed": SEED,
        },
        "cohort": {"n_total": int(len(labels_valid)), "n_mdd": n_mdd, "n_hc": n_hc},
        "n_candidate_clusters": int(len(clusters)),
        "n_significant_clusters_p05": int(np.sum(cluster_pvals < CLUSTER_P)),
        "min_cluster_p": float(cluster_pvals.min()) if len(cluster_pvals) else None,
        "cluster_pvalues_sorted": sorted(float(p) for p in cluster_pvals),
        "clusters": cluster_reports,
        "band_descriptive": band_reports,
        "channels": CH_NAMES,
        "freq_axis_hz": [float(x) for x in freqs_ref],
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote: {OUT_DIR / 'summary.json'}")

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        montage = mne.channels.make_standard_montage("standard_1005")
        rename = {
            "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
            "FZ": "Fz", "CZ": "Cz", "PZ": "Pz",
            "FP1": "Fp1", "FP2": "Fp2",
        }
        ch_names_std = [rename.get(c, c) for c in CH_NAMES]
        info = mne.create_info(ch_names=ch_names_std, sfreq=SFREQ, ch_types="eeg")
        info.set_montage(montage, match_case=False, on_missing="warn")

        fig, axes = plt.subplots(1, len(BANDS), figsize=(3.0 * len(BANDS), 3.2))
        vmax = 0.0
        band_diffs = {}
        for bname, (flo, fhi) in BANDS.items():
            fmask = (freqs_ref >= flo) & (freqs_ref < fhi)
            d = diff[:, fmask].mean(axis=1)
            band_diffs[bname] = d
            vmax = max(vmax, np.max(np.abs(d)))

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
        cbar.set_label("MDD - HC (dB)")
        fig.suptitle(
            f"TDBRAIN group-level PSD (MDD n={n_mdd} vs HC n={n_hc})\n"
            f"Cluster permutation (N={N_PERM}): "
            f"{int(np.sum(cluster_pvals < CLUSTER_P))}/{len(clusters)} clusters p<{CLUSTER_P}; "
            f"min p = {float(cluster_pvals.min()) if len(cluster_pvals) else float('nan'):.3f}",
            fontsize=9,
        )
        fig.savefig(FIG_PATH, bbox_inches="tight", dpi=200)
        print(f"Wrote: {FIG_PATH}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Figure generation failed: {e}")

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
    print("\nBand descriptive (Hedges' g on mean whole-scalp power, MDD - HC):")
    for b, r in band_reports.items():
        print(f"  {b:>6s}: g={r['hedges_g']:+.3f}  "
              f"(mdd={r['mean_mdd_db']:.2f} dB, hc={r['mean_hc_db']:.2f} dB)")


if __name__ == "__main__":
    main()
