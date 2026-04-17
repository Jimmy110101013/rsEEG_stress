"""Group-level cluster-permutation PSD anchor for EEGMAT (rest vs arithmetic).

Parallel to exp26 stress_group_level_psd_anchor.py — same protocol,
same MNE cluster-permutation F-test, 1000 perms, default F-threshold
(p=0.05). Tests whether a reproducible group-level spectral contrast
exists between rest (n=36) and arithmetic (n=36) recordings across the
19-channel x (0.5-45 Hz) log-Welch PSD tensor.

Expected outcome (under SDL's anchored-regime framing): strong cluster
survives p<0.05, localised to posterior alpha, consistent with
Klimesch-type alpha desynchronisation during cognitive load.

Outputs:
    results/studies/exp28_eegmat_psd_anchor/summary.json
    paper/figures/supplementary/fig_eegmat_psd_anchor.pdf
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

from pipeline.eegmat_dataset import EEGMATDataset  # noqa: E402

import mne  # noqa: E402
from mne.stats import permutation_cluster_test  # noqa: E402

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Config (parallel to Stress protocol)
# --------------------------------------------------------------------------- #
SFREQ = 200.0
WINDOW_SEC = 5.0
PSD_WINDOW_SEC = 2.0
PSD_OVERLAP = 0.5
FMIN, FMAX = 0.5, 45.0
N_PERM = 1000
CLUSTER_P = 0.05
SEED = 42

# EEGMAT 19-channel montage
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

OUT_DIR = ROOT / "results/studies/exp28_eegmat_psd_anchor"
FIG_PATH = ROOT / "paper/figures/supplementary/fig_eegmat_psd_anchor.pdf"
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

    ds = EEGMATDataset(
        data_root=str(ROOT / "data/eegmat"),
        target_sfreq=SFREQ,
        window_sec=WINDOW_SEC,
        norm="none",
        cache_dir=str(ROOT / "data/cache_eegmat_nnone"),
    )
    labels = ds.get_labels()
    n_rest = int((labels == 0).sum())
    n_task = int((labels == 1).sum())
    print(f"Cohort: {len(labels)} recordings | rest={n_rest} | task={n_task}")

    psd_list, freqs_ref = [], None
    for i in range(len(ds)):
        epochs, _lab, _n, _pid = ds[i]
        epochs_np = epochs.numpy().astype(np.float64)
        psd, freqs = compute_recording_psd(epochs_np)
        if freqs_ref is None:
            freqs_ref = freqs
        psd_list.append(psd)
    psd_tensor = np.stack(psd_list, axis=0)
    print(f"PSD tensor: {psd_tensor.shape}; freqs {freqs_ref[0]:.2f}-{freqs_ref[-1]:.2f} Hz ({len(freqs_ref)} bins)")

    psd_db = 10.0 * np.log10(psd_tensor + 1e-20)
    group_task = psd_db[labels == 1]
    group_rest = psd_db[labels == 0]

    print(f"Running permutation_cluster_test with n_permutations={N_PERM} ...")
    T_obs, clusters, cluster_pvals, H0 = permutation_cluster_test(
        [group_task, group_rest],
        n_permutations=N_PERM,
        threshold=None, tail=0, stat_fun=None, n_jobs=1,
        seed=SEED, out_type="mask", verbose=False,
    )
    print(f"Found {len(clusters)} candidate clusters; "
          f"{int(np.sum(cluster_pvals < CLUSTER_P))} significant at p<{CLUSTER_P}")

    cluster_reports = []
    mean_task = group_task.mean(axis=0)
    mean_rest = group_rest.mean(axis=0)
    diff = mean_task - mean_rest

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
        h_vals = avg_in_cluster(group_task)
        l_vals = avg_in_cluster(group_rest)
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
            "mean_power_task_db": float(h_vals.mean()),
            "mean_power_rest_db": float(l_vals.mean()),
            "significant_at_p05": bool(p < CLUSTER_P),
        })

    band_reports = {}
    for bname, (flo, fhi) in BANDS.items():
        fmask = (freqs_ref >= flo) & (freqs_ref < fhi)
        if not fmask.any():
            continue
        h_band = group_task[..., fmask].mean(axis=(1, 2))
        l_band = group_rest[..., fmask].mean(axis=(1, 2))
        n1, n2 = len(h_band), len(l_band)
        s1, s2 = h_band.var(ddof=1), l_band.var(ddof=1)
        s_pool = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        d = (h_band.mean() - l_band.mean()) / (s_pool + 1e-12)
        J = 1.0 - 3.0 / (4 * (n1 + n2) - 9)
        band_reports[bname] = {
            "freq_range_hz": [flo, fhi],
            "mean_task_db": float(h_band.mean()),
            "mean_rest_db": float(l_band.mean()),
            "hedges_g": float(J * d),
        }

    summary = {
        "analysis": "group_level_cluster_permutation_PSD",
        "dataset": f"EEGMAT ({len(labels)} recordings; task={n_task} / rest={n_rest})",
        "label_convention": "label=1 task (arithmetic) vs label=0 rest",
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
        "cohort": {"n_total": int(len(labels)), "n_task": n_task, "n_rest": n_rest},
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
        cbar.set_label("Task - Rest (dB)")
        fig.suptitle(
            f"EEGMAT group-level PSD (task n={n_task} vs rest n={n_rest})\n"
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
    print("\nBand descriptive (Hedges' g on mean whole-scalp power, Task - Rest):")
    for b, r in band_reports.items():
        print(f"  {b:>6s}: g={r['hedges_g']:+.3f}  "
              f"(task={r['mean_task_db']:.2f} dB, rest={r['mean_rest_db']:.2f} dB)")


if __name__ == "__main__":
    main()
