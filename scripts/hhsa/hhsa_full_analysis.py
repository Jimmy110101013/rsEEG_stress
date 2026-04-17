"""Full HHSA analysis suite (dirs 04–08) for EEGMAT and Meditation.

Uses L1 cache for IMF-level analyses (topography, nonlinearity, energy)
and 60s holospectra for holospectrum visualization and AM coherence.

Outputs to results/hhsa/{04..08}_<name>/<dataset>/

Usage:
    python scripts/hhsa_full_analysis.py
"""
import os
import sys
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline.hhsa as hhsa

L1_CACHE = "results/hhsa/cache"
HOLO_DIR = "results/hhsa/holospectra"
FS = 200.0

# EEGMAT: _1 = rest (cond0), _2 = task (cond1)
# Meditation: ses-01 vs ses-02 (within-subject), expert vs novice (between-group)

COMMON_19_NAMES = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz',
                   'C4','T4','T5','P3','Pz','P4','T6','O1','O2']


def load_l1_cache(dataset):
    """Load all L1 cache files for a dataset. Returns list of dicts."""
    recs = []
    for f in sorted(glob.glob(f"{L1_CACHE}/{dataset}/*.npz")):
        d = np.load(f)
        recs.append({
            "path": f,
            "rec_id": os.path.basename(f).replace(".npz", ""),
            "IF": d["IF"],       # (n_ch, n_samp, n_imf)
            "IA": d["IA"],
            "ch_names": list(d["ch_names"]),
            "n_imf_per_ch": d["n_imf_per_ch"],
        })
    return recs


def load_holospectra(dataset):
    """Load all holospectra for a dataset. Returns list of dicts."""
    recs = []
    for f in sorted(glob.glob(f"{HOLO_DIR}/{dataset}/*.npz")):
        d = np.load(f)
        recs.append({
            "path": f,
            "rec_id": os.path.basename(f).replace(".npz", ""),
            "H_chan_agg": d["H_chan_agg"],  # (n_win, n_fc, n_fa)
            "H_windows": d["H_windows"],   # (n_win, n_ch, n_fc, n_fa)
            "ch_names": list(d["ch_names"]),
        })
    return recs


# =====================================================================
# Dir 04: Per-channel alpha-band AM topography
# =====================================================================
def dir04_topography(dataset, recs_l1, out_dir):
    """Per-channel alpha power and AM energy from L1 cache."""
    print(f"\n=== Dir 04: Topography ({dataset}) ===")
    os.makedirs(out_dir, exist_ok=True)

    alpha_lo, alpha_hi = 8.0, 16.0
    all_alpha_power = []  # (n_rec, n_ch)
    ch_names = recs_l1[0]["ch_names"]
    n_ch = len(ch_names)

    for rec in recs_l1:
        IF, IA = rec["IF"], rec["IA"]
        per_ch = []
        for ci in range(n_ch):
            # Find IMFs whose median IF falls in alpha band
            n_imf = int(rec["n_imf_per_ch"][ci])
            alpha_power = 0.0
            for k in range(n_imf):
                med_if = np.nanmedian(IF[ci, :, k])
                if alpha_lo <= med_if <= alpha_hi:
                    alpha_power += np.mean(IA[ci, :, k] ** 2)
            per_ch.append(alpha_power)
        all_alpha_power.append(per_ch)

    alpha_arr = np.array(all_alpha_power)  # (n_rec, n_ch)
    mean_alpha = alpha_arr.mean(axis=0)

    np.savez_compressed(os.path.join(out_dir, "topography_data.npz"),
                        alpha_power=alpha_arr, ch_names=np.array(ch_names),
                        mean_alpha=mean_alpha)

    # Bar plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(n_ch), mean_alpha)
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(ch_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean alpha-band IA² power")
    ax.set_title(f"Dir 04: Alpha Topography — {dataset}")
    fig.savefig(os.path.join(out_dir, "topography_alpha.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {out_dir}")


# =====================================================================
# Dir 05: IF nonlinearity (CV of instantaneous frequency per IMF)
# =====================================================================
def dir05_nonlinearity(dataset, recs_l1, out_dir):
    """CV of IF per IMF — higher CV = more nonlinear FM."""
    print(f"\n=== Dir 05: IF Nonlinearity ({dataset}) ===")
    os.makedirs(out_dir, exist_ok=True)

    max_imf = 8
    all_cv = []  # (n_rec, n_imf)

    for rec in recs_l1:
        IF = rec["IF"]
        n_ch = IF.shape[0]
        rec_cv = np.zeros(max_imf)
        for k in range(max_imf):
            cvs = []
            for ci in range(n_ch):
                if_k = IF[ci, :, k]
                valid = if_k[if_k > 0]
                if len(valid) > 10:
                    cvs.append(np.std(valid) / (np.mean(valid) + 1e-12))
            rec_cv[k] = np.mean(cvs) if cvs else 0.0
        all_cv.append(rec_cv)

    cv_arr = np.array(all_cv)
    mean_cv = cv_arr.mean(axis=0)

    np.savez_compressed(os.path.join(out_dir, "nonlinearity_data.npz"),
                        if_cv=cv_arr, mean_cv=mean_cv)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([cv_arr[:, k] for k in range(max_imf)], labels=[f"IMF{k}" for k in range(max_imf)])
    ax.set_ylabel("IF coefficient of variation")
    ax.set_title(f"Dir 05: IF Nonlinearity — {dataset}")
    fig.savefig(os.path.join(out_dir, "if_cv_per_imf.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {out_dir}")


# =====================================================================
# Dir 06: IMF energy distribution
# =====================================================================
def dir06_imf_energy(dataset, recs_l1, out_dir):
    """Fractional energy per IMF."""
    print(f"\n=== Dir 06: IMF Energy ({dataset}) ===")
    os.makedirs(out_dir, exist_ok=True)

    max_imf = 8
    all_frac = []

    for rec in recs_l1:
        IA = rec["IA"]
        n_ch = IA.shape[0]
        # Mean across channels
        energy_per_imf = np.zeros(max_imf)
        for k in range(max_imf):
            energy_per_imf[k] = np.mean([np.mean(IA[ci, :, k] ** 2) for ci in range(n_ch)])
        total = energy_per_imf.sum() + 1e-12
        all_frac.append(energy_per_imf / total)

    frac_arr = np.array(all_frac)
    mean_frac = frac_arr.mean(axis=0)

    np.savez_compressed(os.path.join(out_dir, "imf_energy_data.npz"),
                        energy_frac=frac_arr, mean_frac=mean_frac)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(max_imf), mean_frac, yerr=frac_arr.std(axis=0), capsize=3)
    ax.set_xticks(range(max_imf))
    ax.set_xticklabels([f"IMF{k}" for k in range(max_imf)])
    ax.set_ylabel("Fractional energy")
    ax.set_title(f"Dir 06: IMF Energy Distribution — {dataset}")
    fig.savefig(os.path.join(out_dir, "imf_energy_fraction.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {out_dir}")


# =====================================================================
# Dir 07: Holospectrum visualization (mean + condition contrast)
# =====================================================================
def dir07_holospectrum(dataset, recs_holo, condition_split, out_dir):
    """Mean holospectrum + condition contrast (t-stat map)."""
    print(f"\n=== Dir 07: Holospectrum Visualization ({dataset}) ===")
    os.makedirs(out_dir, exist_ok=True)

    # Grid
    fc = np.sqrt(hhsa.CARRIER_EDGES[:-1] * hhsa.CARRIER_EDGES[1:])
    fa = np.sqrt(hhsa.AM_EDGES[:-1] * hhsa.AM_EDGES[1:])

    # Collect per-recording mean holospectra
    all_H = {}  # rec_id → mean holospectrum
    for rec in recs_holo:
        all_H[rec["rec_id"]] = rec["H_chan_agg"].mean(axis=0)  # (n_fc, n_fa)

    # Grand mean
    grand_mean = np.mean(list(all_H.values()), axis=0)

    # Plot grand mean
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(fc, fa, np.log10(grand_mean.T + 1e-12),
                       cmap="hot", shading="auto")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Carrier frequency (Hz)")
    ax.set_ylabel("AM frequency (Hz)")
    ax.set_title(f"Mean Holospectrum — {dataset}")
    # Diagonal: AM < carrier
    diag = np.linspace(fa.min(), min(fc.max(), fa.max()), 100)
    ax.plot(diag, diag, "w--", lw=0.5, alpha=0.5)
    plt.colorbar(im, ax=ax, label="log₁₀ power")
    fig.savefig(os.path.join(out_dir, "mean_holospectrum.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Condition contrast if available
    if condition_split is not None:
        cond0_ids, cond1_ids, cond0_label, cond1_label = condition_split
        H0 = np.array([all_H[rid] for rid in cond0_ids if rid in all_H])
        H1 = np.array([all_H[rid] for rid in cond1_ids if rid in all_H])

        if H0.shape[0] >= 2 and H1.shape[0] >= 2:
            t_map, p_map = stats.ttest_ind(H1, H0, axis=0)
            t_map = np.nan_to_num(t_map)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            for ax, data, title in [(axes[0], np.log10(H0.mean(0).T + 1e-12), f"Mean {cond0_label}"),
                                     (axes[1], np.log10(H1.mean(0).T + 1e-12), f"Mean {cond1_label}")]:
                im = ax.pcolormesh(fc, fa, data, cmap="hot", shading="auto")
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_xlabel("Carrier freq (Hz)"); ax.set_ylabel("AM freq (Hz)")
                ax.set_title(title)
                plt.colorbar(im, ax=ax)
            fig.savefig(os.path.join(out_dir, "condition_holospectra.png"), dpi=150, bbox_inches="tight")
            plt.close()

            # t-stat map
            fig, ax = plt.subplots(figsize=(8, 6))
            vmax = np.percentile(np.abs(t_map), 95)
            im = ax.pcolormesh(fc, fa, t_map.T, cmap="RdBu_r", shading="auto",
                               vmin=-vmax, vmax=vmax)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("Carrier frequency (Hz)")
            ax.set_ylabel("AM frequency (Hz)")
            ax.set_title(f"t-stat: {cond1_label} vs {cond0_label} — {dataset}")
            ax.plot(diag, diag, "k--", lw=0.5, alpha=0.5)
            plt.colorbar(im, ax=ax, label="t-statistic")
            fig.savefig(os.path.join(out_dir, "condition_tstat.png"), dpi=150, bbox_inches="tight")
            plt.close()

            np.savez_compressed(os.path.join(out_dir, "condition_contrast.npz"),
                                t_map=t_map, p_map=p_map,
                                H0_mean=H0.mean(0), H1_mean=H1.mean(0),
                                n0=H0.shape[0], n1=H1.shape[0],
                                fc=fc, fa=fa)

    np.savez_compressed(os.path.join(out_dir, "grand_mean.npz"),
                        grand_mean=grand_mean, fc=fc, fa=fa)
    print(f"  Saved to {out_dir}")


# =====================================================================
# Dir 08: AM coherence (cross-frequency modulation consistency)
# =====================================================================
def dir08_am_coherence(dataset, recs_holo, condition_split, out_dir):
    """Per-subject AM coherence: how consistent is the holospectrum across windows."""
    print(f"\n=== Dir 08: AM Coherence ({dataset}) ===")
    os.makedirs(out_dir, exist_ok=True)

    fc = np.sqrt(hhsa.CARRIER_EDGES[:-1] * hhsa.CARRIER_EDGES[1:])
    fa = np.sqrt(hhsa.AM_EDGES[:-1] * hhsa.AM_EDGES[1:])

    # Per-recording: inter-window correlation of holospectrum
    coherences = []
    rec_ids = []
    for rec in recs_holo:
        H = rec["H_chan_agg"]  # (n_win, n_fc, n_fa)
        n_win = H.shape[0]
        if n_win < 2:
            continue
        # Flatten each window's holospectrum, compute mean pairwise correlation
        flat = H.reshape(n_win, -1)
        # Correlation matrix
        corr_mat = np.corrcoef(flat)
        # Mean off-diagonal
        mask = ~np.eye(n_win, dtype=bool)
        mean_corr = corr_mat[mask].mean()
        coherences.append(mean_corr)
        rec_ids.append(rec["rec_id"])

    coh_arr = np.array(coherences)
    np.savez_compressed(os.path.join(out_dir, "am_coherence.npz"),
                        coherence=coh_arr, rec_ids=np.array(rec_ids))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(coh_arr, bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(coh_arr.mean(), color="red", ls="--", label=f"mean={coh_arr.mean():.3f}")
    ax.set_xlabel("Inter-window holospectrum correlation")
    ax.set_ylabel("Count")
    ax.set_title(f"Dir 08: AM Coherence — {dataset}")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "am_coherence_hist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # If conditions available, compare coherence
    if condition_split is not None:
        cond0_ids, cond1_ids, cond0_label, cond1_label = condition_split
        c0 = [c for rid, c in zip(rec_ids, coh_arr) if rid in set(cond0_ids)]
        c1 = [c for rid, c in zip(rec_ids, coh_arr) if rid in set(cond1_ids)]
        if c0 and c1:
            t, p = stats.ttest_ind(c1, c0)
            print(f"  {cond0_label}: {np.mean(c0):.3f} ± {np.std(c0):.3f}")
            print(f"  {cond1_label}: {np.mean(c1):.3f} ± {np.std(c1):.3f}")
            print(f"  t={t:.2f}, p={p:.4f}")

    print(f"  Saved to {out_dir}")


# =====================================================================
# Condition split helpers
# =====================================================================
def eegmat_condition_split(recs):
    """EEGMAT: _1 = rest, _2 = task."""
    rest = [r["rec_id"] for r in recs if r["rec_id"].endswith("_1")]
    task = [r["rec_id"] for r in recs if r["rec_id"].endswith("_2")]
    return rest, task, "Rest", "Task"


def meditation_condition_split(recs):
    """Meditation: ses-01 vs ses-02."""
    ses1 = [r["rec_id"] for r in recs if "ses-01" in r["rec_id"]]
    ses2 = [r["rec_id"] for r in recs if "ses-02" in r["rec_id"]]
    return ses1, ses2, "Session 1", "Session 2"


def main():
    for dataset, split_fn in [("eegmat", eegmat_condition_split),
                               ("meditation", meditation_condition_split)]:
        print(f"\n{'#'*60}")
        print(f"# Dataset: {dataset}")
        print(f"{'#'*60}")

        recs_l1 = load_l1_cache(dataset)
        recs_holo = load_holospectra(dataset)
        print(f"  L1 cache: {len(recs_l1)} recordings")
        print(f"  Holospectra: {len(recs_holo)} recordings")

        cond_split_l1 = split_fn(recs_l1)
        cond_split_holo = split_fn(recs_holo)

        base = "results/hhsa"
        dir04_topography(dataset, recs_l1, f"{base}/04_topography/{dataset}")
        dir05_nonlinearity(dataset, recs_l1, f"{base}/05_nonlinearity/{dataset}")
        dir06_imf_energy(dataset, recs_l1, f"{base}/06_imf_energy/{dataset}")
        dir07_holospectrum(dataset, recs_holo, cond_split_holo, f"{base}/07_full_holospectrum/{dataset}")
        dir08_am_coherence(dataset, recs_holo, cond_split_holo, f"{base}/08_am_coherence/{dataset}")

    print(f"\n{'='*60}")
    print("All analyses complete.")


if __name__ == "__main__":
    main()
