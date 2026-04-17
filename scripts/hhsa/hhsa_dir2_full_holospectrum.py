"""Direction 2: Full holospectrum cluster (no band restriction).

Instead of pre-selecting alpha carrier + slow AM, find where on the entire
2D holospectrum (carrier × AM) the two conditions consistently differ
across subjects.

Method: per-subject condition-difference holospectrum → one-sample Wilcoxon
signed-rank test per bin → cluster correction with max-stat permutation.

Output: results/hhsa/07_full_holospectrum/
"""
import os, sys, glob, time
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from scipy.ndimage import label as cc_label

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pipeline.hhsa as hhsa

hhsa.N_ENSEMBLES_L1 = 24
hhsa.N_ENSEMBLES_L2 = 12
N_JOBS = 64
FS = 200.0
OUT = "results/hhsa/07_full_holospectrum"
os.makedirs(OUT, exist_ok=True)


def compute_one_holo(x, fs, seed):
    res = hhsa.compute_holospectrum(x.astype(np.float64), fs, noise_seed=seed)
    return res.holospectrum.astype(np.float32)


def compute_mean_holospectrum(epochs, fs, ch_indices, base_seed, max_ep=12):
    """Channel-averaged mean holospectrum for a condition."""
    n_ep = min(epochs.shape[0], max_ep)
    n_ch = len(ch_indices)
    rng = np.random.default_rng(base_seed)
    idx = rng.choice(epochs.shape[0], n_ep, replace=False) if epochs.shape[0] > max_ep else np.arange(n_ep)

    STRIDE = 10_000
    tasks = [(epochs[ei, ci, :], fs, base_seed + (int(ei) * n_ch + j) * STRIDE)
             for ei in idx for j, ci in enumerate(ch_indices)]

    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(compute_one_holo)(x, fs, s) for x, fs, s in tasks
    )

    H_all = np.stack(results)  # (n_ep * n_ch, n_fc, n_fa)
    return H_all.mean(axis=0)  # (n_fc, n_fa)


def group_cluster_test(diff_maps, n_perm=1000, threshold_p=0.05, seed=0):
    """Cluster-corrected test on per-subject difference maps.

    For each bin: one-sample Wilcoxon signed-rank (H0: median=0).
    Cluster formation: bins with p < threshold_p.
    Cluster statistic: sum of -log10(p) in cluster.
    Null: sign-flip permutation of per-subject maps.
    """
    n_subj, n_fc, n_fa = diff_maps.shape
    rng = np.random.default_rng(seed)

    # Real p-values per bin
    p_map = np.ones((n_fc, n_fa))
    stat_map = np.zeros((n_fc, n_fa))
    for i in range(n_fc):
        for j in range(n_fa):
            vals = diff_maps[:, i, j]
            if np.std(vals) < 1e-12 or n_subj < 6:
                continue
            try:
                s, p = wilcoxon(vals, alternative='two-sided')
                p_map[i, j] = p
                stat_map[i, j] = -np.log10(p + 1e-15) * np.sign(np.median(vals))
            except ValueError:
                pass

    # Real clusters
    sig_mask = p_map < threshold_p
    labels_real, n_real = cc_label(sig_mask)
    real_masses = []
    for c in range(1, n_real + 1):
        mask = labels_real == c
        real_masses.append(np.abs(stat_map[mask]).sum())

    max_real = max(real_masses) if real_masses else 0

    # Null distribution: sign-flip permutation
    null_max = np.zeros(n_perm)
    for perm in range(n_perm):
        flips = rng.choice([-1, 1], size=n_subj)
        flipped = diff_maps * flips[:, None, None]
        p_perm = np.ones((n_fc, n_fa))
        stat_perm = np.zeros((n_fc, n_fa))
        for i in range(n_fc):
            for j in range(n_fa):
                vals = flipped[:, i, j]
                if np.std(vals) < 1e-12:
                    continue
                try:
                    _, p = wilcoxon(vals, alternative='two-sided')
                    p_perm[i, j] = p
                    stat_perm[i, j] = -np.log10(p + 1e-15)
                except ValueError:
                    pass
        perm_mask = p_perm < threshold_p
        perm_labels, n_perm_c = cc_label(perm_mask)
        if n_perm_c > 0:
            perm_masses = [np.abs(stat_perm[perm_labels == c]).sum() for c in range(1, n_perm_c + 1)]
            null_max[perm] = max(perm_masses)

    null_p95 = np.percentile(null_max, 95)
    surviving = [c for c, m in enumerate(real_masses, 1) if m > null_p95]

    return {
        "p_map": p_map,
        "stat_map": stat_map,
        "labels": labels_real,
        "n_clusters": n_real,
        "real_masses": real_masses,
        "null_p95": null_p95,
        "surviving": surviving,
    }


def load_stress_dss_splits():
    df = pd.read_csv("data/comprehensive_labels.csv")
    subjects = {}
    for pid in sorted(df["Patient_ID"].unique()):
        sub = df[df["Patient_ID"] == pid]
        if len(sub) < 4:
            continue
        med = sub["Stress_Score"].median()
        above = sub[sub["Stress_Score"] >= med]
        below = sub[sub["Stress_Score"] < med]
        if len(above) < 2 or len(below) < 2:
            continue
        pid_str = f"p{pid:02d}"
        above_ep, below_ep = [], []
        for _, row in above.iterrows():
            f = glob.glob(f"data/cache/*_{pid_str}_{int(row['Recording_ID'])}_w5.0.pt")
            if f:
                above_ep.append(torch.load(f[0], map_location="cpu", weights_only=True).numpy())
        for _, row in below.iterrows():
            f = glob.glob(f"data/cache/*_{pid_str}_{int(row['Recording_ID'])}_w5.0.pt")
            if f:
                below_ep.append(torch.load(f[0], map_location="cpu", weights_only=True).numpy())
        if above_ep and below_ep:
            subjects[pid_str] = {
                "above": np.concatenate(above_ep, axis=0),
                "below": np.concatenate(below_ep, axis=0),
            }
    return subjects


def main():
    fc = (hhsa.CARRIER_EDGES[:-1] + hhsa.CARRIER_EDGES[1:]) / 2
    fa = (hhsa.AM_EDGES[:-1] + hhsa.AM_EDGES[1:]) / 2
    n_fc, n_fa = len(fc), len(fa)
    t_start = time.time()

    # Use all channels
    eegmat_ch = list(range(19))
    stress_ch = list(range(30))

    # ===== EEGMAT =====
    print("EEGMAT: computing per-subject difference holospectra...")
    eeg_diffs = []
    for sid in range(36):
        sname = f"Subject{sid:02d}"
        rest = torch.load(f"data/cache_eegmat/eegmat_{sname}_1_w5.0_sr200.0.pt",
                           map_location="cpu", weights_only=True).numpy()
        arith = torch.load(f"data/cache_eegmat/eegmat_{sname}_2_w5.0_sr200.0.pt",
                            map_location="cpu", weights_only=True).numpy()
        H_rest = compute_mean_holospectrum(rest, FS, eegmat_ch, 900000 + sid * 500)
        H_arith = compute_mean_holospectrum(arith, FS, eegmat_ch, 910000 + sid * 500)
        eeg_diffs.append(H_rest - H_arith)  # positive = rest > arith
        if (sid + 1) % 12 == 0:
            print(f"  [{sid+1}/36] {time.time()-t_start:.0f}s")

    eeg_diffs = np.stack(eeg_diffs)  # (36, n_fc, n_fa)
    print(f"  EEGMAT done: {eeg_diffs.shape}")

    # ===== Stress =====
    print("\nStress-DSS: computing per-subject difference holospectra...")
    stress_subjects = load_stress_dss_splits()
    str_diffs = []
    for pid_str, data in stress_subjects.items():
        H_below = compute_mean_holospectrum(data["below"], FS, stress_ch, 920000 + int(pid_str[1:]) * 500)
        H_above = compute_mean_holospectrum(data["above"], FS, stress_ch, 930000 + int(pid_str[1:]) * 500)
        str_diffs.append(H_below - H_above)  # positive = below > above
        print(f"  {pid_str}: done")

    str_diffs = np.stack(str_diffs)  # (11, n_fc, n_fa)
    print(f"  Stress done: {str_diffs.shape}")

    # ===== Cluster test =====
    # Only run cluster test if n >= 6 (Wilcoxon needs it)
    print("\nRunning cluster permutation test (200 perms for speed)...")

    eeg_result = group_cluster_test(eeg_diffs, n_perm=200, seed=0)
    print(f"  EEGMAT: {eeg_result['n_clusters']} raw clusters, "
          f"null_p95={eeg_result['null_p95']:.1f}, "
          f"{len(eeg_result['surviving'])} surviving")

    if str_diffs.shape[0] >= 6:
        str_result = group_cluster_test(str_diffs, n_perm=200, seed=1)
        print(f"  Stress: {str_result['n_clusters']} raw clusters, "
              f"null_p95={str_result['null_p95']:.1f}, "
              f"{len(str_result['surviving'])} surviving")
    else:
        print(f"  Stress: n={str_diffs.shape[0]} < 6, skipping Wilcoxon (need n>=6)")
        str_result = None

    # ===== Plot =====
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Dir 2: Full Holospectrum Group Contrast (no band restriction)\n"
                 "All channels, all subjects", fontsize=13, y=1.01)

    # Row 1: EEGMAT
    # Mean difference
    mean_diff = eeg_diffs.mean(axis=0)
    vabs = max(abs(mean_diff.min()), abs(mean_diff.max()), 1e-6)
    im = axes[0, 0].pcolormesh(fc, fa, mean_diff.T, shading='nearest', cmap='RdBu_r',
                                norm=TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs))
    axes[0, 0].set_title("EEGMAT: mean(rest − arith)", fontsize=10)
    axes[0, 0].set_xlabel("Carrier (Hz)"); axes[0, 0].set_ylabel("AM (Hz)")
    axes[0, 0].set_xscale("log"); axes[0, 0].set_yscale("log")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

    # -log10(p) map
    im2 = axes[0, 1].pcolormesh(fc, fa, -np.log10(eeg_result["p_map"] + 1e-15).T,
                                 shading='nearest', cmap='hot', vmin=0, vmax=5)
    axes[0, 1].set_title("EEGMAT: −log₁₀(p) Wilcoxon", fontsize=10)
    axes[0, 1].set_xlabel("Carrier (Hz)"); axes[0, 1].set_ylabel("AM (Hz)")
    axes[0, 1].set_xscale("log"); axes[0, 1].set_yscale("log")
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    # Surviving clusters
    surv_mask = np.zeros_like(eeg_result["labels"], dtype=float)
    for c in eeg_result["surviving"]:
        surv_mask[eeg_result["labels"] == c] = np.sign(mean_diff[eeg_result["labels"] == c].mean())
    im3 = axes[0, 2].pcolormesh(fc, fa, surv_mask.T, shading='nearest', cmap='RdBu_r',
                                 vmin=-1, vmax=1)
    axes[0, 2].set_title(f"EEGMAT: {len(eeg_result['surviving'])} surviving clusters", fontsize=10)
    axes[0, 2].set_xlabel("Carrier (Hz)"); axes[0, 2].set_ylabel("AM (Hz)")
    axes[0, 2].set_xscale("log"); axes[0, 2].set_yscale("log")
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

    # Row 2: Stress
    mean_diff_s = str_diffs.mean(axis=0)
    vabs_s = max(abs(mean_diff_s.min()), abs(mean_diff_s.max()), 1e-6)
    im4 = axes[1, 0].pcolormesh(fc, fa, mean_diff_s.T, shading='nearest', cmap='RdBu_r',
                                 norm=TwoSlopeNorm(vmin=-vabs_s, vcenter=0, vmax=vabs_s))
    axes[1, 0].set_title("Stress: mean(below − above)", fontsize=10)
    axes[1, 0].set_xlabel("Carrier (Hz)"); axes[1, 0].set_ylabel("AM (Hz)")
    axes[1, 0].set_xscale("log"); axes[1, 0].set_yscale("log")
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

    if str_result is not None:
        im5 = axes[1, 1].pcolormesh(fc, fa, -np.log10(str_result["p_map"] + 1e-15).T,
                                     shading='nearest', cmap='hot', vmin=0, vmax=5)
        axes[1, 1].set_title("Stress: −log₁₀(p) Wilcoxon", fontsize=10)
        plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)

        surv_mask_s = np.zeros_like(str_result["labels"], dtype=float)
        for c in str_result["surviving"]:
            surv_mask_s[str_result["labels"] == c] = np.sign(mean_diff_s[str_result["labels"] == c].mean())
        im6 = axes[1, 2].pcolormesh(fc, fa, surv_mask_s.T, shading='nearest', cmap='RdBu_r',
                                     vmin=-1, vmax=1)
        axes[1, 2].set_title(f"Stress: {len(str_result['surviving'])} surviving clusters", fontsize=10)
        plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
    else:
        axes[1, 1].text(0.5, 0.5, "n < 6\nskipped", transform=axes[1, 1].transAxes, ha='center', fontsize=14)
        axes[1, 2].text(0.5, 0.5, "n < 6\nskipped", transform=axes[1, 2].transAxes, ha='center', fontsize=14)

    for ax in axes[1, 1:]:
        ax.set_xlabel("Carrier (Hz)"); ax.set_ylabel("AM (Hz)")
        ax.set_xscale("log"); ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(f"{OUT}/full_holospectrum_cluster.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT}/full_holospectrum_cluster.png")

    np.savez(f"{OUT}/full_holospectrum_data.npz",
             eeg_diffs=eeg_diffs, str_diffs=str_diffs,
             carrier_centers=fc, am_centers=fa)
    print(f"Total time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
