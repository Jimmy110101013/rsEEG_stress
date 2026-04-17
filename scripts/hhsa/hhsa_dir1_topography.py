"""Direction 1: Full-brain AM topography.

Per-channel holospectrum → per-channel alpha AM contrast (log₂ ratio).
Produces a bar chart (not true topographic map since we don't have
electrode coordinates loaded) showing which channels drive the contrast.

Goal: discover if Stress has a spatial pattern we missed with posterior ROI.

Output: results/hhsa/04_topography/
"""
import os, sys, glob, time
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pipeline.hhsa as hhsa
import emd

hhsa.N_ENSEMBLES_L1 = 24
hhsa.N_ENSEMBLES_L2 = 12
N_JOBS = 64
FS = 200.0
OUT = "results/hhsa/04_topography"
os.makedirs(OUT, exist_ok=True)

EEGMAT_CH = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4", "T5", "T6",
             "C3", "C4", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz"]
STRESS_CH = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3', 'FCZ',
             'FC4', 'FT8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'TP7', 'CP3', 'CPZ',
             'CP4', 'TP8', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'OZ', 'O2']


def compute_one_holo(x, fs, seed):
    res = hhsa.compute_holospectrum(x.astype(np.float64), fs, noise_seed=seed)
    return res.holospectrum.astype(np.float32)


def per_channel_alpha_am(epochs, fs, base_seed, max_ep=10):
    """Compute per-channel alpha-band holospectral energy.

    Returns array of shape (n_channels,) with mean alpha holospectral energy.
    """
    n_ep, n_ch, _ = epochs.shape
    n_ep = min(n_ep, max_ep)
    rng = np.random.default_rng(base_seed)
    idx = rng.choice(epochs.shape[0], n_ep, replace=False) if epochs.shape[0] > max_ep else np.arange(n_ep)

    STRIDE = 10_000
    tasks = [(epochs[ei, ci, :], fs, base_seed + (int(ei) * n_ch + ci) * STRIDE)
             for ei in idx for ci in range(n_ch)]

    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(compute_one_holo)(x, fs, s) for x, fs, s in tasks
    )

    n_fc, n_fa = results[0].shape
    fc = (hhsa.CARRIER_EDGES[:-1] + hhsa.CARRIER_EDGES[1:]) / 2
    fa = (hhsa.AM_EDGES[:-1] + hhsa.AM_EDGES[1:]) / 2
    alpha_mask = (fc >= 8) & (fc <= 13)
    slow_am = (fa >= 0.2) & (fa <= 2.0)

    # Reshape results into (n_ep, n_ch, n_fc, n_fa)
    H = np.stack(results).reshape(n_ep, n_ch, n_fc, n_fa)

    # Per-channel: mean alpha energy, mean alpha slow-AM energy, total energy
    ch_alpha = np.zeros(n_ch)
    ch_alpha_am = np.zeros(n_ch)
    ch_total = np.zeros(n_ch)
    for ci in range(n_ch):
        H_ch = H[:, ci, :, :].mean(axis=0)  # mean over epochs
        ch_alpha[ci] = H_ch[alpha_mask, :].sum()
        ch_alpha_am[ci] = H_ch[alpha_mask, :][:, slow_am].sum()
        ch_total[ci] = H_ch.sum()

    return ch_alpha, ch_alpha_am, ch_total


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
    t_start = time.time()

    # ===== EEGMAT =====
    print("EEGMAT: per-channel topography...")
    eeg_topo = []
    for sid in range(36):
        sname = f"Subject{sid:02d}"
        rest = torch.load(f"data/cache_eegmat/eegmat_{sname}_1_w5.0_sr200.0.pt",
                           map_location="cpu", weights_only=True).numpy()
        arith = torch.load(f"data/cache_eegmat/eegmat_{sname}_2_w5.0_sr200.0.pt",
                            map_location="cpu", weights_only=True).numpy()
        a0, am0, t0 = per_channel_alpha_am(rest, FS, 500000 + sid * 500)
        a1, am1, t1 = per_channel_alpha_am(arith, FS, 600000 + sid * 500)
        # Log ratio per channel
        lr_alpha = np.log2((a0 + 1e-12) / (a1 + 1e-12))
        lr_am = np.log2((am0 + 1e-12) / (am1 + 1e-12))
        eeg_topo.append({"sid": sname, "lr_alpha": lr_alpha, "lr_am": lr_am})
        if (sid + 1) % 6 == 0:
            print(f"  [{sid+1}/36] {time.time()-t_start:.0f}s")

    # ===== Stress =====
    print("\nStress-DSS: per-channel topography...")
    stress_subjects = load_stress_dss_splits()
    str_topo = []
    for pid_str, data in stress_subjects.items():
        a0, am0, t0 = per_channel_alpha_am(data["below"], FS, 700000 + int(pid_str[1:]) * 500)
        a1, am1, t1 = per_channel_alpha_am(data["above"], FS, 800000 + int(pid_str[1:]) * 500)
        lr_alpha = np.log2((a0 + 1e-12) / (a1 + 1e-12))
        lr_am = np.log2((am0 + 1e-12) / (am1 + 1e-12))
        str_topo.append({"sid": pid_str, "lr_alpha": lr_alpha, "lr_am": lr_am})
        print(f"  {pid_str}: done")

    print(f"\nTotal time: {time.time()-t_start:.0f}s")

    # ===== Plot: per-channel alpha contrast =====
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Per-Channel Alpha Contrast Topography (log₂ ratio, all subjects mean ± std)",
                 fontsize=13, y=1.01)

    # EEGMAT alpha power
    eeg_lr_alpha = np.stack([d["lr_alpha"] for d in eeg_topo])  # (36, 19)
    ax = axes[0, 0]
    mean_a = eeg_lr_alpha.mean(axis=0)
    std_a = eeg_lr_alpha.std(axis=0)
    colors = ['salmon' if v < 0 else 'steelblue' for v in mean_a]
    ax.bar(range(len(EEGMAT_CH)), mean_a, yerr=std_a, color=colors, alpha=0.7, capsize=3)
    ax.set_xticks(range(len(EEGMAT_CH)))
    ax.set_xticklabels(EEGMAT_CH, rotation=45, fontsize=8)
    ax.axhline(0, color='gray', ls='--', alpha=0.4)
    ax.set_ylabel("log₂(rest/arith)")
    ax.set_title("EEGMAT: Alpha Power per channel", fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    # EEGMAT alpha AM
    eeg_lr_am = np.stack([d["lr_am"] for d in eeg_topo])
    ax = axes[0, 1]
    mean_am = eeg_lr_am.mean(axis=0)
    std_am = eeg_lr_am.std(axis=0)
    colors = ['salmon' if v < 0 else 'steelblue' for v in mean_am]
    ax.bar(range(len(EEGMAT_CH)), mean_am, yerr=std_am, color=colors, alpha=0.7, capsize=3)
    ax.set_xticks(range(len(EEGMAT_CH)))
    ax.set_xticklabels(EEGMAT_CH, rotation=45, fontsize=8)
    ax.axhline(0, color='gray', ls='--', alpha=0.4)
    ax.set_ylabel("log₂(rest/arith)")
    ax.set_title("EEGMAT: Alpha Slow-AM per channel", fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    # Stress alpha power
    str_lr_alpha = np.stack([d["lr_alpha"] for d in str_topo])  # (11, 30)
    ax = axes[1, 0]
    mean_a = str_lr_alpha.mean(axis=0)
    std_a = str_lr_alpha.std(axis=0)
    colors = ['salmon' if v < 0 else 'steelblue' for v in mean_a]
    ax.bar(range(len(STRESS_CH)), mean_a, yerr=std_a, color=colors, alpha=0.7, capsize=3)
    ax.set_xticks(range(len(STRESS_CH)))
    ax.set_xticklabels(STRESS_CH, rotation=45, fontsize=8)
    ax.axhline(0, color='gray', ls='--', alpha=0.4)
    ax.set_ylabel("log₂(below/above)")
    ax.set_title("Stress: Alpha Power per channel", fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    # Stress alpha AM
    str_lr_am = np.stack([d["lr_am"] for d in str_topo])
    ax = axes[1, 1]
    mean_am = str_lr_am.mean(axis=0)
    std_am = str_lr_am.std(axis=0)
    colors = ['salmon' if v < 0 else 'steelblue' for v in mean_am]
    ax.bar(range(len(STRESS_CH)), mean_am, yerr=std_am, color=colors, alpha=0.7, capsize=3)
    ax.set_xticks(range(len(STRESS_CH)))
    ax.set_xticklabels(STRESS_CH, rotation=45, fontsize=8)
    ax.axhline(0, color='gray', ls='--', alpha=0.4)
    ax.set_ylabel("log₂(below/above)")
    ax.set_title("Stress: Alpha Slow-AM per channel", fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT}/topography_alpha.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT}/topography_alpha.png")

    # Save data
    np.savez(f"{OUT}/topography_data.npz",
             eeg_lr_alpha=eeg_lr_alpha, eeg_lr_am=eeg_lr_am,
             str_lr_alpha=str_lr_alpha, str_lr_am=str_lr_am,
             eegmat_ch=EEGMAT_CH, stress_ch=STRESS_CH)


if __name__ == "__main__":
    main()
