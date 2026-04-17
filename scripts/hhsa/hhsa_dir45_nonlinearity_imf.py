"""Direction 4+5: IF nonlinearity + IMF energy distribution.

Dir 4 — IF variability (nonlinearity):
  For each IMF, std(IF) quantifies how much the instantaneous frequency
  fluctuates. High IF variability = non-stationary, nonlinear oscillation.
  PSD cannot capture this at all.

Dir 5 — IMF energy distribution (adaptive band power):
  Each IMF's energy fraction = IMF's share of total signal energy.
  Like traditional band power but with data-adaptive band boundaries.

Both use all channels, all subjects, Layer 1-2 only (no HHSA needed).

Output: results/hhsa/05_nonlinearity/, results/hhsa/06_imf_energy/
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
OUT_NL = "results/hhsa/05_nonlinearity"
OUT_IMF = "results/hhsa/06_imf_energy"
os.makedirs(OUT_NL, exist_ok=True)
os.makedirs(OUT_IMF, exist_ok=True)


def extract_hht_features(x, fs, seed):
    """Extract IF variability + IMF energy for a single channel-epoch."""
    x = x.astype(np.float64)
    if x.std() < 1e-10 or not np.all(np.isfinite(x)):
        return None
    imfs = hhsa._safe_ceemdan(x, hhsa.N_ENSEMBLES_L1, seed)
    if imfs is None:
        return None
    imfs = hhsa._truncate_imfs(imfs, hhsa.MAX_IMF_L1)
    _, IF, IA = emd.spectra.frequency_transform(imfs, fs, 'nht')
    IF, IA = hhsa._sanitize_nht(IF, IA)

    n_drop = int(1.0 * fs)
    IF = IF[n_drop:-n_drop, :]
    IA = IA[n_drop:-n_drop, :]
    imfs = imfs[n_drop:-n_drop, :]

    n_imf = IF.shape[1]
    # Dir 4: IF variability per IMF (std of IF, weighted by IA to focus on high-amplitude moments)
    if_std = np.array([np.std(IF[:, k][IA[:, k] > 0.01]) if (IA[:, k] > 0.01).sum() > 10 else 0.0
                       for k in range(n_imf)])
    if_cv = np.array([np.std(IF[:, k]) / (np.mean(IF[:, k]) + 1e-12) for k in range(n_imf)])
    median_if = np.array([np.median(IF[:, k]) for k in range(n_imf)])

    # Dir 5: IMF energy fractions
    imf_energy = np.array([np.sum(imfs[:, k] ** 2) for k in range(n_imf)])
    total_energy = imf_energy.sum() + 1e-12
    imf_frac = imf_energy / total_energy

    return {
        "if_std": if_std,        # (n_imf,)
        "if_cv": if_cv,          # (n_imf,)
        "median_if": median_if,  # (n_imf,)
        "imf_frac": imf_frac,    # (n_imf,)
        "n_imf": n_imf,
    }


def compute_subject_features(epochs, fs, base_seed, max_ep=15):
    """Compute mean HHT features across epochs and all channels."""
    n_ep, n_ch, _ = epochs.shape
    n_ep = min(n_ep, max_ep)
    rng = np.random.default_rng(base_seed)
    idx = rng.choice(epochs.shape[0], n_ep, replace=False) if epochs.shape[0] > max_ep else np.arange(n_ep)

    tasks = [(epochs[ei, ci, :], fs, base_seed + int(ei) * 1000 + ci)
             for ei in idx for ci in range(n_ch)]

    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(extract_hht_features)(x, fs, s) for x, fs, s in tasks
    )
    results = [r for r in results if r is not None]
    if not results:
        return None

    # Pad to same n_imf (take max across results, zero-pad shorter ones)
    max_nimf = max(r["n_imf"] for r in results)
    def pad(arr, target_len):
        if len(arr) >= target_len:
            return arr[:target_len]
        return np.concatenate([arr, np.zeros(target_len - len(arr))])

    if_stds = np.stack([pad(r["if_std"], max_nimf) for r in results])
    if_cvs = np.stack([pad(r["if_cv"], max_nimf) for r in results])
    median_ifs = np.stack([pad(r["median_if"], max_nimf) for r in results])
    imf_fracs = np.stack([pad(r["imf_frac"], max_nimf) for r in results])

    return {
        "if_std_mean": if_stds.mean(axis=0),
        "if_cv_mean": if_cvs.mean(axis=0),
        "median_if_mean": median_ifs.mean(axis=0),
        "imf_frac_mean": imf_fracs.mean(axis=0),
        # Summary scalars
        "total_if_cv": if_cvs.mean(),  # overall nonlinearity
        "n_imf_mean": np.mean([r["n_imf"] for r in results]),
    }


def load_stress_dss_splits():
    """Load Stress subjects with DSS personal-median split."""
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
    print("EEGMAT: extracting HHT features (all channels)...")
    eeg_data = []
    for sid in range(36):
        sname = f"Subject{sid:02d}"
        rest = torch.load(f"data/cache_eegmat/eegmat_{sname}_1_w5.0_sr200.0.pt",
                           map_location="cpu", weights_only=True).numpy()
        arith = torch.load(f"data/cache_eegmat/eegmat_{sname}_2_w5.0_sr200.0.pt",
                            map_location="cpu", weights_only=True).numpy()
        f_rest = compute_subject_features(rest, FS, 100000 + sid * 500)
        f_arith = compute_subject_features(arith, FS, 200000 + sid * 500)
        if f_rest and f_arith:
            eeg_data.append({"sid": sname, "cond0": f_rest, "cond1": f_arith})
        if (sid + 1) % 12 == 0:
            print(f"  [{sid+1}/36] {time.time()-t_start:.0f}s")

    # ===== Stress =====
    print("\nStress-DSS: extracting HHT features (all channels)...")
    stress_subjects = load_stress_dss_splits()
    str_data = []
    for pid_str, data in stress_subjects.items():
        f_below = compute_subject_features(data["below"], FS, 300000 + int(pid_str[1:]) * 500)
        f_above = compute_subject_features(data["above"], FS, 400000 + int(pid_str[1:]) * 500)
        if f_below and f_above:
            str_data.append({"sid": pid_str, "cond0": f_below, "cond1": f_above})
        print(f"  {pid_str}: done")

    print(f"\nTotal time: {time.time()-t_start:.0f}s")
    print(f"EEGMAT: {len(eeg_data)} subjects, Stress: {len(str_data)} subjects")

    # ===== Dir 4: Nonlinearity (IF CV) =====
    print("\n--- Dir 4: IF Nonlinearity ---")
    eeg_nl_lr = np.array([np.log2((d["cond0"]["total_if_cv"] + 1e-12) /
                                   (d["cond1"]["total_if_cv"] + 1e-12)) for d in eeg_data])
    str_nl_lr = np.array([np.log2((d["cond0"]["total_if_cv"] + 1e-12) /
                                   (d["cond1"]["total_if_cv"] + 1e-12)) for d in str_data])
    _, p_nl = mannwhitneyu(np.abs(eeg_nl_lr), np.abs(str_nl_lr), alternative='greater')
    print(f"  EEGMAT: med={np.median(eeg_nl_lr):.3f}, std={eeg_nl_lr.std():.3f}")
    print(f"  Stress: med={np.median(str_nl_lr):.3f}, std={str_nl_lr.std():.3f}")
    print(f"  MWU |EEGMAT|>|Stress|: p={p_nl:.4f}")

    # Per-IMF IF CV comparison
    max_imf = 6
    fig_nl, axes_nl = plt.subplots(1, 2, figsize=(14, 6))
    fig_nl.suptitle("Dir 4: IF Coefficient of Variation (Nonlinearity)\n"
                     "Higher = more non-stationary oscillation", fontsize=13)

    for ax, dataset, data_list, title in [
        (axes_nl[0], "EEGMAT", eeg_data, "EEGMAT: rest (blue) vs arith (red)"),
        (axes_nl[1], "Stress", str_data, "Stress: below (blue) vs above (red)"),
    ]:
        cv0 = np.stack([d["cond0"]["if_cv_mean"][:max_imf] for d in data_list])
        cv1 = np.stack([d["cond1"]["if_cv_mean"][:max_imf] for d in data_list])
        mif = np.stack([d["cond0"]["median_if_mean"][:max_imf] for d in data_list])

        x = np.arange(max_imf)
        ax.bar(x - 0.15, cv0.mean(axis=0), 0.3, yerr=cv0.std(axis=0), color='steelblue',
               alpha=0.7, label="Cond 0", capsize=3)
        ax.bar(x + 0.15, cv1.mean(axis=0), 0.3, yerr=cv1.std(axis=0), color='salmon',
               alpha=0.7, label="Cond 1", capsize=3)
        ax.set_xlabel("IMF index")
        ax.set_ylabel("IF Coefficient of Variation")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"IMF{i}\n({mif.mean(axis=0)[i]:.1f}Hz)" for i in range(max_imf)], fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)

    fig_nl.tight_layout()
    fig_nl.savefig(f"{OUT_NL}/if_cv_per_imf.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUT_NL}/if_cv_per_imf.png")

    # ===== Dir 5: IMF Energy Distribution =====
    print("\n--- Dir 5: IMF Energy Distribution ---")

    fig_imf, axes_imf = plt.subplots(1, 2, figsize=(14, 6))
    fig_imf.suptitle("Dir 5: IMF Energy Fraction (Adaptive Band Power)\n"
                      "How signal energy is distributed across data-driven frequency bands",
                      fontsize=13)

    for ax, dataset, data_list, title in [
        (axes_imf[0], "EEGMAT", eeg_data, "EEGMAT: rest (blue) vs arith (red)"),
        (axes_imf[1], "Stress", str_data, "Stress: below (blue) vs above (red)"),
    ]:
        frac0 = np.stack([d["cond0"]["imf_frac_mean"][:max_imf] for d in data_list])
        frac1 = np.stack([d["cond1"]["imf_frac_mean"][:max_imf] for d in data_list])
        mif = np.stack([d["cond0"]["median_if_mean"][:max_imf] for d in data_list])

        x = np.arange(max_imf)
        ax.bar(x - 0.15, frac0.mean(axis=0), 0.3, yerr=frac0.std(axis=0), color='steelblue',
               alpha=0.7, label="Cond 0", capsize=3)
        ax.bar(x + 0.15, frac1.mean(axis=0), 0.3, yerr=frac1.std(axis=0), color='salmon',
               alpha=0.7, label="Cond 1", capsize=3)
        ax.set_xlabel("IMF index")
        ax.set_ylabel("Energy Fraction")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"IMF{i}\n({mif.mean(axis=0)[i]:.1f}Hz)" for i in range(max_imf)], fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)

    fig_imf.tight_layout()
    fig_imf.savefig(f"{OUT_IMF}/imf_energy_fraction.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUT_IMF}/imf_energy_fraction.png")

    # IMF energy log-ratio (per IMF, boxplot)
    fig_lr, axes_lr = plt.subplots(1, max_imf, figsize=(3 * max_imf, 5))
    fig_lr.suptitle("IMF Energy Log₂ Ratio per IMF (cond0/cond1)", fontsize=13)

    for k in range(max_imf):
        ax = axes_lr[k]
        eeg_lr = np.array([np.log2((d["cond0"]["imf_frac_mean"][k] + 1e-12) /
                                    (d["cond1"]["imf_frac_mean"][k] + 1e-12)) for d in eeg_data])
        str_lr = np.array([np.log2((d["cond0"]["imf_frac_mean"][k] + 1e-12) /
                                    (d["cond1"]["imf_frac_mean"][k] + 1e-12)) for d in str_data])
        bp = ax.boxplot([eeg_lr, str_lr], positions=[0, 1], widths=0.35,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', linewidth=2))
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightyellow')
        ax.axhline(0, color='gray', ls='--', alpha=0.4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["EEG", "Str"], fontsize=8)
        mif_k = np.mean([d["cond0"]["median_if_mean"][k] for d in eeg_data])
        ax.set_title(f"IMF{k}\n(~{mif_k:.0f}Hz)", fontsize=9)
        if k == 0:
            ax.set_ylabel("log₂ ratio")

        _, p = mannwhitneyu(np.abs(eeg_lr), np.abs(str_lr), alternative='greater')
        ax.text(0.5, 0.02, f"p={p:.3f}", transform=ax.transAxes, fontsize=7, ha='center')

    fig_lr.tight_layout()
    fig_lr.savefig(f"{OUT_IMF}/imf_energy_logratio_boxplot.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUT_IMF}/imf_energy_logratio_boxplot.png")

    # Save data
    np.savez(f"{OUT_NL}/nonlinearity_data.npz",
             eeg_nl_lr=eeg_nl_lr, str_nl_lr=str_nl_lr)
    np.savez(f"{OUT_IMF}/imf_energy_data.npz",
             eeg_sids=np.array([d["sid"] for d in eeg_data]),
             str_sids=np.array([d["sid"] for d in str_data]))
    print("\nDone.")


if __name__ == "__main__":
    main()
