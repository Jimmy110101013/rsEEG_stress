"""Direction 3: Cross-channel AM coherence.

For each channel pair, compute correlation of instantaneous amplitude (IA)
time series within the same IMF. High IA correlation = synchronized amplitude
modulation = functional connectivity in the AM domain.

This is HHT-unique: PSD gives no inter-channel AM coupling information.

Method:
  1. Extract IMFs + IA for all channels (Layer 1-2, no HHSA needed)
  2. For each IMF, compute pairwise IA Pearson correlation across channels
  3. Average over epochs → one (n_ch × n_ch) connectivity matrix per IMF
  4. Compare between conditions: log₂ ratio of mean connectivity strength

Output: results/hhsa/08_am_coherence/
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
OUT = "results/hhsa/08_am_coherence"
os.makedirs(OUT, exist_ok=True)

EEGMAT_CH = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4", "T5", "T6",
             "C3", "C4", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz"]
STRESS_CH = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3', 'FCZ',
             'FC4', 'FT8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'TP7', 'CP3', 'CPZ',
             'CP4', 'TP8', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'OZ', 'O2']


def extract_ia_all_channels(epoch_data, fs, seed):
    """Extract IA for all channels of a single epoch.

    Parameters
    ----------
    epoch_data : ndarray, shape (n_ch, n_samp)

    Returns
    -------
    IA_dict : dict {imf_idx: (n_samp_trust, n_ch)} or None
    median_ifs : (max_imf,) median IF per IMF (from first valid channel)
    """
    n_ch, n_samp = epoch_data.shape
    n_drop = int(1.0 * fs)
    max_imf = 6  # focus on first 6 IMFs

    # Extract per-channel
    all_IA = []  # list of (n_samp_trust, n_imf) per channel
    ref_median_if = None
    for ci in range(n_ch):
        x = epoch_data[ci, :].astype(np.float64)
        if x.std() < 1e-10:
            all_IA.append(None)
            continue
        imfs = hhsa._safe_ceemdan(x, hhsa.N_ENSEMBLES_L1, seed + ci)
        if imfs is None:
            all_IA.append(None)
            continue
        imfs = hhsa._truncate_imfs(imfs, max_imf)
        _, IF, IA = emd.spectra.frequency_transform(imfs, fs, 'nht')
        IF, IA = hhsa._sanitize_nht(IF, IA)
        IA = IA[n_drop:-n_drop, :]
        IF = IF[n_drop:-n_drop, :]
        all_IA.append(IA)
        if ref_median_if is None:
            ref_median_if = np.array([np.median(IF[:, k]) for k in range(IF.shape[1])])

    if ref_median_if is None:
        return None, None

    # Align to common n_imf (pad shorter ones)
    n_trust = n_samp - 2 * n_drop
    IA_matrix = {}  # {imf_k: (n_trust, n_ch)}
    for k in range(min(max_imf, len(ref_median_if))):
        ia_k = np.zeros((n_trust, n_ch))
        for ci in range(n_ch):
            if all_IA[ci] is not None and k < all_IA[ci].shape[1]:
                ia_k[:, ci] = all_IA[ci][:, k]
        IA_matrix[k] = ia_k

    return IA_matrix, ref_median_if


def compute_am_connectivity(epochs, fs, base_seed, max_ep=8):
    """Compute mean AM connectivity matrix per IMF.

    Returns
    -------
    conn : dict {imf_k: (n_ch, n_ch)} mean Pearson correlation of IA
    median_ifs : (n_imf,)
    """
    n_ep, n_ch, _ = epochs.shape
    n_ep = min(n_ep, max_ep)
    rng = np.random.default_rng(base_seed)
    idx = rng.choice(epochs.shape[0], n_ep, replace=False) if epochs.shape[0] > max_ep else np.arange(n_ep)

    all_conn = {}  # {imf_k: list of (n_ch, n_ch)}
    ref_if = None

    for ei in idx:
        ia_dict, mif = extract_ia_all_channels(epochs[ei], fs, base_seed + int(ei) * 100)
        if ia_dict is None:
            continue
        if ref_if is None:
            ref_if = mif
        for k, ia_k in ia_dict.items():
            # Pearson correlation matrix of IA across channels
            # ia_k shape: (n_trust, n_ch)
            corr = np.corrcoef(ia_k.T)  # (n_ch, n_ch)
            corr = np.where(np.isfinite(corr), corr, 0)
            all_conn.setdefault(k, []).append(corr)

    # Average over epochs
    conn_mean = {}
    for k, corr_list in all_conn.items():
        conn_mean[k] = np.mean(corr_list, axis=0)

    return conn_mean, ref_if


def conn_strength(conn_matrix):
    """Mean off-diagonal absolute correlation (global connectivity strength)."""
    n = conn_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return np.abs(conn_matrix[mask]).mean()


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
    max_imf = 6

    # ===== EEGMAT =====
    print("EEGMAT: AM coherence (all channels, 8 epochs/condition)...")
    eeg_data = []
    for sid in range(36):
        sname = f"Subject{sid:02d}"
        rest = torch.load(f"data/cache_eegmat/eegmat_{sname}_1_w5.0_sr200.0.pt",
                           map_location="cpu", weights_only=True).numpy()
        arith = torch.load(f"data/cache_eegmat/eegmat_{sname}_2_w5.0_sr200.0.pt",
                            map_location="cpu", weights_only=True).numpy()
        conn_rest, mif = compute_am_connectivity(rest, FS, 1100000 + sid * 500)
        conn_arith, _ = compute_am_connectivity(arith, FS, 1200000 + sid * 500)
        if conn_rest and conn_arith:
            strengths = {}
            for k in range(min(max_imf, len(conn_rest), len(conn_arith))):
                if k in conn_rest and k in conn_arith:
                    strengths[k] = {
                        "cond0": conn_strength(conn_rest[k]),
                        "cond1": conn_strength(conn_arith[k]),
                    }
            eeg_data.append({"sid": sname, "strengths": strengths, "mif": mif,
                             "conn_rest": conn_rest, "conn_arith": conn_arith})
        if (sid + 1) % 12 == 0:
            print(f"  [{sid+1}/36] {time.time()-t_start:.0f}s")

    # ===== Stress =====
    print("\nStress-DSS: AM coherence...")
    stress_subjects = load_stress_dss_splits()
    str_data = []
    for pid_str, data in stress_subjects.items():
        conn_below, mif = compute_am_connectivity(data["below"], FS, 1300000 + int(pid_str[1:]) * 500)
        conn_above, _ = compute_am_connectivity(data["above"], FS, 1400000 + int(pid_str[1:]) * 500)
        if conn_below and conn_above:
            strengths = {}
            for k in range(min(max_imf, len(conn_below), len(conn_above))):
                if k in conn_below and k in conn_above:
                    strengths[k] = {
                        "cond0": conn_strength(conn_below[k]),
                        "cond1": conn_strength(conn_above[k]),
                    }
            str_data.append({"sid": pid_str, "strengths": strengths, "mif": mif,
                             "conn_below": conn_below, "conn_above": conn_above})
        print(f"  {pid_str}: done")

    print(f"\nTotal time: {time.time()-t_start:.0f}s")

    # ===== Plot: per-IMF connectivity strength comparison =====
    fig, axes = plt.subplots(1, max_imf, figsize=(3.5 * max_imf, 5))
    fig.suptitle("Dir 3: AM Coherence (mean |correlation| across channel pairs)\n"
                 "Condition contrast per IMF", fontsize=13)

    for k in range(max_imf):
        ax = axes[k]
        eeg_lr = []
        for d in eeg_data:
            if k in d["strengths"]:
                s = d["strengths"][k]
                eeg_lr.append(np.log2((s["cond0"] + 1e-12) / (s["cond1"] + 1e-12)))
        str_lr = []
        for d in str_data:
            if k in d["strengths"]:
                s = d["strengths"][k]
                str_lr.append(np.log2((s["cond0"] + 1e-12) / (s["cond1"] + 1e-12)))

        eeg_lr = np.array(eeg_lr) if eeg_lr else np.array([0])
        str_lr = np.array(str_lr) if str_lr else np.array([0])

        bp = ax.boxplot([eeg_lr, str_lr], positions=[0, 1], widths=0.35,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', linewidth=2))
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightyellow')
        ax.axhline(0, color='gray', ls='--', alpha=0.4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["EEG", "Str"], fontsize=8)

        mif_k = np.mean([d["mif"][k] for d in eeg_data if d["mif"] is not None and k < len(d["mif"])])
        ax.set_title(f"IMF{k}\n(~{mif_k:.0f}Hz)", fontsize=9)
        if k == 0:
            ax.set_ylabel("log₂ ratio\n(cond0/cond1)")

        try:
            _, p = mannwhitneyu(np.abs(eeg_lr), np.abs(str_lr), alternative='greater')
        except:
            p = 1.0
        ax.text(0.5, 0.02, f"p={p:.3f}", transform=ax.transAxes, fontsize=7, ha='center')

    fig.tight_layout()
    fig.savefig(f"{OUT}/am_coherence_boxplot.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT}/am_coherence_boxplot.png")

    # Plot example connectivity matrices (first EEGMAT subject, first Stress subject)
    if eeg_data and str_data:
        fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
        fig2.suptitle("Example AM Connectivity Matrices (IMF1 ≈ alpha)", fontsize=13)

        k_alpha = 1  # typically alpha
        for row, (data, ch_names, label) in enumerate([
            (eeg_data[0], EEGMAT_CH, "EEGMAT Sub00"),
            (str_data[0], STRESS_CH, f"Stress {str_data[0]['sid']}"),
        ]):
            conn_key0 = "conn_rest" if row == 0 else "conn_below"
            conn_key1 = "conn_arith" if row == 0 else "conn_above"
            c0 = data[conn_key0].get(k_alpha, np.zeros((len(ch_names), len(ch_names))))
            c1 = data[conn_key1].get(k_alpha, np.zeros((len(ch_names), len(ch_names))))

            im0 = axes2[row, 0].imshow(c0, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            axes2[row, 0].set_title(f"{label}: Cond 0", fontsize=9)
            axes2[row, 0].set_xticks(range(len(ch_names)))
            axes2[row, 0].set_xticklabels(ch_names, rotation=90, fontsize=5)
            axes2[row, 0].set_yticks(range(len(ch_names)))
            axes2[row, 0].set_yticklabels(ch_names, fontsize=5)
            plt.colorbar(im0, ax=axes2[row, 0], fraction=0.046)

            im1 = axes2[row, 1].imshow(c1, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            axes2[row, 1].set_title(f"{label}: Cond 1", fontsize=9)
            axes2[row, 1].set_xticks(range(len(ch_names)))
            axes2[row, 1].set_xticklabels(ch_names, rotation=90, fontsize=5)
            plt.colorbar(im1, ax=axes2[row, 1], fraction=0.046)

            diff = c0 - c1
            vd = max(abs(diff.min()), abs(diff.max()), 0.1)
            im2 = axes2[row, 2].imshow(diff, cmap='RdBu_r', vmin=-vd, vmax=vd, aspect='auto')
            axes2[row, 2].set_title(f"{label}: Diff", fontsize=9)
            axes2[row, 2].set_xticks(range(len(ch_names)))
            axes2[row, 2].set_xticklabels(ch_names, rotation=90, fontsize=5)
            plt.colorbar(im2, ax=axes2[row, 2], fraction=0.046)

            # Histogram of off-diagonal differences
            mask = ~np.eye(len(ch_names), dtype=bool)
            axes2[row, 3].hist(diff[mask].ravel(), bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            axes2[row, 3].axvline(0, color='red', ls='--')
            axes2[row, 3].set_title(f"{label}: Diff distribution", fontsize=9)
            axes2[row, 3].set_xlabel("Δ correlation")

        fig2.tight_layout()
        fig2.savefig(f"{OUT}/am_coherence_matrices.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {OUT}/am_coherence_matrices.png")

    print("Done.")


if __name__ == "__main__":
    main()
