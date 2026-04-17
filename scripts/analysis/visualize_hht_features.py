"""Visualize three HHT/HHSA feature representations for condition comparison.

Figure 1: Marginal Hilbert Spectrum (1D power spectrum, HHT version of PSD)
Figure 2: Hilbert Time-Frequency Spectrum (2D, sharper than STFT)
Figure 3: AM Spectrum at alpha carrier (~8-13 Hz slice of holospectrum)

Each figure: EEGMAT (rest vs arith) on top, Stress (DSS-below vs above) on bottom.
"""
import os, sys, glob
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pipeline.hhsa as hhsa
import emd

# Fast params
hhsa.N_ENSEMBLES_L1 = 24
hhsa.N_ENSEMBLES_L2 = 12

OUT_DIR = "results/wsci_phase1"
os.makedirs(OUT_DIR, exist_ok=True)
N_JOBS = 64
FS = 200.0


# ---- Data loading (reuse from visualize_holospectrum.py) ----

def load_eegmat_subject00():
    rest = torch.load("data/cache_eegmat/eegmat_Subject00_1_w5.0_sr200.0.pt",
                       map_location="cpu", weights_only=True).numpy()
    arith = torch.load("data/cache_eegmat/eegmat_Subject00_2_w5.0_sr200.0.pt",
                        map_location="cpu", weights_only=True).numpy()
    return rest, arith


def load_stress_p03():
    df = pd.read_csv("data/comprehensive_labels.csv")
    sub = df[df["Patient_ID"] == 3]
    med = sub["Stress_Score"].median()
    above_ids = sub[sub["Stress_Score"] >= med]["Recording_ID"].tolist()
    below_ids = sub[sub["Stress_Score"] < med]["Recording_ID"].tolist()
    above_ep, below_ep = [], []
    for rid in above_ids:
        f = glob.glob(f"data/cache/*_p03_{rid}_w5.0.pt")
        if f:
            above_ep.append(torch.load(f[0], map_location="cpu", weights_only=True).numpy())
    for rid in below_ids:
        f = glob.glob(f"data/cache/*_p03_{rid}_w5.0.pt")
        if f:
            below_ep.append(torch.load(f[0], map_location="cpu", weights_only=True).numpy())
    return np.concatenate(below_ep, axis=0), np.concatenate(above_ep, axis=0)


# ---- Per-epoch HHT extraction (Layer 1+2 only, no HHSA) ----

def extract_hht_single(x_1d, fs, noise_seed):
    """Extract HHT features for a single (epoch, channel) signal.

    Returns dict with:
      - imfs: (n_samp, n_imf)
      - IF: (n_samp, n_imf) instantaneous frequency
      - IA: (n_samp, n_imf) instantaneous amplitude
    """
    x = x_1d.astype(np.float64)
    if x.std() < 1e-10 or not np.all(np.isfinite(x)):
        return None
    imfs = hhsa._safe_ceemdan(x, hhsa.N_ENSEMBLES_L1, noise_seed)
    if imfs is None:
        return None
    imfs = hhsa._truncate_imfs(imfs, hhsa.MAX_IMF_L1)
    _, IF, IA = emd.spectra.frequency_transform(imfs, fs, 'nht')
    IF, IA = hhsa._sanitize_nht(IF, IA)
    return {"imfs": imfs, "IF": IF, "IA": IA}


def extract_hht_multi(epochs, fs, base_seed, max_epochs=10, select_channel=None):
    """Extract HHT for a subset of epochs.

    If select_channel is None, uses channel 0 (Fp1 for EEGMAT, first channel for Stress).
    Returns list of dicts.
    """
    n_ep, n_ch, n_samp = epochs.shape
    ch = select_channel if select_channel is not None else 0

    rng = np.random.default_rng(base_seed)
    if n_ep > max_epochs:
        idx = rng.choice(n_ep, max_epochs, replace=False)
        epochs = epochs[idx]
        n_ep = max_epochs

    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(extract_hht_single)(epochs[ei, ch, :], fs, base_seed + ei * 1000)
        for ei in range(n_ep)
    )
    return [r for r in results if r is not None]


# ---- Figure 1: Marginal Hilbert Spectrum ----

def compute_marginal_hilbert(hht_list, fs):
    """Compute mean marginal Hilbert spectrum from a list of HHT extractions.

    Marginal = time-average of Hilbert spectrum = histogram of IF weighted by IA^2.
    """
    freq_edges = np.linspace(0.5, 45, 90)
    freq_centers = (freq_edges[:-1] + freq_edges[1:]) / 2
    all_spectra = []

    for h in hht_list:
        IF, IA = h["IF"], h["IA"]
        # Trust window: drop first/last 1s
        n_drop = int(1.0 * fs)
        IF = IF[n_drop:-n_drop, :]
        IA = IA[n_drop:-n_drop, :]
        # Weighted histogram: energy = IA^2 placed at IF bin
        spectrum = np.zeros(len(freq_centers))
        for k in range(IF.shape[1]):
            weights = IA[:, k] ** 2
            hist, _ = np.histogram(IF[:, k], bins=freq_edges, weights=weights)
            spectrum += hist
        all_spectra.append(spectrum)

    return freq_centers, np.mean(all_spectra, axis=0)


def plot_marginal_hilbert(ax, freqs, spec0, spec1, label0, label1, title):
    ax.semilogy(freqs, spec0 + 1e-10, 'b-', alpha=0.8, linewidth=1.5, label=label0)
    ax.semilogy(freqs, spec1 + 1e-10, 'r-', alpha=0.8, linewidth=1.5, label=label1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Marginal Hilbert Power")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0.5, 45)
    ax.grid(True, alpha=0.3)


# ---- Figure 2: Hilbert Time-Frequency Spectrum ----

def compute_hilbert_spectrum(hht_list, fs):
    """Compute mean Hilbert time-frequency spectrum.

    H(t, f) = sum over IMFs of IA(t) placed at IF(t).
    Returns (time_axis, freq_edges, mean_H).
    """
    freq_edges = np.linspace(0.5, 45, 90)
    n_drop = int(1.0 * fs)
    # All epochs should have same sample count after trust window
    n_samp = hht_list[0]["IF"].shape[0] - 2 * n_drop
    t_axis = np.arange(n_samp) / fs + 1.0  # start at 1s (after trust window)

    all_H = []
    for h in hht_list:
        IF = h["IF"][n_drop:-n_drop, :]
        IA = h["IA"][n_drop:-n_drop, :]
        H = np.zeros((n_samp, len(freq_edges) - 1))
        for k in range(IF.shape[1]):
            for t in range(n_samp):
                f = IF[t, k]
                a = IA[t, k]
                if f > 0 and a > 0:
                    bin_idx = np.searchsorted(freq_edges, f) - 1
                    if 0 <= bin_idx < len(freq_edges) - 1:
                        H[t, bin_idx] += a ** 2
        all_H.append(H)

    freq_centers = (freq_edges[:-1] + freq_edges[1:]) / 2
    return t_axis, freq_centers, np.mean(all_H, axis=0)


def plot_hilbert_tf(ax, t, f, H, title, vmax=None):
    if vmax is None:
        vmax = np.percentile(H[H > 0], 99) if (H > 0).any() else 1
    im = ax.pcolormesh(t, f, H.T, shading="nearest", cmap="magma",
                        vmin=0, vmax=vmax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0.5, 45)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ---- Figure 3: AM spectrum at alpha carrier ----

def plot_am_at_alpha(ax, holo_data, dataset_label):
    """Plot AM spectrum sliced at alpha carrier (8-13 Hz)."""
    d = holo_data
    fc, fa = d["carrier_centers"], d["am_centers"]

    # Find alpha carrier bins (8-13 Hz)
    alpha_mask = (fc >= 8) & (fc <= 13)
    if not alpha_mask.any():
        ax.text(0.5, 0.5, "No alpha bins", transform=ax.transAxes, ha="center")
        return

    for key, color, label in d["conditions"]:
        H = d[key]
        # Sum over alpha carrier bins → AM spectrum
        am_spec = H[alpha_mask, :].mean(axis=0)
        ax.plot(fa, am_spec, color=color, linewidth=1.5, label=label)

    ax.set_xlabel("AM Frequency (Hz)")
    ax.set_ylabel("Holospectral Energy at Alpha Carrier")
    ax.set_title(f"{dataset_label}: AM of Alpha (8-13 Hz)", fontsize=11)
    ax.set_xscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


# ---- Main ----

def main():
    print("Loading data...")
    eeg_rest, eeg_arith = load_eegmat_subject00()
    stress_below, stress_above = load_stress_p03()

    # Use a central channel: EEGMAT ch=16 (Fz), Stress ch=14 (rough Cz)
    eeg_ch = 16   # Fz
    stress_ch = 14  # approximate Cz

    print(f"Extracting HHT: EEGMAT ch={eeg_ch}, Stress ch={stress_ch}, max 10 epochs each...")

    hht_eeg_rest = extract_hht_multi(eeg_rest, FS, base_seed=100, select_channel=eeg_ch)
    hht_eeg_arith = extract_hht_multi(eeg_arith, FS, base_seed=200, select_channel=eeg_ch)
    hht_str_below = extract_hht_multi(stress_below, FS, base_seed=300, select_channel=stress_ch)
    hht_str_above = extract_hht_multi(stress_above, FS, base_seed=400, select_channel=stress_ch)
    print(f"  EEGMAT: rest={len(hht_eeg_rest)}, arith={len(hht_eeg_arith)} epochs")
    print(f"  Stress: below={len(hht_str_below)}, above={len(hht_str_above)} epochs")

    # =========== Figure 1: Marginal Hilbert Spectrum ===========
    print("\nFigure 1: Marginal Hilbert Spectrum...")
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle("Marginal Hilbert Spectrum (HHT version of PSD)", fontsize=13)

    f_e, mhs_rest = compute_marginal_hilbert(hht_eeg_rest, FS)
    _, mhs_arith = compute_marginal_hilbert(hht_eeg_arith, FS)
    plot_marginal_hilbert(axes1[0], f_e, mhs_rest, mhs_arith,
                          "Rest", "Arithmetic", "EEGMAT Sub00 (Fz)")

    f_s, mhs_below = compute_marginal_hilbert(hht_str_below, FS)
    _, mhs_above = compute_marginal_hilbert(hht_str_above, FS)
    plot_marginal_hilbert(axes1[1], f_s, mhs_below, mhs_above,
                          "DSS-below", "DSS-above", "Stress p03 (Cz)")

    fig1.tight_layout()
    fig1.savefig(f"{OUT_DIR}/fig1_marginal_hilbert.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig1_marginal_hilbert.png")

    # =========== Figure 2: Hilbert Time-Frequency ===========
    print("\nFigure 2: Hilbert Time-Frequency Spectrum...")
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle("Hilbert Time-Frequency Spectrum", fontsize=13, y=0.98)

    t_e, f_e2, H_rest = compute_hilbert_spectrum(hht_eeg_rest, FS)
    _, _, H_arith = compute_hilbert_spectrum(hht_eeg_arith, FS)
    vmax_eeg = max(np.percentile(H_rest[H_rest > 0], 99) if (H_rest > 0).any() else 1,
                    np.percentile(H_arith[H_arith > 0], 99) if (H_arith > 0).any() else 1)
    plot_hilbert_tf(axes2[0, 0], t_e, f_e2, H_rest, "EEGMAT: Rest", vmax=vmax_eeg)
    plot_hilbert_tf(axes2[0, 1], t_e, f_e2, H_arith, "EEGMAT: Arithmetic", vmax=vmax_eeg)
    # Difference
    diff_eeg = H_arith - H_rest
    vabs = max(abs(diff_eeg.min()), abs(diff_eeg.max()), 1e-6)
    im = axes2[0, 2].pcolormesh(t_e, f_e2, diff_eeg.T, shading="nearest", cmap="RdBu_r",
                                 norm=TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs))
    axes2[0, 2].set_title("EEGMAT: Arith − Rest", fontsize=11)
    axes2[0, 2].set_xlabel("Time (s)")
    axes2[0, 2].set_ylabel("Frequency (Hz)")
    axes2[0, 2].set_ylim(0.5, 45)
    plt.colorbar(im, ax=axes2[0, 2], fraction=0.046, pad=0.04)

    t_s, f_s2, H_below = compute_hilbert_spectrum(hht_str_below, FS)
    _, _, H_above = compute_hilbert_spectrum(hht_str_above, FS)
    vmax_str = max(np.percentile(H_below[H_below > 0], 99) if (H_below > 0).any() else 1,
                    np.percentile(H_above[H_above > 0], 99) if (H_above > 0).any() else 1)
    plot_hilbert_tf(axes2[1, 0], t_s, f_s2, H_below, "Stress: DSS-below", vmax=vmax_str)
    plot_hilbert_tf(axes2[1, 1], t_s, f_s2, H_above, "Stress: DSS-above", vmax=vmax_str)
    diff_str = H_above - H_below
    vabs_s = max(abs(diff_str.min()), abs(diff_str.max()), 1e-6)
    im2 = axes2[1, 2].pcolormesh(t_s, f_s2, diff_str.T, shading="nearest", cmap="RdBu_r",
                                  norm=TwoSlopeNorm(vmin=-vabs_s, vcenter=0, vmax=vabs_s))
    axes2[1, 2].set_title("Stress: Above − Below", fontsize=11)
    axes2[1, 2].set_xlabel("Time (s)")
    axes2[1, 2].set_ylabel("Frequency (Hz)")
    axes2[1, 2].set_ylim(0.5, 45)
    plt.colorbar(im2, ax=axes2[1, 2], fraction=0.046, pad=0.04)

    fig2.tight_layout()
    fig2.savefig(f"{OUT_DIR}/fig2_hilbert_tf.png", dpi=150, bbox_inches="tight")
    print(f"  Saved fig2_hilbert_tf.png")

    # =========== Figure 3: AM at Alpha Carrier ===========
    print("\nFigure 3: AM Spectrum at Alpha Carrier...")
    # Load holospectrum data from previous run
    holo_path = f"{OUT_DIR}/holospectrum_diagnostic.npz"
    if os.path.exists(holo_path):
        hd = np.load(holo_path)
        fc, fa = hd["carrier_centers"], hd["am_centers"]

        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
        fig3.suptitle("AM Spectrum at Alpha Carrier (8-13 Hz)", fontsize=13)

        alpha_mask = (fc >= 8) & (fc <= 13)

        # EEGMAT
        am_rest = hd["eegmat_rest"][alpha_mask, :].mean(axis=0)
        am_arith = hd["eegmat_arith"][alpha_mask, :].mean(axis=0)
        axes3[0].plot(fa, am_rest, 'b-', lw=1.5, label="Rest")
        axes3[0].plot(fa, am_arith, 'r-', lw=1.5, label="Arithmetic")
        axes3[0].set_xlabel("AM Frequency (Hz)")
        axes3[0].set_ylabel("Holospectral Energy")
        axes3[0].set_title("EEGMAT Sub00: What modulates alpha?", fontsize=11)
        axes3[0].set_xscale("log")
        axes3[0].legend()
        axes3[0].grid(True, alpha=0.3)

        # Stress
        am_below = hd["stress_below"][alpha_mask, :].mean(axis=0)
        am_above = hd["stress_above"][alpha_mask, :].mean(axis=0)
        axes3[1].plot(fa, am_below, 'b-', lw=1.5, label="DSS-below")
        axes3[1].plot(fa, am_above, 'r-', lw=1.5, label="DSS-above")
        axes3[1].set_xlabel("AM Frequency (Hz)")
        axes3[1].set_ylabel("Holospectral Energy")
        axes3[1].set_title("Stress p03: What modulates alpha?", fontsize=11)
        axes3[1].set_xscale("log")
        axes3[1].legend()
        axes3[1].grid(True, alpha=0.3)

        fig3.tight_layout()
        fig3.savefig(f"{OUT_DIR}/fig3_am_at_alpha.png", dpi=150, bbox_inches="tight")
        print(f"  Saved fig3_am_at_alpha.png")
    else:
        print(f"  Skipped (no holospectrum data at {holo_path})")

    print("\nDone.")


if __name__ == "__main__":
    main()
