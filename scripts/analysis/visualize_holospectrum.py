"""Visualize raw holospectra: condition A vs B, for one subject each dataset.

Generates a 2×3 figure:
  Row 1 (EEGMAT Subject00): [rest mean] [arith mean] [difference]
  Row 2 (Stress p03):       [DSS-below] [DSS-above]  [difference]

Each panel is a 2D heatmap: x = carrier freq, y = AM freq, color = energy.
This is the "eyeball test" before worrying about WSCI numbers.
"""
import os, sys, glob
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pipeline.hhsa as hhsa
from pipeline.hhsa import aggregate_channels_geometric, CARRIER_EDGES, AM_EDGES
from joblib import Parallel, delayed

# Use fast params
hhsa.N_ENSEMBLES_L1 = 24
hhsa.N_ENSEMBLES_L2 = 12

OUT_DIR = "results/wsci_phase1"
os.makedirs(OUT_DIR, exist_ok=True)

SEED_STRIDE = 10_000
N_JOBS = 64


def compute_one(x, fs, seed):
    res = hhsa.compute_holospectrum(x.astype(np.float64), fs, noise_seed=seed)
    return res.holospectrum.astype(np.float32)


def compute_mean_holospectrum(epochs, fs, base_seed, max_epochs=20):
    """Compute mean holospectrum over a subset of epochs (channel-aggregated).

    Subsamples to max_epochs to keep runtime ~2 min per condition.
    """
    n_ep, n_ch, n_samp = epochs.shape
    if n_ep > max_epochs:
        rng = np.random.default_rng(base_seed)
        idx = rng.choice(n_ep, max_epochs, replace=False)
        epochs = epochs[idx]
        n_ep = max_epochs

    tasks = []
    for ei in range(n_ep):
        for ci in range(n_ch):
            seed = base_seed + (ei * n_ch + ci) * SEED_STRIDE
            tasks.append((epochs[ei, ci, :], fs, seed))

    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(compute_one)(x, fs, s) for x, fs, s in tasks
    )

    n_fc, n_fa = results[0].shape
    H = np.zeros((n_ep, n_ch, n_fc, n_fa), dtype=np.float32)
    idx = 0
    for ei in range(n_ep):
        for ci in range(n_ch):
            H[ei, ci] = results[idx]
            idx += 1

    # Channel aggregation: arithmetic mean (not geometric — for visualization
    # we want to see actual energy scale, not log-compressed)
    H_chan = H.mean(axis=1)   # (n_ep, n_fc, n_fa)
    H_mean = H_chan.mean(axis=0)  # (n_fc, n_fa)
    return H_mean, H_chan


def plot_comparison(H_mean_0, H_mean_1, label_0, label_1, dataset_name,
                    carrier_centers, am_centers, ax_row):
    """Plot condition 0 | condition 1 | difference on a row of 3 axes."""
    # Shared color scale for conditions
    vmax = max(H_mean_0.max(), H_mean_1.max())
    vmin = 1e-6

    for ax, H, title in [(ax_row[0], H_mean_0, f"{dataset_name}: {label_0}"),
                          (ax_row[1], H_mean_1, f"{dataset_name}: {label_1}")]:
        im = ax.pcolormesh(
            carrier_centers, am_centers, H.T,
            shading="nearest", cmap="viridis",
            norm=LogNorm(vmin=max(vmin, H[H > 0].min() if (H > 0).any() else vmin),
                         vmax=max(vmax, vmin * 10)),
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Carrier freq (Hz)")
        ax.set_ylabel("AM freq (Hz)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Difference: cond1 - cond0
    diff = H_mean_1 - H_mean_0
    vabs = max(abs(diff.min()), abs(diff.max()), 1e-6)
    im = ax_row[2].pcolormesh(
        carrier_centers, am_centers, diff.T,
        shading="nearest", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs),
    )
    ax_row[2].set_title(f"{dataset_name}: {label_1} − {label_0}", fontsize=10)
    ax_row[2].set_xlabel("Carrier freq (Hz)")
    ax_row[2].set_ylabel("AM freq (Hz)")
    ax_row[2].set_xscale("log")
    ax_row[2].set_yscale("log")
    plt.colorbar(im, ax=ax_row[2], fraction=0.046, pad=0.04)


def main():
    fc = (CARRIER_EDGES[:-1] + CARRIER_EDGES[1:]) / 2
    fa = (AM_EDGES[:-1] + AM_EDGES[1:]) / 2
    fs = 200.0

    # ----- EEGMAT Subject00 -----
    print("Loading EEGMAT Subject00...")
    rest = torch.load("data/cache_eegmat/eegmat_Subject00_1_w5.0_sr200.0.pt",
                       map_location="cpu", weights_only=True).numpy()
    arith = torch.load("data/cache_eegmat/eegmat_Subject00_2_w5.0_sr200.0.pt",
                        map_location="cpu", weights_only=True).numpy()
    print(f"  rest: {rest.shape}, arith: {arith.shape}")

    print("Computing EEGMAT holospectra (max 20 epochs each)...")
    H_rest, _ = compute_mean_holospectrum(rest, fs, base_seed=0)
    H_arith, _ = compute_mean_holospectrum(arith, fs, base_seed=1)
    print(f"  rest energy: {H_rest.sum():.2f}, arith energy: {H_arith.sum():.2f}")

    # ----- Stress p03 (mixed-label subject, DSS split) -----
    print("\nLoading Stress p03...")
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

    above_all = np.concatenate(above_ep, axis=0)
    below_all = np.concatenate(below_ep, axis=0)
    print(f"  above-median: {above_all.shape}, below-median: {below_all.shape}")

    print("Computing Stress holospectra (max 20 epochs each)...")
    H_above, _ = compute_mean_holospectrum(above_all, fs, base_seed=2)
    H_below, _ = compute_mean_holospectrum(below_all, fs, base_seed=3)
    print(f"  above energy: {H_above.sum():.2f}, below energy: {H_below.sum():.2f}")

    # ----- Plot -----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Holospectrum Condition Comparison (eyeball test)", fontsize=14, y=0.98)

    plot_comparison(H_rest, H_arith, "Rest", "Arithmetic", "EEGMAT Sub00",
                    fc, fa, axes[0])
    plot_comparison(H_below, H_above, "DSS-below", "DSS-above", "Stress p03",
                    fc, fa, axes[1])

    plt.tight_layout()
    out_path = f"{OUT_DIR}/holospectrum_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")

    # Also save raw data for notebook exploration
    np.savez(
        f"{OUT_DIR}/holospectrum_diagnostic.npz",
        eegmat_rest=H_rest, eegmat_arith=H_arith,
        stress_below=H_below, stress_above=H_above,
        carrier_centers=fc, am_centers=fa,
    )
    print(f"Saved raw data to {OUT_DIR}/holospectrum_diagnostic.npz")


if __name__ == "__main__":
    main()
