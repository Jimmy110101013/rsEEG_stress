"""Diagnostic figure: does fine-tuning actually reshape the LaBraM
representation's variance structure?

Motivated by the finding that on Stress, the per-dim ω² ratio appears to
drop dramatically after fine-tuning (9.1× → 1.6×) but PERMANOVA doesn't
improve at all. The per-fold ω² number turned out to be an artifact of
having only 1 increase-subject per fold (nested decomposition degenerate),
and the pooled view shows the label fraction is actually unchanged by
fine-tuning on Stress. This script visualizes that directly.

Produces `paper/figures/label_subspace.{pdf,png}` with three rows:

  Row 1 — Global label variance fraction (SS_label / SS_total, pooled)
          per dataset × regime. Flat bar + change annotation.
  Row 2 — Cumulative label SS curve across sorted feature dims. If
          fine-tuning concentrates label signal into a few dims, the FT
          curve should rise more steeply.
  Row 3 — Top-2 PCA projection per dataset × regime, colored by label.
          Visual confirmation of whether the representation's global
          geometry rearranged for the label axis.

Also produces a separate t-SNE figure for each dataset (in a 2-col grid:
frozen | fine-tuned), saved alongside.

Run under timm_eeg (uses sklearn.manifold.TSNE):
    conda run -n timm_eeg python scripts/build_label_subspace_figure.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src import variance_analysis as va  # noqa: E402


DATASETS = {
    "Stress": {
        "frozen": "results/cross_dataset/features_stress_19ch.npz",
        "ft_dir": "results/20260406_0419_ft_subjectdass_aug75_labram_feat",
        "label_names": ("normal", "increase"),
    },
    "ADFTD": {
        "frozen": "results/cross_dataset/features_adftd_19ch.npz",
        "ft_dir": "results/20260406_0935_ft_dass_aug75_labram_adftd_feat",
        "label_names": ("HC", "AD"),
    },
    "TDBRAIN": {
        "frozen": "results/cross_dataset/features_tdbrain_19ch.npz",
        "ft_dir": "results/20260407_1533_ft_dass_aug75_labram_tdbrain_feat",
        "label_names": ("HC", "MDD"),
    },
}

OUT_DIR = "paper/figures"
OUT_PDF = f"{OUT_DIR}/label_subspace.pdf"
OUT_PNG = f"{OUT_DIR}/label_subspace.png"
OUT_TSNE_PNG = f"{OUT_DIR}/tsne_frozen_vs_ft.png"


def load_regime_pair(cfg: dict) -> dict:
    """Return dict with frozen and ft_pooled features/labels."""
    fz_f, fz_y, fz_s = va.load_frozen_features(cfg["frozen"], cfg["ft_dir"])
    ft_per_fold = va.load_ft_features_per_fold(cfg["ft_dir"])
    pooled_f = np.concatenate([t[0] for t in ft_per_fold])
    pooled_y = np.concatenate([t[1] for t in ft_per_fold])
    pooled_s = np.concatenate([t[2] for t in ft_per_fold])
    return {
        "frozen":    (fz_f, fz_y, fz_s),
        "ft_pooled": (pooled_f, pooled_y, pooled_s),
    }


def global_label_fraction(f: np.ndarray, y: np.ndarray) -> float:
    """SS_label / SS_total, summed over feature dims."""
    grand = f.mean(axis=0)
    label_ss = sum(
        (y == lab).sum() * ((f[y == lab].mean(axis=0) - grand) ** 2)
        for lab in np.unique(y)
    )
    total_ss = ((f - grand) ** 2).sum(axis=0)
    return float(label_ss.sum() / max(total_ss.sum(), 1e-18))


def cumulative_label_curve(
    f: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sorted per-dim label SS and its cumulative fraction.

    Returns (sorted_label_ss_desc, cumulative_normalized_fraction).
    """
    grand = f.mean(axis=0)
    label_ss = np.zeros(f.shape[1])
    for lab in np.unique(y):
        mask = y == lab
        label_ss += mask.sum() * (f[mask].mean(axis=0) - grand) ** 2
    sorted_desc = np.sort(label_ss)[::-1]
    cum = np.cumsum(sorted_desc) / max(sorted_desc.sum(), 1e-18)
    return sorted_desc, cum


def pca_top2(f: np.ndarray) -> np.ndarray:
    centered = f - f.mean(axis=0, keepdims=True)
    u, sv, _ = np.linalg.svd(centered, full_matrices=False)
    return (u[:, :2] * sv[:2])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_data = {name: load_regime_pair(cfg) for name, cfg in DATASETS.items()}

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 120,
    })

    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    dataset_names = list(DATASETS.keys())

    # ------------------------------------------------------------------
    # Row 1: Global label variance fraction (bar)
    # ------------------------------------------------------------------
    for col, name in enumerate(dataset_names):
        ax = axes[0, col]
        regimes = all_data[name]
        fz_f, fz_y, _ = regimes["frozen"]
        ft_f, ft_y, _ = regimes["ft_pooled"]
        frac_fz = global_label_fraction(fz_f, fz_y)
        frac_ft = global_label_fraction(ft_f, ft_y)
        xs = [0, 1]
        vals = [frac_fz, frac_ft]
        colors = ["#A6A6A6", "#404040"]
        ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.7)
        ax.set_xticks(xs)
        ax.set_xticklabels(["Frozen", "Fine-tuned"])
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.03, f"{v*100:.1f}%",
                    ha="center", va="bottom", fontsize=9)
        # Fold change annotation
        if frac_fz > 0:
            mult = frac_ft / frac_fz
            sign = "↑" if mult > 1.05 else ("↓" if mult < 0.95 else "→")
            color = {"↑": "#388E3C", "↓": "#D32F2F", "→": "#555555"}[sign]
            ax.text(0.5, max(vals) * 1.2,
                    f"{sign} {mult:.2f}×", ha="center",
                    fontsize=11, color=color, fontweight="bold",
                    transform=ax.transData)
        ax.set_ylim(0, max(vals) * 1.4)
        ax.set_title(f"{name}  n={fz_f.shape[0]}")
        if col == 0:
            ax.set_ylabel(r"$SS_{\mathrm{label}} / SS_{\mathrm{total}}$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Row 2: Cumulative label SS curves
    # ------------------------------------------------------------------
    for col, name in enumerate(dataset_names):
        ax = axes[1, col]
        regimes = all_data[name]
        fz_f, fz_y, _ = regimes["frozen"]
        ft_f, ft_y, _ = regimes["ft_pooled"]
        _, cum_fz = cumulative_label_curve(fz_f, fz_y)
        _, cum_ft = cumulative_label_curve(ft_f, ft_y)
        ranks = np.arange(1, len(cum_fz) + 1)
        ax.plot(ranks, cum_fz, color="#A6A6A6", lw=2, label="Frozen")
        ax.plot(ranks, cum_ft, color="#404040", lw=2, label="Fine-tuned")
        ax.axhline(0.8, color="red", linestyle=":", lw=0.8, alpha=0.6)
        # Annotate dims_for_80pct
        def _rank_at_80(cum):
            return int(np.searchsorted(cum, 0.80) + 1)
        k_fz = _rank_at_80(cum_fz)
        k_ft = _rank_at_80(cum_ft)
        ax.axvline(k_fz, color="#A6A6A6", linestyle="--", lw=0.6)
        ax.axvline(k_ft, color="#404040", linestyle="--", lw=0.6)
        ax.text(0.98, 0.05,
                f"80% at dim  {k_fz} → {k_ft}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8,
                          edgecolor="none"))
        ax.set_xlim(0, min(100, len(cum_fz)))
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Sorted feature-dim rank")
        if col == 0:
            ax.set_ylabel("Cumulative label SS fraction")
        ax.legend(loc="lower right", frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Row 3: Top-2 PCA projection, colored by label
    # ------------------------------------------------------------------
    for col, name in enumerate(dataset_names):
        ax = axes[2, col]
        cfg = DATASETS[name]
        regimes = all_data[name]
        fz_f, fz_y, _ = regimes["frozen"]
        ft_f, ft_y, _ = regimes["ft_pooled"]

        # Overlay frozen (grey markers) and ft (colored) — or do side-by-side
        # via the subject-level centroids. Simpler: show FT only, with
        # label color. Frozen is implied by "no FT change" → we visualize
        # the contrast across the row as a whole.
        proj = pca_top2(ft_f)
        colors = ["#1976D2" if lab == 0 else "#E64A19" for lab in ft_y]
        ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=30,
                   edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.set_xlabel("PC1 (fine-tuned)")
        if col == 0:
            ax.set_ylabel("PC2 (fine-tuned)")
        label_names = cfg["label_names"]
        ax.legend(
            handles=[
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor="#1976D2", markersize=7,
                           label=label_names[0]),
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor="#E64A19", markersize=7,
                           label=label_names[1]),
            ], loc="upper right", frameon=False, fontsize=8,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Does fine-tuning rewrite the LaBraM representation? "
        "(label variance structure, frozen vs fine-tuned)",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
    print(f"Saved → {OUT_PDF}")
    print(f"Saved → {OUT_PNG}")

    # ------------------------------------------------------------------
    # Separate figure: t-SNE, frozen vs fine-tuned
    # ------------------------------------------------------------------
    try:
        from sklearn.manifold import TSNE
    except Exception as exc:
        print(f"[skip] t-SNE: {exc}")
        return

    fig2, axes2 = plt.subplots(3, 2, figsize=(7.5, 10))
    for row, name in enumerate(dataset_names):
        cfg = DATASETS[name]
        regimes = all_data[name]
        for col, regime in enumerate(["frozen", "ft_pooled"]):
            f, y, _ = regimes[regime]
            # t-SNE with a deterministic seed.
            perp = min(30, max(5, f.shape[0] // 3))
            emb = TSNE(
                n_components=2, perplexity=perp, random_state=0,
                init="pca", learning_rate="auto",
            ).fit_transform(f)
            ax = axes2[row, col]
            colors = ["#1976D2" if lab == 0 else "#E64A19" for lab in y]
            ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=20,
                       edgecolor="black", linewidth=0.25, alpha=0.85)
            title_regime = "Frozen" if regime == "frozen" else "Fine-tuned"
            ax.set_title(f"{name} — {title_regime}  (n={f.shape[0]})")
            ax.set_xticks([]); ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col == 1 and row == 0:
                label_names = cfg["label_names"]
                ax.legend(
                    handles=[
                        plt.Line2D([0], [0], marker="o", color="w",
                                   markerfacecolor="#1976D2", markersize=7,
                                   label=label_names[0]),
                        plt.Line2D([0], [0], marker="o", color="w",
                                   markerfacecolor="#E64A19", markersize=7,
                                   label=label_names[1]),
                    ], loc="upper right", frameon=False, fontsize=7,
                )

    fig2.suptitle("t-SNE of LaBraM features, colored by label", y=1.00)
    fig2.tight_layout()
    fig2.savefig(OUT_TSNE_PNG, bbox_inches="tight", dpi=200)
    print(f"Saved → {OUT_TSNE_PNG}")


if __name__ == "__main__":
    main()
