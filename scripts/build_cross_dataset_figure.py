"""Generate the cross-dataset signal-strength figure for paper §4.3.

Loads frozen and fine-tuned LaBraM features for UCSD Stress, ADFTD, and
TDBRAIN, computes η² for subject and label factors, and produces a 2-panel
figure: (left) recording-level BA per dataset; (right) η² ratio
(subject:label) per dataset for both frozen and FT features.

Output: paper/figures/cross_dataset_signal_strength.pdf

Run from project root:
    conda run -n timm_eeg python scripts/build_cross_dataset_figure.py
"""
from __future__ import annotations

import json
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
RESULTS = "results"
CROSS_DIR = f"{RESULTS}/cross_dataset"
STRESS_FT = f"{RESULTS}/20260406_0419_ft_subjectdass_aug75_labram_feat"
ADFTD_FT = f"{RESULTS}/20260406_0935_ft_dass_aug75_labram_adftd_feat"
TDBRAIN_FT = f"{RESULTS}/20260407_1533_ft_dass_aug75_labram_tdbrain_feat"
OUT_PDF = "paper/figures/cross_dataset_signal_strength.pdf"
OUT_PNG = "paper/figures/cross_dataset_signal_strength.png"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_ba(run_dir: str) -> float:
    with open(f"{run_dir}/summary.json") as f:
        s = json.load(f)
    return float(s.get("subject_bal_acc", s.get("bal_acc")))


def load_ft_features(run_dir: str):
    """Concatenate test-fold features across folds."""
    feats, labels, pids = [], [], []
    for npz in sorted(glob(f"{run_dir}/fold*_features.npz")):
        d = np.load(npz)
        feats.append(d["features"])
        labels.append(d["labels"])
        pids.append(d["patient_ids"])
    return (
        np.concatenate(feats),
        np.concatenate(labels),
        np.concatenate(pids),
    )


def eta_squared(features: np.ndarray, factor: np.ndarray) -> float:
    """One-way ANOVA SS_between / SS_total per feature dim, averaged."""
    f = np.asarray(features, dtype=np.float64)
    g = np.asarray(factor)
    grand_mean = f.mean(axis=0, keepdims=True)
    ss_total = ((f - grand_mean) ** 2).sum(axis=0)
    ss_between = np.zeros(f.shape[1])
    for u in np.unique(g):
        mask = g == u
        if mask.sum() < 2:
            continue
        gmean = f[mask].mean(axis=0)
        ss_between += mask.sum() * (gmean - grand_mean.squeeze()) ** 2
    return float(np.mean(ss_between / (ss_total + 1e-12)))


def load_frozen_with_meta(npz_path: str, ft_run_dir: str | None = None):
    """Load frozen features.

    If labels/pids are stored in the npz, use those directly. Otherwise
    derive them from the FT-feature dir's `test_idx` field (which maps each
    fold's test samples back to the original dataset order, so concatenating
    across folds and sorting by test_idx recovers a per-recording label/pid
    array that matches the frozen features' sequential order).
    """
    d = np.load(npz_path)
    feats = d["features"]
    if "labels" in d.files and "patient_ids" in d.files:
        return feats, d["labels"], d["patient_ids"]

    if ft_run_dir is None:
        raise ValueError(f"{npz_path} has no labels and no fallback dir")

    # Concatenate all folds, then sort by test_idx to recover dataset order
    test_idxs, labels, pids = [], [], []
    for npz in sorted(glob(f"{ft_run_dir}/fold*_features.npz")):
        d2 = np.load(npz)
        test_idxs.append(d2["test_idx"])
        labels.append(d2["labels"])
        pids.append(d2["patient_ids"])
    test_idxs = np.concatenate(test_idxs)
    labels = np.concatenate(labels)
    pids = np.concatenate(pids)

    order = np.argsort(test_idxs)
    labels_sorted = labels[order]
    pids_sorted = pids[order]
    idx_sorted = test_idxs[order]

    n = feats.shape[0]
    if not np.array_equal(idx_sorted, np.arange(n)):
        # FT features may not cover all rows if some recordings were
        # dropped during training. Match by index.
        out_labels = np.full(n, -1, dtype=labels_sorted.dtype)
        out_pids = np.full(n, -1, dtype=pids_sorted.dtype)
        out_labels[idx_sorted] = labels_sorted
        out_pids[idx_sorted] = pids_sorted
        valid = out_labels >= 0
        return feats[valid], out_labels[valid], out_pids[valid]

    return feats, labels_sorted, pids_sorted


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("Loading FT BAs from summary.json ...")
    ba = {
        "Stress":   load_ba(STRESS_FT),
        "TDBRAIN":  load_ba(TDBRAIN_FT),
        "ADFTD":    load_ba(ADFTD_FT),
    }
    print(f"  BA: {ba}")

    print("Loading FT features ...")
    ft = {
        "Stress":  load_ft_features(STRESS_FT),
        "TDBRAIN": load_ft_features(TDBRAIN_FT),
        "ADFTD":   load_ft_features(ADFTD_FT),
    }
    for name, (f, y, p) in ft.items():
        print(f"  {name} FT: feats={f.shape} subjects={len(np.unique(p))}")

    print("Loading frozen features ...")
    frozen = {
        "Stress":  load_frozen_with_meta(f"{CROSS_DIR}/features_stress_19ch.npz",  STRESS_FT),
        "TDBRAIN": load_frozen_with_meta(f"{CROSS_DIR}/features_tdbrain_19ch.npz", TDBRAIN_FT),
        "ADFTD":   load_frozen_with_meta(f"{CROSS_DIR}/features_adftd_19ch.npz",   ADFTD_FT),
    }
    for name, (f, y, p) in frozen.items():
        print(f"  {name} frozen: feats={f.shape} subjects={len(np.unique(p))}")

    print("Computing η² ...")
    rows = []
    for name in ["Stress", "TDBRAIN", "ADFTD"]:
        f_fz, y_fz, p_fz = frozen[name]
        f_ft, y_ft, p_ft = ft[name]

        eta_subj_fz = eta_squared(f_fz, p_fz)
        eta_lab_fz  = eta_squared(f_fz, y_fz)
        eta_subj_ft = eta_squared(f_ft, p_ft)
        eta_lab_ft  = eta_squared(f_ft, y_ft)

        rows.append({
            "dataset": name,
            "ba": ba[name],
            "n_subj": len(np.unique(p_fz)),
            "eta_subj_frozen": eta_subj_fz,
            "eta_label_frozen": eta_lab_fz,
            "ratio_frozen": eta_subj_fz / max(eta_lab_fz, 1e-9),
            "eta_subj_ft": eta_subj_ft,
            "eta_label_ft": eta_lab_ft,
            "ratio_ft": eta_subj_ft / max(eta_lab_ft, 1e-9),
        })
        print(
            f"  {name:8s} | BA={ba[name]:.3f} | "
            f"frozen ratio={rows[-1]['ratio_frozen']:.1f} | "
            f"FT ratio={rows[-1]['ratio_ft']:.1f}"
        )

    # Save numerical summary alongside the figure
    summary_path = "paper/figures/cross_dataset_signal_strength.json"
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved numerical summary → {summary_path}")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 120,
    })

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    names = [r["dataset"] for r in rows]
    colors = ["#5B9BD5", "#ED7D31", "#70AD47"]  # blue/orange/green
    x = np.arange(len(names))

    # Panel A: BA bars
    ax = axes[0]
    bas = [r["ba"] for r in rows]
    bars = ax.bar(x, bas, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    for b, v in zip(bars, bas):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Subject-level Balanced Accuracy")
    ax.set_ylim(0.4, 0.85)
    ax.set_title("(a) Fine-tuned LaBraM, subject-level CV")
    ax.legend(loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: η² ratio bars (frozen vs FT side-by-side)
    ax = axes[1]
    width = 0.35
    rfz = [r["ratio_frozen"] for r in rows]
    rft = [r["ratio_ft"] for r in rows]
    b1 = ax.bar(x - width / 2, rfz, width, label="Frozen", color="#A6A6A6", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + width / 2, rft, width, label="Fine-tuned", color="#404040", edgecolor="black", linewidth=0.6)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                    f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel(r"$\eta^{2}_{\mathrm{subject}}\;/\;\eta^{2}_{\mathrm{label}}$")
    ax.set_title(r"(b) Subject vs label variance dominance")
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Cross-dataset signal strength: LaBraM on resting-state EEG",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
    print(f"Saved → {OUT_PDF}")
    print(f"Saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
