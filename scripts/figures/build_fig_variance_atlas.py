"""Build the variance-atlas convergent-evidence figure for §4.1.

Three panels, 12 (FM × dataset) cells:

  A. Stacked horizontal bar — pooled variance fractions (Label / Subject / Residual)
  B. Scatter — RSA subject-r vs label-r with y=x diagonal
  C. Forest plot — cluster-bootstrap 95% CI on Subject/Label variance ratio
                   (subject-level resample, log-scale ratio, reference line at 1)

Data sources:
  - Variance fractions (Panel A, Panel C point estimate):
        paper/figures/source_tables/variance_analysis_all.json
  - RSA correlations (Panel B):
        results/studies/exp06_fm_task_fitness/fitness_metrics_full.json
  - Cluster-bootstrap CI (Panel C error bars):
        computed here by resampling subjects with replacement and recomputing
        SS_subject_within_label / SS_label per iteration.

Stress frozen cache preserves the mixed-label subject filter used elsewhere
(drops pids with recordings in both DASS classes -> 55 rec / 14 subj).

Usage
-----
/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python \
    scripts/build_fig_variance_atlas.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.variance_analysis import nested_ss, cluster_bootstrap  # noqa: E402
# Stress/ADFTD/TDBRAIN use nested (subject-in-label); EEGMAT is within-subject
# so the decomposition is crossed (subject × label).
from scripts.analysis.analyze_eegmat import crossed_decomposition  # noqa: E402

CACHE = REPO / "results/features_cache"
VAR_JSON = REPO / "paper/figures/source_tables/variance_analysis_all.json"
FIT_JSON = REPO / "results/studies/exp06_fm_task_fitness/fitness_metrics_full.json"
OUT_DIR = REPO / "paper/figures/main"
OUT_STEM = "fig_variance_atlas"
OUT_SRC_TABLE = REPO / "paper/figures/source_tables/variance_atlas_bootstrap.json"

CH_TAG = {"stress": "30ch", "adftd": "19ch", "tdbrain": "19ch", "eegmat": "19ch"}
MODELS = ["labram", "cbramod", "reve"]
DATASETS = ["stress", "eegmat", "adftd", "tdbrain"]
DATASET_PRINT = {"stress": "Stress", "eegmat": "EEGMAT",
                 "adftd": "ADFTD", "tdbrain": "TDBRAIN"}
MODEL_PRINT = {"labram": "LaBraM", "cbramod": "CBraMod", "reve": "REVE"}

CB = sns.color_palette("colorblind")
FM_COLOR = {"LaBraM": CB[0], "CBraMod": CB[1], "REVE": CB[2]}
DS_MARKER = {"Stress": "o", "EEGMAT": "s", "ADFTD": "^", "TDBRAIN": "D"}


def load_frozen(model: str, dataset: str):
    npz = CACHE / f"frozen_{model}_{dataset}_{CH_TAG[dataset]}.npz"
    d = np.load(npz, allow_pickle=True)
    f, y, p = d["features"], d["labels"], d["patient_ids"]
    if dataset == "stress":
        mixed = {pid for pid in np.unique(p)
                 if len(np.unique(y[p == pid])) > 1}
        keep = np.array([pid not in mixed for pid in p])
        f, y, p = f[keep], y[keep], p[keep]
    return f, y, p


def ratio_stat_nested(f, s, y):
    """Subject/Label variance ratio from the nested decomposition."""
    ss = nested_ss(f, s, y)
    lab = float(ss["label"].sum())
    sub = float(ss["subject_within_label"].sum())
    if lab <= 0:
        return float("nan")
    return sub / lab


def ratio_stat_crossed(f, s, y):
    """Subject/Label variance ratio from the crossed decomposition (EEGMAT)."""
    cd = crossed_decomposition(f, s, y)
    return cd["ratio_subject_to_label"]


def compute_bootstrap_ratios(n_boot: int = 1000) -> dict:
    """Per-cell cluster-bootstrap CI on Subject/Label ratio. Cached to JSON."""
    if OUT_SRC_TABLE.exists():
        cached = json.loads(OUT_SRC_TABLE.read_text())
        if cached.get("n_boot") == n_boot and len(cached.get("cells", {})) == 12:
            print(f"  Loaded bootstrap cache from {OUT_SRC_TABLE.name}")
            return cached["cells"]
    print(f"  Computing cluster bootstrap (n_boot={n_boot}) on 12 cells...")
    out = {}
    for m in MODELS:
        for d in DATASETS:
            f, y, p = load_frozen(m, d)
            stat = ratio_stat_crossed if d == "eegmat" else ratio_stat_nested
            boot = cluster_bootstrap(
                f, p, y, stat,
                n_boot=n_boot, seed=0, log_transform=True,
            )
            key = f"{m}_{d}"
            out[key] = {
                "point": boot["point"],
                "geo_mean": boot["mean"],
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
                "n_valid": boot["n_valid"],
            }
            print(f"    {key:22s}  point={boot['point']:7.2f}  "
                  f"CI=[{boot['ci_low']:7.2f}, {boot['ci_high']:7.2f}]  "
                  f"n_valid={boot['n_valid']}")
    OUT_SRC_TABLE.write_text(json.dumps(
        {"n_boot": n_boot, "cells": out}, indent=2))
    print(f"  Cached to {OUT_SRC_TABLE.relative_to(REPO)}")
    return out


def load_cell_rows():
    """Assemble a list of 12 cells with all metrics needed, ordered for display."""
    va = json.loads(VAR_JSON.read_text())
    fit = json.loads(FIT_JSON.read_text())["per_model_dataset"]
    boot = compute_bootstrap_ratios()

    rows = []
    for d in DATASETS:
        for m in MODELS:
            key = f"{m}_{d}"
            v = va[key]
            fr = fit[key]["frozen"]
            b = boot[key]
            lab = v["frozen_label_frac"]
            subj = v["frozen_subject_frac"]
            rows.append({
                "model": MODEL_PRINT[m],
                "dataset": DATASET_PRINT[d],
                "cell_id": f"{MODEL_PRINT[m]} × {DATASET_PRINT[d]}",
                "label_pct": lab,
                "subject_pct": subj,
                "residual_pct": max(0.0, 100.0 - lab - subj),
                "rsa_label_r": fr["rsa_label_r"],
                "rsa_subject_r": fr["rsa_subject_r"],
                "ratio_point": b["point"],
                "ratio_ci_low": b["ci_low"],
                "ratio_ci_high": b["ci_high"],
            })
    return rows


# -----------------------------------------------------------------------------
# Panels
# -----------------------------------------------------------------------------
def panel_a(ax, rows):
    labels = [r["cell_id"] for r in rows]
    y_pos = np.arange(len(rows))
    label_pct = np.array([r["label_pct"] for r in rows])
    subj_pct = np.array([r["subject_pct"] for r in rows])
    resid_pct = np.array([r["residual_pct"] for r in rows])

    ax.barh(y_pos, label_pct, color="#d62728",
            label="Label", edgecolor="white", linewidth=0.4)
    ax.barh(y_pos, subj_pct, left=label_pct, color="#1f77b4",
            label="Subject", edgecolor="white", linewidth=0.4)
    ax.barh(y_pos, resid_pct, left=label_pct + subj_pct,
            color="#c7c7c7", label="Residual",
            edgecolor="white", linewidth=0.4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Variance fraction (\\%)")
    ax.invert_yaxis()
    ax.set_title("A. Pooled variance decomposition",
                 fontsize=10, loc="left", pad=8)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)


def panel_b(ax, rows):
    max_r = max(max(abs(r["rsa_subject_r"]) for r in rows),
                max(abs(r["rsa_label_r"]) for r in rows))
    lim = max(0.32, max_r * 1.1)

    # Diagonal (subject = label).
    ax.plot([-lim, lim], [-lim, lim], ls="--", color="#888888",
            linewidth=0.8, zorder=0)
    # Axes at zero.
    ax.axhline(0, color="#cccccc", linewidth=0.5, zorder=0)
    ax.axvline(0, color="#cccccc", linewidth=0.5, zorder=0)

    for r in rows:
        ax.scatter(
            r["rsa_label_r"], r["rsa_subject_r"],
            marker=DS_MARKER[r["dataset"]],
            color=FM_COLOR[r["model"]],
            edgecolor="black", linewidth=0.5, s=56, zorder=3,
        )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("RSA Spearman $r$ (label)")
    ax.set_ylabel("RSA Spearman $r$ (subject)")
    ax.set_title("B. Rank-based RSA", fontsize=10, loc="left", pad=8)
    ax.set_aspect("equal")

    # Two-part legend: color = model, marker = dataset.
    model_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   color=FM_COLOR[m], markeredgecolor="black",
                   markeredgewidth=0.5, markersize=7, label=m)
        for m in ["LaBraM", "CBraMod", "REVE"]
    ]
    ds_handles = [
        plt.Line2D([0], [0], marker=DS_MARKER[d], linestyle="",
                   color="#888888", markeredgecolor="black",
                   markeredgewidth=0.5, markersize=7, label=d)
        for d in ["Stress", "EEGMAT", "ADFTD", "TDBRAIN"]
    ]
    leg1 = ax.legend(handles=model_handles, loc="upper left",
                     fontsize=7, frameon=False, title="Model", title_fontsize=7)
    ax.add_artist(leg1)
    ax.legend(handles=ds_handles, loc="lower right",
              fontsize=7, frameon=False, title="Dataset", title_fontsize=7)


def panel_c(ax, rows):
    labels = [r["cell_id"] for r in rows]
    y_pos = np.arange(len(rows))
    pts = np.array([r["ratio_point"] for r in rows])
    lo = np.array([r["ratio_ci_low"] for r in rows])
    hi = np.array([r["ratio_ci_high"] for r in rows])
    colors = [FM_COLOR[r["model"]] for r in rows]

    ax.axvline(1.0, color="#888888", linestyle="--", linewidth=0.8, zorder=0)
    for i, (p, l, h, c) in enumerate(zip(pts, lo, hi, colors)):
        ax.plot([l, h], [i, i], color=c, linewidth=2.0, zorder=2)
        ax.plot([p], [i], "o", color=c, markeredgecolor="black",
                markeredgewidth=0.5, markersize=5, zorder=3)

    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    # Sensible log-scale window bracketing everything.
    ax.set_xlim(max(0.3, np.nanmin(lo) / 1.5),
                min(1e5, np.nanmax(hi) * 1.5))
    ax.set_xlabel("Subject / Label variance ratio (log)")
    ax.set_title("C. Cluster-bootstrap 95\\% CI",
                 fontsize=10, loc="left", pad=8)
    ax.grid(axis="x", which="both", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    rows = load_cell_rows()

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.5,
        "text.usetex": False,
        "axes.titleweight": "bold",
    })
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.15],
                                          "wspace": 0.45})
    panel_a(axes[0], rows)
    panel_b(axes[1], rows)
    panel_c(axes[2], rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"{OUT_STEM}.{ext}"
        fig.savefig(out, bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
        print(f"  Wrote {out.relative_to(REPO)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
