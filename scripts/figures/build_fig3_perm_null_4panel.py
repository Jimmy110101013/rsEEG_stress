"""Fig 3 — permutation-null histograms across 4 datasets.

Replaces the 2-panel ``fig3_honest_evaluation`` (Stress + EEGMAT) with a
2×2 factorial layout. Rows = CV regime (within-subject paired vs
subject-label), columns = label substrate (neural signature vs
behavioral/state):

                    neural-signal label     behavioral / state label
    within-subject  EEGMAT                  SleepDep
    subject-label   ADFTD                   Stress

The left column (neural-signal) clears the null in both rows; the
right column (behavioral/state) stays inside the null in both rows.
That the column carves the data regardless of row is the visual
argument for the label-substrate axis over the CV-regime axis.

Each panel shows the 30-seed LaBraM FT null distribution (histogram)
with the 3-seed real BA as a red vertical line and the empirical
p-value ``p = (#{null ≥ real} + 1) / (n_null + 1)``.

Sources
-------
- Null BAs: ``results/studies/exp27_paired_null/{ds}/perm_s*/summary.json``
- Real BAs: ``paper/figures/_historical/source_tables/master_frozen_ft_table_v2.json``
  for EEGMAT / ADFTD; ``results/studies/exp_30_sdl_vs_between/tables/fm_performance.json``
  for Stress / SleepDep (per-dataset 3-seed FT under the recipe that matches
  the null chain's training config).

ADFTD null is subject-level permutation (``--permute-level subject``);
others are recording-level. SleepDep null may still be in-flight — the
script uses whatever seeds are complete and annotates the panel with
``n``.

Usage
-----
/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python \
    scripts/figures/build_fig3_perm_null_4panel.py
"""
from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "paper/figures/main/fig3_honest_evaluation_4panel.pdf"
OUT_PNG = OUT.with_suffix(".png")

NULL_ROOTS = {
    "EEGMAT":   "results/studies/exp27_paired_null/eegmat",
    "ADFTD":    "results/studies/exp27_paired_null/adftd",
    "Stress":   "results/studies/exp27_paired_null/stress",
    "SleepDep": "results/studies/exp27_paired_null/sleepdep",
}

# 2×2 factorial, row-major. Rows = CV regime; columns = label substrate.
#   (0,0) within × neural     (0,1) within × behavioral
#   (1,0) subject-label × neural  (1,1) subject-label × behavioral
PANELS = [
    ("EEGMAT",   "rest vs arithmetic → theta/alpha"),
    ("SleepDep", "normal vs sleep-deprived"),
    ("ADFTD",    "AD vs HC → 1/f aperiodic slope"),
    ("Stress",   "DASS-21 score threshold"),
]
ROW_LABELS = ["within-subject\npaired", "subject-label\ntrait/score"]
COL_LABELS = ["neural-signal label", "behavioral / state label"]

# Permutation level per dataset, for the methods annotation
PERM_LEVEL = {
    "EEGMAT":   "recording",
    "ADFTD":    "subject",
    "Stress":   "recording",
    "SleepDep": "recording",
}


def _load_null(ds: str) -> np.ndarray:
    root = REPO / NULL_ROOTS[ds]
    paths = sorted(root.glob("perm_s*/summary.json"))
    return np.array([json.load(open(p))["subject_bal_acc"] for p in paths])


def _real_labram_ft(ds: str) -> tuple[float, float, int]:
    """Return (mean, sd, n) of LaBraM FT BA under the recipe matching
    the null chain. Stress uses best-HP (lr=1e-4) per exp27 chain config;
    EEGMAT/ADFTD use canonical (lr=1e-5); SleepDep uses its canonical
    (lr=1e-5, bs=4)."""
    if ds in {"EEGMAT", "ADFTD"}:
        tab = json.load(open(
            REPO / "paper/figures/_historical/source_tables/master_frozen_ft_table_v2.json"
        ))["table"]
        r = tab["labram"][ds.lower()]
        return float(r["ft_mean"]), float(r["ft_std"]), int(r["ft_n"])
    # Stress + SleepDep: pull from exp_30 fm_performance, matching null recipe
    perf = json.load(open(
        REPO / "results/studies/exp_30_sdl_vs_between/tables/fm_performance.json"
    ))
    rows = [r for r in perf
            if r["mode"] == "ft" and r["fm"] == "labram"
            and r["dataset"] == ds.lower() and r["bal_acc"] is not None]
    bas = [r["bal_acc"] for r in rows]
    if not bas:
        raise RuntimeError(f"No real-FT seeds found for {ds}")
    sd = statistics.stdev(bas) if len(bas) > 1 else 0.0
    return statistics.mean(bas), sd, len(bas)


def _p_value(real: float, null: np.ndarray) -> float:
    k = int(np.sum(null >= real))
    return (k + 1) / (len(null) + 1)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.2),
                             gridspec_kw={"wspace": 0.22, "hspace": 0.60,
                                          "left": 0.11, "right": 0.98,
                                          "top": 0.87, "bottom": 0.10})
    flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for ax, (ds, subtitle) in zip(flat, PANELS):
        try:
            null = _load_null(ds)
            real_m, real_sd, n_real = _real_labram_ft(ds)
        except Exception as e:
            ax.set_title(f"{ds} — unavailable\n{e}", fontsize=8)
            continue

        ax.hist(null, bins=12, color="#BBBBBB", edgecolor="k",
                linewidth=0.4, alpha=0.9)
        ax.axvline(real_m, color="#B8442C", linewidth=2.2,
                   label=f"real FT (n={n_real})")
        ax.axvline(0.5, color="k", linestyle=":", linewidth=0.6,
                   label="chance")

        p = _p_value(real_m, null)
        ax.set_title(
            f"{ds}    real = {real_m:.3f} ± {real_sd:.3f}    p = {p:.2f}\n"
            f"null n = {len(null)}, {PERM_LEVEL[ds]}-level permutation",
            fontsize=8.5, pad=4,
        )
        ax.text(0.02, 0.97, subtitle, transform=ax.transAxes,
                fontsize=7.5, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5",
                          edgecolor="#AAA", lw=0.4))
        ax.set_xlabel("Subject-level BA", fontsize=8)
        ax.set_ylabel("# permutation seeds", fontsize=8)
        ax.set_xlim(0.35, 0.80)
        ax.tick_params(labelsize=7.5)

    # Factorial axis labels: columns on top, rows on left
    for col, label in enumerate(COL_LABELS):
        x = axes[0, col].get_position().x0 + axes[0, col].get_position().width / 2
        fig.text(x, 0.935, label, ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color="#333")
    for row, label in enumerate(ROW_LABELS):
        y = axes[row, 0].get_position().y0 + axes[row, 0].get_position().height / 2
        fig.text(0.015, y, label, ha="left", va="center",
                 fontsize=9.5, fontweight="bold", rotation=90, color="#333")

    fig.suptitle(
        "Fig 3 — Permutation null (2×2 factorial): label-substrate column "
        "carves the data regardless of CV-regime row.",
        fontsize=10, y=0.995,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
