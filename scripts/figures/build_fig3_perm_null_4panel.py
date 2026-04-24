"""Fig 3 — permutation-null histograms across 4 datasets.

Replaces the 2-panel ``fig3_honest_evaluation`` (Stress + EEGMAT) with a
2×2 factorial layout. Rows = CV regime (within-subject paired vs
subject-label trait), columns = task-substrate alignment (strong vs
weak):

                    strong-aligned task     weak-aligned task
    within-subject  EEGMAT                  SleepDep
    subject-label   ADFTD                   Stress

The strong-aligned column (labels with a canonical neural signature the
FM learned during pretraining) clears the null in both rows; the
weak-aligned column (behavioral / state summary labels without a stable
EEG substrate) stays inside the null in both rows. That the column
carves the data regardless of row is the visual argument for task-
substrate alignment as the primary axis over CV regime.

Each panel shows the 30-seed LaBraM FT null distribution (histogram)
with the 3-seed real BA as a red vertical line and the empirical
p-value ``p = (#{null ≥ real} + 1) / (n_null + 1)``.

Sources
-------
- Null BAs: ``results/studies/exp27_paired_null/{ds}/perm_s*/summary.json``
- Real BAs: ``results/final/source_tables/master_frozen_ft_table_v2.json``
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

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import results  # noqa: E402

OUT = REPO / "paper/figures/main/fig3_honest_evaluation_4panel.pdf"
OUT_PNG = OUT.with_suffix(".png")

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
COL_LABELS = ["strong-aligned task", "weak-aligned task"]

# Permutation level per dataset, for the methods annotation
PERM_LEVEL = {
    "EEGMAT":   "recording",
    "ADFTD":    "subject",
    "Stress":   "recording",
    "SleepDep": "recording",
}


def _load_null(ds: str) -> np.ndarray:
    return np.array([s["subject_bal_acc"]
                     for s in results.perm_null_summaries(ds)])


def _real_labram_ft(ds: str) -> tuple[float, float, int]:
    """LaBraM FT BA under the recipe matching exp27's null chain.

    Handled by src.results.labram_ft_ba_null_matched, which hides the
    EEGMAT/ADFTD (master table) vs Stress/SleepDep (exp_30 fm_performance)
    bifurcation in the underlying storage.
    """
    return results.labram_ft_ba_null_matched(ds)


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
        "Fig 3 — Permutation null (2×2 factorial): task-substrate alignment "
        "column carves the data regardless of CV-regime row.",
        fontsize=10, y=0.995,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
