"""Appendix figure: permutation-null densities for LaBraM FT on
UCSD Stress vs EEGMAT.

Two horizontal density panels sharing the same BA axis. Each panel
overlays the real 3-seed values (vertical ticks) on the 30-permutation
null kernel-density estimate. Demonstrates:
  - Stress: real sits inside null distribution (consistent with no
    exploitable label signal at this cohort size).
  - EEGMAT: real is a clean separation outside the null support
    (consistent with anchored within-subject contrast).

Usage
-----
/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python \
    scripts/build_fig_perm_null_density.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "paper/figures/main/fig_perm_null_density.pdf"
OUT_PNG = OUT.with_suffix(".png")

REAL_GLOBS = {
    "Stress": "results/studies/exp05_stress_feat_multiseed/s*_llrd1.0/summary.json",
    "EEGMAT": "results/studies/exp04_eegmat_feat_multiseed/s*_llrd1.0/summary.json",
}
NULL_GLOBS = {
    "Stress": "results/studies/exp27_paired_null/stress/perm_s*/summary.json",
    "EEGMAT": "results/studies/exp27_paired_null/eegmat/perm_s*/summary.json",
}


def _load(pattern: str) -> np.ndarray:
    paths = sorted(glob.glob(str(REPO / pattern)))
    return np.array([json.load(open(p))["subject_bal_acc"] for p in paths])


def main() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.0), sharex=True)
    for ax, ds in zip(axes, ["Stress", "EEGMAT"]):
        real = _load(REAL_GLOBS[ds])
        null = _load(NULL_GLOBS[ds])

        xs = np.linspace(0.25, 0.85, 400)
        kde = gaussian_kde(null, bw_method=0.35)
        ys = kde(xs)
        ax.fill_between(xs, ys, color="#c0c0c0", alpha=0.5,
                        label=f"Null ($n={len(null)}$)")
        ax.plot(xs, ys, color="#707070", linewidth=0.8)

        for v in real:
            ax.axvline(v, color="#d62728", linewidth=1.5, alpha=0.9)
        ax.plot([], [], color="#d62728", linewidth=1.5,
                label=f"Real ($n={len(real)}$)")

        ax.axvline(0.5, color="#888", linestyle=":", linewidth=0.8)
        ax.set_ylabel(f"{ds}\ndensity", fontsize=9)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", fontsize=8, frameon=False)

        real_m, real_sd = real.mean(), real.std(ddof=1)
        null_m, null_sd = null.mean(), null.std(ddof=1)
        txt = (f"real {real_m:.3f}$\\pm${real_sd:.3f}   "
               f"null {null_m:.3f}$\\pm${null_sd:.3f}")
        ax.text(0.02, 0.85, txt, transform=ax.transAxes,
                fontsize=8, color="#333")

    axes[-1].set_xlabel("Balanced accuracy")
    axes[-1].set_xlim(0.25, 0.85)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
