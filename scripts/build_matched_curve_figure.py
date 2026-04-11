"""Generate the N-controlled matched-subsample curve figure for the paper.

Reads precomputed matched-subsample results from
`paper/figures/variance_analysis_matched.json` (produced by
`scripts/run_variance_analysis.py --matched all --matched-n-draws 100
--permute-labels-check`) and the canonical full-N values from
`paper/figures/variance_analysis.json`, and produces a 2-panel figure:

  (a) ADFTD: pooled label fraction vs matched N, frozen + FT curves with
      95% CI shading, Stress full-N reference line, permutation null as
      dashed reference, full-N canonical point as marker.

  (b) TDBRAIN: same layout, different y-range to show fold-drift dilution.

The message of the figure: under N-control, ADFTD's FT rewrite (+5 pp)
and TDBRAIN's FT dilution (−1.5 pp) are both N-invariant and clearly
exceed the permutation null, while Stress has no FT effect at any N.

Run from project root:
    /raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python scripts/build_matched_curve_figure.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

CANONICAL_JSON = "paper/figures/variance_analysis.json"
MATCHED_JSON = "paper/figures/variance_analysis_matched.json"
OUT_PDF = "paper/figures/matched_subsample_curves.pdf"
OUT_PNG = "paper/figures/matched_subsample_curves.png"

# Styling
FROZEN_COLOR = "#1f77b4"   # blue
FT_COLOR = "#d62728"       # red
STRESS_COLOR = "#7f7f7f"   # grey
NULL_ALPHA = 0.35


def _load(path: str) -> dict:
    if not os.path.isfile(path):
        sys.exit(f"missing: {path}")
    with open(path) as f:
        return json.load(f)


def _collect_rungs(matched: dict, source: str) -> list[dict]:
    """Extract all matched regimes that target `source`, ordered by total N.

    Returns a list of dicts with keys: N, frozen_mean, frozen_std, frozen_lo,
    frozen_hi, ft_mean, ft_std, ft_lo, ft_hi, delta_mean, frac_positive,
    regime_name.
    """
    regimes = matched["regimes"]
    permuted = matched.get("permuted_regimes", {})
    rows = []
    for name, r in regimes.items():
        if r.get("source") != source:
            continue
        # Skip the Stress-ratio ADFTD variant for the main curve — keep only
        # rungs that preserve the source dataset's own class ratio, so all
        # points sit on a consistent curve. Stress-ratio variant appears in
        # a side annotation.
        if "stressRatio" in name:
            continue
        n_total = sum(r["config"]["n_per_label"].values())
        row = {
            "regime_name": name,
            "N": n_total,
            "frozen_mean": r["pooled_label_fraction_frozen"]["mean"],
            "frozen_std": r["pooled_label_fraction_frozen"]["std"],
            "frozen_lo": r["pooled_label_fraction_frozen"]["ci_low"],
            "frozen_hi": r["pooled_label_fraction_frozen"]["ci_high"],
            "ft_mean": r["pooled_label_fraction_ft"]["mean"],
            "ft_std": r["pooled_label_fraction_ft"]["std"],
            "ft_lo": r["pooled_label_fraction_ft"]["ci_low"],
            "ft_hi": r["pooled_label_fraction_ft"]["ci_high"],
            "delta_mean": r["delta_pooled_label_fraction"]["mean"],
            "delta_std": r["delta_pooled_label_fraction"]["std"],
            "frac_positive": r["delta_pooled_label_fraction"]["frac_positive"],
        }
        # Paired null (permuted) values if available
        if name in permuted:
            pn = permuted[name]
            row["null_frozen_mean"] = pn["pooled_label_fraction_frozen"]["mean"]
            row["null_ft_mean"] = pn["pooled_label_fraction_ft"]["mean"]
            row["null_delta_mean"] = pn["delta_pooled_label_fraction"]["mean"]
        rows.append(row)
    rows.sort(key=lambda r: r["N"])
    return rows


def _canonical_point(canonical: dict, source: str) -> tuple[int, float, float]:
    """(N_full, frozen_pooled_label_fraction, ft_pooled_label_fraction)."""
    ds = canonical["datasets"][source]["analysis"]
    fz = ds["frozen"]["pooled_fractions"]["label"]
    ft = ds["ft_pooled"]["pooled_fractions"]["label"]
    n_full = ds["frozen"]["n_subjects"]
    return n_full, fz, ft


def _plot_panel(ax, rows, canonical_point, stress_fz, stress_ft,
                title_prefix, effect_direction):
    """Plot one dataset panel."""
    N = np.array([r["N"] for r in rows])
    fz_mean = np.array([r["frozen_mean"] for r in rows]) * 100
    fz_lo = np.array([r["frozen_lo"] for r in rows]) * 100
    fz_hi = np.array([r["frozen_hi"] for r in rows]) * 100
    ft_mean = np.array([r["ft_mean"] for r in rows]) * 100
    ft_lo = np.array([r["ft_lo"] for r in rows]) * 100
    ft_hi = np.array([r["ft_hi"] for r in rows]) * 100

    # Full-N canonical point
    n_full, fz_full, ft_full = canonical_point
    N_ext = np.append(N, n_full)
    fz_ext = np.append(fz_mean, fz_full * 100)
    ft_ext = np.append(ft_mean, ft_full * 100)

    # Sort extended series (canonical N may be > all matched rungs)
    order = np.argsort(N_ext)
    N_ext_s = N_ext[order]
    fz_ext_s = fz_ext[order]
    ft_ext_s = ft_ext[order]

    # Shaded CI only on matched rungs (canonical has no CI in this JSON)
    ax.fill_between(N, fz_lo, fz_hi, color=FROZEN_COLOR, alpha=0.20,
                    label=None)
    ax.fill_between(N, ft_lo, ft_hi, color=FT_COLOR, alpha=0.20, label=None)

    # Curves through matched rungs only (dashed tail to canonical N)
    ax.plot(N, fz_mean, "o-", color=FROZEN_COLOR, lw=2.0, ms=6,
            label="Frozen (observed)", zorder=5)
    ax.plot(N, ft_mean, "o-", color=FT_COLOR, lw=2.0, ms=6,
            label="FT (observed)", zorder=5)

    # Tail: connect last matched rung to canonical full-N point
    N_last = N[-1]
    if n_full > N_last:
        ax.plot([N_last, n_full], [fz_mean[-1], fz_full * 100], "--",
                color=FROZEN_COLOR, lw=1.2, alpha=0.7, zorder=4)
        ax.plot([N_last, n_full], [ft_mean[-1], ft_full * 100], "--",
                color=FT_COLOR, lw=1.2, alpha=0.7, zorder=4)
    ax.plot([n_full], [fz_full * 100], marker="D", color=FROZEN_COLOR,
            ms=9, mec="k", mew=1.0, label="Frozen (full N)", zorder=6)
    ax.plot([n_full], [ft_full * 100], marker="D", color=FT_COLOR,
            ms=9, mec="k", mew=1.0, label="FT (full N)", zorder=6)

    # Permutation null (dashed, faded)
    if all("null_frozen_mean" in r for r in rows):
        null_fz = np.array([r["null_frozen_mean"] for r in rows]) * 100
        null_ft = np.array([r["null_ft_mean"] for r in rows]) * 100
        ax.plot(N, null_fz, ":", color=FROZEN_COLOR, lw=1.5,
                alpha=NULL_ALPHA + 0.15,
                label="Frozen (permuted null)", zorder=3)
        ax.plot(N, null_ft, ":", color=FT_COLOR, lw=1.5,
                alpha=NULL_ALPHA + 0.15,
                label="FT (permuted null)", zorder=3)

    # Stress reference line — show both frozen and FT (they're basically equal)
    ax.axhline(stress_fz * 100, color=STRESS_COLOR, ls="-.", lw=1.5,
               alpha=0.7, zorder=2,
               label=f"Stress (N=17, frozen≈FT≈{stress_fz*100:.1f}%)")

    # Labels + grid
    ax.set_xlabel("N (matched subject count)", fontsize=11)
    ax.set_ylabel("Pooled label fraction (%)", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlim(left=0)

    # Title with effect summary
    avg_delta = np.mean([r["delta_mean"] for r in rows]) * 100
    ax.set_title(f"{title_prefix}\n"
                 f"FT effect {effect_direction} "
                 f"(Δ̄ = {avg_delta:+.2f} pp across rungs)",
                 fontsize=11, loc="left")


def main():
    canonical = _load(CANONICAL_JSON)
    matched = _load(MATCHED_JSON)

    # Stress reference from canonical
    stress_n_full, stress_fz, stress_ft = _canonical_point(canonical, "Stress")

    adftd_rows = _collect_rungs(matched, "ADFTD")
    adftd_canonical = _canonical_point(canonical, "ADFTD")
    tdbrain_rows = _collect_rungs(matched, "TDBRAIN")
    tdbrain_canonical = _canonical_point(canonical, "TDBRAIN")

    print(f"ADFTD rungs: {[r['regime_name'] for r in adftd_rows]}")
    print(f"TDBRAIN rungs: {[r['regime_name'] for r in tdbrain_rows]}")
    print(f"Stress canonical: N={stress_n_full}  "
          f"frozen={stress_fz*100:.2f}%  FT={stress_ft*100:.2f}%")
    print(f"ADFTD canonical: N={adftd_canonical[0]}  "
          f"frozen={adftd_canonical[1]*100:.2f}%  FT={adftd_canonical[2]*100:.2f}%")
    print(f"TDBRAIN canonical: N={tdbrain_canonical[0]}  "
          f"frozen={tdbrain_canonical[1]*100:.2f}%  FT={tdbrain_canonical[2]*100:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=False)

    _plot_panel(
        axes[0], adftd_rows, adftd_canonical, stress_fz, stress_ft,
        title_prefix="(a) ADFTD (AD vs HC)  —  injection mode",
        effect_direction="INJECTS label signal",
    )
    axes[0].legend(loc="upper right", fontsize=7.5, framealpha=0.92,
                   ncol=1, handlelength=2.0)

    _plot_panel(
        axes[1], tdbrain_rows, tdbrain_canonical, stress_fz, stress_ft,
        title_prefix="(b) TDBRAIN (MDD vs HC)  —  erosion mode",
        effect_direction="ERODES label signal",
    )
    # Place legend in lower-right of panel (b) so it doesn't cover the data:
    # the curves are descending and live in the upper-left of the panel.
    axes[1].legend(loc="upper right", fontsize=7.5, framealpha=0.92,
                   ncol=1, handlelength=2.0)
    # Make sure y starts at 0 in both panels for visual fairness.
    for ax in axes:
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Pooled label fraction under N-controlled subsampling "
        "(100 draws/rung, frozen vs fine-tuned LaBraM)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
    print(f"\nSaved → {OUT_PDF}")
    print(f"Saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
