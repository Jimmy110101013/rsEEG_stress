"""Build Fig 2 — 2×2 grid of representation-structure panels, one per dataset.

Layout:
  (0,0) EEGMAT   — within × coherent     | callout: dir_cons (3 FMs)
  (0,1) SleepDep — within × incoherent   | callout: dir_cons (3 FMs)
  (1,0) ADFTD    — between × coherent    | callout: FT Δlabel_frac (3 FMs)
  (1,1) Stress   — between × absent      | callout: FT Δlabel_frac (3 FMs)

Each panel shows variance decomposition bars: 3 FMs × {frozen, FT}, with
subject_frac (lower) + label_frac (upper, colored). Callout = the panel's
axis-defining diagnostic metric, printed as 3 values (one per FM).

Output: paper/figures/main/fig2_representation_2x2.{pdf,png}
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT  = REPO / "paper/figures/main"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

FMS        = ["labram", "cbramod", "reve"]
FM_PRETTY  = {"labram": "LaBraM", "cbramod": "CBraMod", "reve": "REVE"}
FM_COLOR   = {"labram": "#1f3a5f", "cbramod": "#B8442C", "reve": "#2E8B57"}

# ── load source tables via canonical accessor ──────────────
from src import results

va_all  = results.source_table("variance_analysis_all")
va_sd   = results.source_table("sleepdep_variance_rsa")
dc_main = results.source_table("f14_within_subject")       # eegmat+stress
dc_sd   = results.source_table("sleepdep_within_subject")


def variance_entry(fm: str, ds: str) -> dict | None:
    key = f"{fm}_{ds}"
    if ds == "sleepdep":
        return va_sd.get(key)
    return va_all.get(key)


def dir_consistency(fm: str, ds: str) -> float | None:
    if ds == "eegmat":
        return dc_main["frozen"]["eegmat"][fm]["dir_consistency"]
    if ds == "sleepdep":
        return dc_sd["frozen"]["sleepdep"][fm]["dir_consistency"]
    return None


# ── panel definitions ──────────────────────────────────────
PANELS = [
    ("eegmat",   "EEGMAT",        "Within-subject × coherent",    "dir_cons"),
    ("sleepdep", "SleepDep",      "Within-subject × incoherent",  "dir_cons"),
    ("adftd",    "ADFTD",         "Between-subject × coherent",   "delta_label"),
    ("stress",   "Stress (DASS)", "Between-subject × absent",     "delta_label"),
]


def draw_panel(ax, ds: str, pretty: str, quadrant: str, metric: str):
    """Variance bars + callout for one dataset."""
    xticks = []
    xlabels = []
    for i, fm in enumerate(FMS):
        entry = variance_entry(fm, ds)
        if entry is None:
            xticks.extend([3*i, 3*i+1]); xlabels.extend([f"{FM_PRETTY[fm]}\nfrz", f"{FM_PRETTY[fm]}\nft"])
            continue
        fl = entry.get("frozen_label_frac")     or 0
        fs = entry.get("frozen_subject_frac")   or 0
        tl = entry.get("ft_label_frac")         or 0
        ts = entry.get("ft_subject_frac")       or 0

        # frozen bar (index 3*i)
        ax.bar(3*i,     fs, width=0.8, color=FM_COLOR[fm], alpha=0.35, edgecolor="k", lw=0.5)
        ax.bar(3*i,     fl, width=0.8, bottom=fs, color=FM_COLOR[fm], alpha=0.95, edgecolor="k", lw=0.5)
        # FT bar (index 3*i+1)
        ax.bar(3*i+1,   ts, width=0.8, color=FM_COLOR[fm], alpha=0.35, edgecolor="k", lw=0.5, hatch="///")
        ax.bar(3*i+1,   tl, width=0.8, bottom=ts, color=FM_COLOR[fm], alpha=0.95, edgecolor="k", lw=0.5, hatch="///")

        # label_frac text on top of stack
        ax.text(3*i,   fs + fl + 2, f"{fl:.1f}", ha="center", fontsize=6.5, fontweight="bold")
        ax.text(3*i+1, ts + tl + 2, f"{tl:.1f}", ha="center", fontsize=6.5, fontweight="bold")

    # Put frz/FT labels inside the bars instead of as xticks, use xticks only for FM centers
    ax.set_xticks([3*i + 0.5 for i in range(len(FMS))])
    ax.set_xticklabels([FM_PRETTY[fm] for fm in FMS], fontsize=8, fontweight="bold")
    for i, fm in enumerate(FMS):
        ax.get_xticklabels()[i].set_color(FM_COLOR[fm])
    # frz/FT tiny labels inside bars (upper region, white on bar)
    for i, fm in enumerate(FMS):
        entry = variance_entry(fm, ds)
        if entry is None: continue
        ax.text(3*i,   4, "frz", ha="center", fontsize=6, color="white", fontweight="bold")
        ax.text(3*i+1, 4, "FT",  ha="center", fontsize=6, color="white", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_xlim(-1, 8)
    ax.set_ylabel("variance explained (%)" if ds in ("eegmat", "adftd") else "")
    ax.set_title(f"{pretty}\n({quadrant})", fontsize=9, pad=4)
    ax.grid(axis="y", alpha=0.25, lw=0.4)

    # callout: axis-defining metric for this quadrant
    if metric == "dir_cons":
        vals = {fm: dir_consistency(fm, ds) for fm in FMS}
        vals_s = "  ".join(f"{FM_PRETTY[fm][:3]}={v:+.3f}" if v is not None else f"{FM_PRETTY[fm][:3]}=—"
                           for fm, v in vals.items())
        ax.text(0.02, 0.97, f"dir_consistency (frozen):\n{vals_s}",
                transform=ax.transAxes, fontsize=6.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8E1", edgecolor="#888", lw=0.4))
    else:  # delta_label
        vals = {}
        for fm in FMS:
            e = variance_entry(fm, ds)
            vals[fm] = (e.get("delta_label_frac") if e else None)
        vals_s = "  ".join(f"{FM_PRETTY[fm][:3]}={v:+.1f}" if v is not None else f"{FM_PRETTY[fm][:3]}=—"
                           for fm, v in vals.items())
        ax.text(0.02, 0.97, f"Δlabel_frac (FT − frz, %):\n{vals_s}",
                transform=ax.transAxes, fontsize=6.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#888", lw=0.4))


# ── figure ─────────────────────────────────────────────────
CM = 1/2.54
W  = 18.3 * CM
fig, axes = plt.subplots(2, 2, figsize=(W, W*0.75), sharey=True)
for ax, (ds, pretty, quadrant, metric) in zip(axes.flat, PANELS):
    draw_panel(ax, ds, pretty, quadrant, metric)

# shared legend
legend = [
    Patch(facecolor="#888", alpha=0.35, edgecolor="k", lw=0.5, label="subject_frac"),
    Patch(facecolor="#888", alpha=0.95, edgecolor="k", lw=0.5, label="label_frac"),
    Patch(facecolor="white", edgecolor="k", lw=0.5, label="frozen"),
    Patch(facecolor="white", edgecolor="k", lw=0.5, hatch="///", label="fine-tuned"),
]
fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=7,
           frameon=False, bbox_to_anchor=(0.5, -0.01))

fig.suptitle("Representation structure across the 4-dataset 2×2 factorial",
             fontsize=10, y=0.995)
plt.tight_layout(rect=[0, 0.04, 1, 0.97])

fig.savefig(OUT / "fig2_representation_2x2.pdf")
fig.savefig(OUT / "fig2_representation_2x2.png")
print(f"saved → {(OUT/'fig2_representation_2x2.pdf').relative_to(REPO)} + .png")
