"""Fig 8c: Non-FM baselines match FM ceiling on Stress (F20).

Bar chart: ShallowConvNet + EEGNet vs 3 FM best-HP FT, all 3-seed on Stress.
Supports the "task property, not model property" claim in §7.

Sources:
- Non-FM: results/studies/exp15_nonfm_baselines/sweep/{shallowconvnet_lr1e-4,eegnet_lr5e-4}_s*/summary.json
- FM FT best-HP: findings.md F05 (LaBraM 0.524±0.008, CBraMod 0.548±0.026, REVE 0.577±0.041)

Output: paper/figures/main/fig8c_non_fm_baselines.pdf
"""
import json
import statistics as st
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,
})


def agg_seeds(cfg_prefix):
    base = Path("results/studies/exp15_nonfm_baselines/sweep")
    vals = []
    for d in base.iterdir():
        if not d.name.startswith(cfg_prefix):
            continue
        # Only keep the 3-seed runs (s42, s123, s2024)
        if not any(d.name.endswith(s) for s in ["_s42", "_s123", "_s2024"]):
            continue
        with open(d / "summary.json") as f:
            s = json.load(f)
        ba = s.get("mean_bal_acc") or s.get("bal_acc")
        if ba is None:
            for k, v in s.items():
                if isinstance(v, (int, float)) and "bal" in k.lower():
                    ba = v
                    break
        if ba is not None:
            vals.append(ba)
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


scn = agg_seeds("shallowconvnet_lr1e-4")
egn = agg_seeds("eegnet_lr5e-4")

# FM best-HP numbers from F05 (findings.md), sample-std convention (n-1 divisor)
fm = {
    "LaBraM FT":  (0.524, 0.010),
    "CBraMod FT": (0.548, 0.031),
    "REVE FT":    (0.577, 0.051),
}

models = ["ShallowConvNet", "EEGNet"] + list(fm.keys())
means  = [scn[0], egn[0]] + [fm[k][0] for k in fm]
stds   = [scn[1], egn[1]] + [fm[k][1] for k in fm]
is_fm  = [False, False, True, True, True]

fig, ax = plt.subplots(figsize=(7.2, 4.0))

x = np.arange(len(models))
colors = ["#4c78a8" if not f else "#f28e2b" for f in is_fm]
bars = ax.bar(
    x, means, yerr=stds, color=colors, edgecolor="black",
    linewidth=0.8, capsize=5, error_kw={"elinewidth": 1.2},
)

for b, m, s in zip(bars, means, stds):
    ax.text(
        b.get_x() + b.get_width() / 2,
        m + s + 0.01,
        f"{m:.3f}",
        ha="center",
        fontsize=9,
    )

# Chance line
ax.axhline(0.5, color="#666", linewidth=0.8, linestyle="--", alpha=0.7)
ax.text(len(models) - 0.5, 0.505, "chance", fontsize=8, color="#666")

# Highlight FM range
fm_low = min(fm[k][0] - fm[k][1] for k in fm)
fm_high = max(fm[k][0] + fm[k][1] for k in fm)
ax.axhspan(fm_low, fm_high, alpha=0.10, color="#f28e2b", zorder=0,
           label=f"FM FT range [{fm_low:.3f}, {fm_high:.3f}]")

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha="right")
ax.set_ylabel("Balanced Accuracy (3-seed mean ± 1σ)")
ax.set_title("ShallowConvNet from scratch matches FM ceiling on Stress\n(task property, not model property — see §7 power-floor caveat)")
ax.set_ylim(0.35, 0.70)
ax.grid(True, axis="y", alpha=0.25)

from matplotlib.patches import Patch
handles = [
    Patch(facecolor="#4c78a8", edgecolor="black", label="Non-FM (from scratch)"),
    Patch(facecolor="#f28e2b", edgecolor="black", label="FM FT (best HP)"),
]
ax.legend(handles=handles, loc="upper left", framealpha=0.95)

plt.tight_layout()
out = "paper/figures/main/fig8c_non_fm_baselines.pdf"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"wrote {out}")
print(f"ShallowConvNet: {scn[0]:.3f}±{scn[1]:.3f} (n={scn[2]})")
print(f"EEGNet:         {egn[0]:.3f}±{egn[1]:.3f} (n={egn[2]})")
