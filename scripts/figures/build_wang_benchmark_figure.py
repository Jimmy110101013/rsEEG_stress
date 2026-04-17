"""Supplementary figure: Wang 2025 benchmark reference.

Three-way comparison on Stress dataset:
  (a) exp19: Wang setting — 82 rec from comprehensive_labels_stress.csv,
      --max-duration 400, trial-level StratifiedKFold(5), per-rec DASS
  (b) exp18: our 70-rec trial-level CV, per-rec DASS
  (c) hp_sweep 20260410_dass: our 70-rec subject-level StratifiedGroupKFold(5)

Wang et al. (arXiv:2505.23042) report 0.90 BA on setting (a). We cannot
reproduce that number: our trial BA on the same 82 rec is ~0.6, below even
our own 70-rec trial. Subject CV pulls all three FMs toward the same floor
(0.52–0.58), confirming the inflation is a CV-protocol artifact plus possible
additional Wang-specific confounders (e.g. different model, stride, or data
cleaning not documented in arXiv:2505.23042).

Output: paper/figures/supplementary/fig_wang_benchmark.{pdf,png}
"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,
})

SEEDS = [42, 123, 2024]

HP = {
    "labram":  "encoderlrscale1.0_lr1e-4",
    "cbramod": "encoderlrscale0.1_lr1e-5",
    "reve":    "encoderlrscale0.1_lr3e-5",
}


def load(paths):
    vals = []
    for p in paths:
        if not Path(p).exists():
            continue
        with open(p) as fh:
            d = json.load(fh)
        ba = d.get("subject_bal_acc") or d.get("bal_acc")
        if ba is not None:
            vals.append(float(ba))
    return vals


def msd(v):
    m = mean(v) if v else np.nan
    s = stdev(v) if len(v) > 1 else 0.0  # sample std
    return m, s, len(v)


bars = {}  # (model, condition) -> (mean, std, n)
for m in ["labram", "cbramod", "reve"]:
    # (a) Wang 82-rec trial
    p82 = [f"results/studies/exp19_wang_82rec_trial/{m}_s{s}/summary.json" for s in SEEDS]
    bars[(m, "wang82_trial")] = msd(load(p82))
    # (b) Our 70-rec trial
    p70t = [f"results/studies/exp18_trial_dass_multiseed/{m}_s{s}/summary.json" for s in SEEDS]
    bars[(m, "ours70_trial")] = msd(load(p70t))
    # (c) Our 70-rec subject
    p70s = [f"results/hp_sweep/20260410_dass/{m}/{m}_{HP[m]}_s{s}/summary.json" for s in SEEDS]
    bars[(m, "ours70_subject")] = msd(load(p70s))
    for cond in ["wang82_trial", "ours70_trial", "ours70_subject"]:
        mm, ss, nn = bars[(m, cond)]
        print(f"  {m:<8} {cond:<16}: {mm:.3f} ± {ss:.3f} (n={nn})")

models_label = ["LaBraM", "REVE", "CBraMod"]
keys = ["labram", "reve", "cbramod"]

conds = ["wang82_trial", "ours70_trial", "ours70_subject"]
cond_label = {
    "wang82_trial":   "Wang 2025 setting\n(82 rec, 400s filter, trial CV)",
    "ours70_trial":   "Our 70 rec\n(DSS-paired, trial CV)",
    "ours70_subject": "Our 70 rec\n(DSS-paired, subject CV)",
}
cond_color = {
    "wang82_trial":   "#e76f51",
    "ours70_trial":   "#d62728",
    "ours70_subject": "#4c78a8",
}

x = np.arange(len(keys))
w = 0.26

fig, ax = plt.subplots(figsize=(9.5, 5.0))

for i, cond in enumerate(conds):
    means = [bars[(k, cond)][0] for k in keys]
    stds  = [bars[(k, cond)][1] for k in keys]
    offset = (i - 1) * w
    bar = ax.bar(x + offset, means, w, yerr=stds, capsize=4,
                 color=cond_color[cond], edgecolor="black", linewidth=0.7,
                 error_kw={"ecolor": "#222", "lw": 0.9},
                 label=cond_label[cond])
    for rect, v, sd in zip(bar, means, stds):
        if np.isnan(v):
            continue
        ax.text(rect.get_x() + rect.get_width() / 2, v + sd + 0.012,
                f"{v:.3f}", ha="center", fontsize=8)

# Wang's reported 0.90 line
ax.axhline(0.90, color="#7a0", linewidth=1.4, linestyle="--", alpha=0.8)
ax.text(len(keys) - 0.4, 0.905, "Wang 2025 reported 0.90 (LaBraM FT)",
        fontsize=9, color="#4a0", ha="right", va="bottom", fontweight="bold")

ax.axhline(0.5, color="#666", linewidth=0.8, linestyle=":", alpha=0.7)
ax.text(len(keys) - 0.5, 0.51, "chance", fontsize=8, color="#666")

ax.set_xticks(x)
ax.set_xticklabels(models_label, fontsize=11)
ax.set_ylabel("Balanced Accuracy on Stress")
ax.set_title(
    "Wang 2025 benchmark reference (supplementary)\n"
    "Same per-rec DASS labels; three settings differ in recording set × CV protocol"
)
ax.set_ylim(0.35, 1.00)
ax.grid(True, axis="y", alpha=0.25)
ax.legend(loc="upper right", framealpha=0.95, fontsize=8.5, ncol=1)

fig.text(
    0.5, -0.02,
    "All FMs use reference-paper trial recipe (lr=1e-5 LaBraM/CBraMod, 3e-5 REVE; 75% overlap on minority class). "
    "3 seeds (42/123/2024), sample std. Wang's 0.90 is far above any of our reproductions, "
    "suggesting additional Wang-specific factors beyond the 82-rec × trial-CV combination.",
    ha="center", fontsize=7.8, style="italic", color="#555",
)

plt.tight_layout()
out_pdf = "paper/figures/supplementary/fig_wang_benchmark.pdf"
Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
out_png = out_pdf.replace(".pdf", ".png")
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"\nwrote {out_pdf}")
