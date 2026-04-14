"""Fig 2: Trial-level vs subject-level CV on the same 3 FMs (Stress).

Same data, same architecture, two partition rules — isolates the gap as
protocol-driven rather than model-driven.

Data source (all per-recording DASS, 3 seeds, sample std):
  Trial  : results/studies/exp18_trial_dass_multiseed/{model}_s{seed}/summary.json
  Subject: results/hp_sweep/20260410_dass/{model}/{model}_{hp}_s{seed}/summary.json
           (F05 best-HP — labram lr1e-4/enc1.0, cbramod lr1e-5/enc0.1, reve lr3e-5/enc0.1)

Output: paper/figures/main/fig2_cv_gap.pdf
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


def load_seeds(paths):
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


SEEDS = [42, 123, 2024]

MODEL_HP = {
    "labram":  "encoderlrscale1.0_lr1e-4",
    "cbramod": "encoderlrscale0.1_lr1e-5",
    "reve":    "encoderlrscale0.1_lr3e-5",
}

trial_ba = {}
subj_ba = {}
for m in ["labram", "cbramod", "reve"]:
    trial_paths = [f"results/studies/exp18_trial_dass_multiseed/{m}_s{s}/summary.json"
                   for s in SEEDS]
    subj_paths = [f"results/hp_sweep/20260410_dass/{m}/{m}_{MODEL_HP[m]}_s{s}/summary.json"
                  for s in SEEDS]
    trial_ba[m] = load_seeds(trial_paths)
    subj_ba[m] = load_seeds(subj_paths)
    print(f"  {m}: trial n={len(trial_ba[m])}, subject n={len(subj_ba[m])}")


def msd(v):
    m = mean(v) if v else 0.0
    s = stdev(v) if len(v) > 1 else 0.0  # sample std (n-1)
    return m, s


models_label = ["LaBraM", "REVE", "CBraMod"]  # order matches original figure
keys = ["labram", "reve", "cbramod"]

trial_m = []; trial_s = []
subj_m  = []; subj_s  = []
for k in keys:
    m, s = msd(trial_ba[k]); trial_m.append(m); trial_s.append(s)
    m, s = msd(subj_ba[k]);  subj_m.append(m);  subj_s.append(s)

gap_pp = [(t - s) * 100 for t, s in zip(trial_m, subj_m)]

x = np.arange(len(keys))
w = 0.35

fig, ax = plt.subplots(figsize=(7.8, 4.4))

bars_t = ax.bar(x - w / 2, trial_m, w, yerr=trial_s, capsize=4,
                color="#d62728", edgecolor="black", linewidth=0.8,
                error_kw={"ecolor": "#222", "lw": 1.0},
                label="Trial-level CV\n(Wang 2025-style, subject leakage)")
bars_s = ax.bar(x + w / 2, subj_m,  w, yerr=subj_s, capsize=4,
                color="#4c78a8", edgecolor="black", linewidth=0.8,
                error_kw={"ecolor": "#222", "lw": 1.0},
                label="Subject-level CV\n(StratifiedGroupKFold, ours)")

for rect, v, sd in zip(bars_t, trial_m, trial_s):
    ax.text(rect.get_x() + rect.get_width() / 2, v + sd + 0.012,
            f"{v:.3f}", ha="center", fontsize=9)
for rect, v, sd in zip(bars_s, subj_m, subj_s):
    ax.text(rect.get_x() + rect.get_width() / 2, v + sd + 0.012,
            f"{v:.3f}", ha="center", fontsize=9)

for xi, t, s, g in zip(x, trial_m, subj_m, gap_pp):
    ax.annotate(
        "",
        xy=(xi + w / 2 + 0.02, s), xytext=(xi + w / 2 + 0.02, t),
        arrowprops=dict(arrowstyle="<->", color="#333", lw=1.2),
    )
    sign = "+" if g >= 0 else ""
    ax.text(
        xi + w / 2 + 0.12, (t + s) / 2,
        f"{sign}{g:.1f} pp",
        fontsize=9.2, color="#333", va="center", ha="left",
        fontweight="bold",
    )

ax.axhline(0.5, color="#666", linewidth=0.8, linestyle="--", alpha=0.7)
ax.text(len(keys) - 0.5, 0.51, "chance", fontsize=8, color="#666")

ax.set_xticks(x)
ax.set_xticklabels(models_label, fontsize=11)
ax.set_ylabel("Balanced Accuracy on Stress (70 recordings)")
ax.set_title(
    "Trial-level vs subject-level CV on the same 3 FMs\n"
    "Same data, per-rec DASS labels, 3 seeds — gap is protocol-driven (F01)"
)
ax.set_ylim(0.35, 0.90)
ax.grid(True, axis="y", alpha=0.25)
ax.legend(loc="upper right", framealpha=0.95, fontsize=9)

fig.text(
    0.5, -0.02,
    "Labels: --label dass (per-recording). Bars show mean ± sample std over seeds {42,123,2024}. "
    "Trial-level inherits subject leakage; subject-level uses StratifiedGroupKFold(5).",
    ha="center", fontsize=8, style="italic", color="#555",
)

plt.tight_layout()
out_pdf = "paper/figures/main/fig2_cv_gap.pdf"
out_png = out_pdf.replace(".pdf", ".png")
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"\nwrote {out_pdf}")
for label, k, tm, ts, sm, ss, g in zip(
    models_label, keys, trial_m, trial_s, subj_m, subj_s, gap_pp,
):
    print(f"  {label:<8} trial {tm:.3f}±{ts:.3f}  subject {sm:.3f}±{ss:.3f}  Δ={g:+.1f} pp")
print(f"  mean gap: {np.mean(gap_pp):+.1f} pp")
