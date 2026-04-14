"""Statistical hardening figure (exp10).

Panel A: Bootstrap CI forest plot (frozen vs FT per model)
Panel B: Cohen's d bar chart
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 10, "font.family": "sans-serif"})

OUT_DIR = "results/studies/exp10_stat_hardening"
data = json.load(open(f"{OUT_DIR}/stat_hardening.json"))

models = ["labram", "cbramod", "reve"]
model_labels = ["LaBraM", "CBraMod", "REVE"]
model_colors = {"labram": "#4C72B0", "cbramod": "#DD8452", "reve": "#55A868"}

# =====================================================================
# Figure: 2 panels
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                gridspec_kw={"width_ratios": [2, 1]})

# --- Panel A: Forest plot ---
y_positions = []
y_labels = []
y_idx = 0
spacing = 0.8

for i, (model, label) in enumerate(zip(models, model_labels)):
    frozen_key = f"{model}_frozen"
    ft_key = f"{model}_ft"
    frz = data["bootstrap_ci"][frozen_key]
    ft = data["bootstrap_ci"][ft_key]

    # Frozen
    y = y_idx
    y_positions.append(y)
    y_labels.append(f"{label} frozen (n={frz['n_seeds']})")
    ax1.errorbar(frz["mean"], y,
                 xerr=[[frz["mean"] - frz["ci_95_lo"]],
                        [frz["ci_95_hi"] - frz["mean"]]],
                 fmt="o", color=model_colors[model], markersize=8,
                 capsize=5, linewidth=2, markeredgecolor="white",
                 markeredgewidth=0.5)
    # Individual seeds
    for v in frz["values"]:
        ax1.plot(v, y, "x", color=model_colors[model], alpha=0.4,
                 markersize=5, markeredgewidth=1)

    # FT
    y = y_idx - spacing * 0.4
    y_positions.append(y)
    y_labels.append(f"{label} FT (n={ft['n_seeds']})")
    ax1.errorbar(ft["mean"], y,
                 xerr=[[ft["mean"] - ft["ci_95_lo"]],
                        [ft["ci_95_hi"] - ft["mean"]]],
                 fmt="s", color=model_colors[model], markersize=8,
                 capsize=5, linewidth=2, alpha=0.6,
                 markeredgecolor="black", markeredgewidth=0.8)
    for v in ft["values"]:
        ax1.plot(v, y, "+", color=model_colors[model], alpha=0.4,
                 markersize=6, markeredgewidth=1)

    y_idx -= spacing

ax1.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
ax1.set_yticks([p for p in y_positions])
ax1.set_yticklabels(y_labels, fontsize=9)
ax1.set_xlabel("Balanced Accuracy")
ax1.set_xlim(0.35, 0.72)
ax1.set_title("A. Bootstrap 95% CI (frozen vs FT)", fontweight="bold", fontsize=12)
ax1.legend(fontsize=8, loc="upper right")
ax1.invert_yaxis()

# Annotation: overlap check
for i, model in enumerate(models):
    frz = data["bootstrap_ci"][f"{model}_frozen"]
    ft = data["bootstrap_ci"][f"{model}_ft"]
    overlap = frz["ci_95_lo"] <= ft["ci_95_hi"] and ft["ci_95_lo"] <= frz["ci_95_hi"]
    if not overlap:
        mid_y = -i * spacing - spacing * 0.2
        ax1.text(0.36, mid_y, "no overlap", fontsize=7, color="red",
                 style="italic", va="center")

# --- Panel B: Cohen's d ---
x = np.arange(len(models))
ds = [data["cohens_d"][m]["d"] for m in models]
colors = [model_colors[m] for m in models]

bars = ax2.bar(x, ds, color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)

for bar, d in zip(bars, ds):
    y_pos = d + 0.15 if d >= 0 else d - 0.15
    va = "bottom" if d >= 0 else "top"
    ax2.text(bar.get_x() + bar.get_width() / 2, y_pos,
             f"d={d:.2f}", ha="center", va=va, fontsize=10, fontweight="bold")

ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
ax2.axhline(0.8, color="green", linestyle=":", linewidth=0.8, alpha=0.5)
ax2.axhline(-0.8, color="green", linestyle=":", linewidth=0.8, alpha=0.5)
ax2.text(2.6, 0.9, "large", fontsize=7, color="green", style="italic")
ax2.text(2.6, -0.9, "large", fontsize=7, color="green", style="italic")

ax2.set_xticks(x)
ax2.set_xticklabels(model_labels)
ax2.set_ylabel("Cohen's d (frozen − FT)")
ax2.set_ylim(-5, 6)
ax2.set_title("B. Effect size", fontweight="bold", fontsize=12)

# Annotate direction
ax2.text(0.02, 0.97, "← frozen better (erosion)", transform=ax2.transAxes,
         fontsize=8, color="#4C72B0", va="top", style="italic")
ax2.text(0.02, 0.03, "FT better (injection) →", transform=ax2.transAxes,
         fontsize=8, color="#DD8452", va="bottom", style="italic")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/stat_hardening.pdf", bbox_inches="tight", dpi=300)
plt.savefig(f"{OUT_DIR}/stat_hardening.png", bbox_inches="tight", dpi=150)
print(f"Saved → {OUT_DIR}/stat_hardening.{{pdf,png}}")
