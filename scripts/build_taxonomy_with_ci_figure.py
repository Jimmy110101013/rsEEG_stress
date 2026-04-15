"""Fig 5: Cross-dataset × cross-model FT taxonomy with 3-seed error bars.

Addresses paper-review Phase 1 critique: effect sizes are small (~1pp) and
need visible uncertainty. Data from findings.md F17 (3-seed multi-model).

Output: paper/figures/main/fig5_cross_dataset_taxonomy.pdf (overwrite)
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,
})

# F17 data: (mean, sample-std) of pp Delta (FT - Frozen) across 3 seeds
# sample-std = stdev computed with (n-1) divisor; see docs/findings.md notation
data = {
    "LaBraM":  {"ADFTD": (+1.03, 0.74), "TDBRAIN": (-1.56, 0.28)},
    "CBraMod": {"ADFTD": (+0.83, 3.35), "TDBRAIN": (-0.02, 0.04)},
    "REVE":    {"ADFTD": (-1.53, 0.28), "TDBRAIN": (+0.44, 0.32)},
}

models = ["LaBraM", "CBraMod", "REVE"]
datasets = ["ADFTD", "TDBRAIN"]
colors = {"ADFTD": "#1f77b4", "TDBRAIN": "#ff7f0e"}

fig, ax = plt.subplots(figsize=(7.0, 4.2))

x = np.arange(len(models))
width = 0.36

for i, ds in enumerate(datasets):
    means = [data[m][ds][0] for m in models]
    stds = [data[m][ds][1] for m in models]
    bars = ax.bar(
        x + (i - 0.5) * width,
        means,
        width,
        yerr=stds,
        capsize=5,
        color=colors[ds],
        edgecolor="black",
        linewidth=0.8,
        label=ds,
        error_kw={"elinewidth": 1.2, "ecolor": "#333"},
    )
    # annotate values
    for b, m, s in zip(bars, means, stds):
        sign = "+" if m >= 0 else ""
        ax.text(
            b.get_x() + b.get_width() / 2,
            m + (0.35 if m >= 0 else -0.35) + (s if m >= 0 else -s),
            f"{sign}{m:.2f}",
            ha="center",
            va="bottom" if m >= 0 else "top",
            fontsize=8.5,
        )

ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel(r"$\Delta$ pooled label fraction (FT $-$ Frozen, pp)")
ax.set_title("Model × Dataset FT interaction (3-seed, mean ± 1σ)")
ax.legend(title="Dataset", loc="lower right", framealpha=0.95)
ax.grid(True, axis="y", alpha=0.25)
ax.set_ylim(-5.5, 5.5)

# Annotate the CBraMod ADFTD std-dominated case
ax.annotate(
    "σ > |μ|",
    xy=(1 - 0.5 * width, data["CBraMod"]["ADFTD"][0] + data["CBraMod"]["ADFTD"][1]),
    xytext=(1 - 0.5 * width - 0.15, 5),
    fontsize=8,
    color="#b22222",
    ha="center",
    arrowprops=dict(arrowstyle="->", color="#b22222", lw=0.8),
)

plt.tight_layout()
out = "paper/figures/main/fig5_cross_dataset_taxonomy.pdf"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"wrote {out}")
