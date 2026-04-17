"""Fig 7c: Band-stop ablation with cross-model consistency framing.

Key claim: on EEGMAT, LaBraM+REVE both peak at ALPHA → cross-model
convergence = real neural signature. On Stress, LaBraM peaks at BETA but
REVE at ALPHA → divergence = consistent with power-floor (no converging
neural signature). CBraMod always peaks at delta across datasets, treated
as an architecture artifact and annotated separately.

Source: paper/figures/source_tables/exp14_band_stop.json
Output: paper/figures/main/fig7c_band_stop.pdf
"""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,
})

with open("paper/figures/source_tables/exp14_band_stop.json") as f:
    data = json.load(f)

bands = ["delta", "theta", "alpha", "beta"]
base_colors = {
    "delta": "#cccccc",
    "theta": "#cccccc",
    "alpha": "#d62728",  # alpha hypothesis color
    "beta":  "#cccccc",
}
models = ["labram", "cbramod", "reve"]
model_label = {"labram": "LaBraM", "cbramod": "CBraMod", "reve": "REVE"}
datasets = ["eegmat", "stress"]
dataset_label = {"eegmat": "EEGMAT\n(clean signal)", "stress": "Stress\n(power floor)"}

fig, axes = plt.subplots(2, 3, figsize=(10, 5.8), sharey="row")

# Track peak bands for cross-model consistency summary
peaks = {ds: {} for ds in datasets}

for row, ds in enumerate(datasets):
    for col, m in enumerate(models):
        ax = axes[row, col]
        vals = [data[ds][m][b]["mean_distance"] for b in bands]
        errs = [data[ds][m][b]["std_distance"] for b in bands]
        argmax = int(np.argmax(vals))
        peak_band = bands[argmax]
        peaks[ds][m] = peak_band

        # Color bars; the peak band gets a solid colored outline
        bar_colors = [base_colors[b] for b in bands]
        bars = ax.bar(
            bands,
            vals,
            yerr=errs,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.6,
            capsize=3,
            error_kw={"elinewidth": 0.8},
        )
        # Highlight the actual peak bar with a distinctive outline
        bars[argmax].set_edgecolor("#b22222")
        bars[argmax].set_linewidth(2.2)

        if row == 0:
            ax.set_title(model_label[m])
        if col == 0:
            ax.set_ylabel(f"{dataset_label[ds]}\nRep. distance")
        ax.grid(True, axis="y", alpha=0.2)
        ax.tick_params(axis="x", labelsize=8.5)

        # Small text "peak: <band>" above subplot
        ax.text(
            0.02, 0.95,
            f"peak: {peak_band}",
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            color="#b22222",
            va="top",
        )

# Cross-model consistency annotations — based on LaBraM+REVE agreement only
# (CBraMod always peaks at delta; treated as architecture artifact)
for row, ds in enumerate(datasets):
    lab_peak = peaks[ds]["labram"]
    reve_peak = peaks[ds]["reve"]
    converges = lab_peak == reve_peak
    verdict = (
        f"LaBraM & REVE peak at {lab_peak.upper()} — CONVERGES"
        if converges
        else f"LaBraM peaks at {lab_peak.upper()}, REVE at {reve_peak.upper()} — DIVERGES"
    )
    color = "#2ca02c" if converges else "#b22222"
    # Put the summary text inside the leftmost subplot, top-right corner
    axes[row, 0].text(
        1.0, -0.28,
        verdict,
        transform=axes[row, 0].transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=color,
        ha="left",
        va="top",
    )

fig.suptitle(
    "Band-stop ablation — cross-model consistency separates real signal from noise floor",
    y=1.00,
    fontsize=11,
)

# Legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles = [
    Patch(facecolor="#d62728", edgecolor="black", label="alpha (a priori hypothesis)"),
    Patch(facecolor="#cccccc", edgecolor="black", label="control bands"),
    Line2D([0], [0], color="#b22222", lw=2.5, label="observed peak band"),
]
fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.04),
    ncol=3,
    frameon=False,
    fontsize=9,
)

plt.tight_layout(rect=[0, 0.04, 1, 0.97])
out = "paper/figures/main/fig7c_band_stop.pdf"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"wrote {out}")

# Print peak summary
print("\nCross-model peak summary:")
for ds in datasets:
    p = peaks[ds]
    conv = "✓" if p["labram"] == p["reve"] else "✗"
    print(f"  {ds}: LaBraM={p['labram']}, CBraMod={p['cbramod']}, REVE={p['reve']}  convergence({conv})")
