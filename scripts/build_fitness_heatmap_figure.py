"""FM-task fitness heatmap (exp06).

3 FMs × 4 datasets heatmaps for key metrics: RSA label r, kNN BA, label fraction.
Frozen vs FT side by side.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import TwoSlopeNorm
matplotlib.rcParams.update({"font.size": 10, "font.family": "sans-serif"})

OUT_DIR = "results/studies/exp06_fm_task_fitness"
data = json.load(open(f"{OUT_DIR}/fitness_metrics_full.json"))

models = ["labram", "cbramod", "reve"]
model_labels = ["LaBraM", "CBraMod", "REVE"]
datasets = ["stress", "adftd", "tdbrain", "eegmat"]
dataset_labels = ["Stress", "ADFTD", "TDBRAIN", "EEGMAT"]

metrics = [
    ("rsa_label_r", "RSA label r", "RdBu_r", None),
    ("rsa_subject_r", "RSA subject r", "Oranges", None),
    ("knn_ba", "kNN BA", "RdBu_r", None),
    ("label_frac_pct", "Label fraction (%)", "RdBu_r", None),
]

# =====================================================================
# Figure: 4 rows (metrics) × 2 cols (frozen, FT)
# =====================================================================
fig, axes = plt.subplots(len(metrics), 2, figsize=(9, 12),
                         gridspec_kw={"hspace": 0.4, "wspace": 0.3})

for row, (metric_key, metric_label, cmap, norm) in enumerate(metrics):
    for col, feat_type in enumerate(["frozen", "ft"]):
        ax = axes[row, col]
        mat = np.zeros((len(datasets), len(models)))

        for i, ds in enumerate(datasets):
            for j, model in enumerate(models):
                key = f"{model}_{ds}"
                mat[i, j] = data["per_model_dataset"][key][feat_type][metric_key]

        # Determine norm
        if metric_key == "rsa_label_r":
            vmax = max(0.1, np.abs(mat).max())
            norm_obj = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        elif metric_key == "knn_ba":
            norm_obj = TwoSlopeNorm(vmin=0.4, vcenter=0.5, vmax=0.7)
        elif metric_key == "label_frac_pct":
            norm_obj = None
            cmap = "YlOrRd"
        else:
            norm_obj = None

        im = ax.imshow(mat, cmap=cmap, norm=norm_obj, aspect="auto")

        # Annotate cells
        for i in range(len(datasets)):
            for j in range(len(models)):
                v = mat[i, j]
                if metric_key == "label_frac_pct":
                    txt = f"{v:.1f}"
                else:
                    txt = f"{v:.3f}"
                color = "white" if abs(v) > 0.5 * (mat.max() - mat.min()) + mat.min() else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(model_labels, fontsize=9)
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(dataset_labels if col == 0 else [], fontsize=9)

        title = f"{'Frozen' if feat_type == 'frozen' else 'FT'}"
        if col == 0:
            title = f"{metric_label}\n{title}"
        ax.set_title(title, fontsize=10, fontweight="bold")

        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05)

plt.suptitle("FM-Task Fitness Metrics (3 FMs × 4 Datasets)",
             fontsize=13, fontweight="bold", y=1.01)
plt.savefig(f"{OUT_DIR}/fitness_heatmap.pdf", bbox_inches="tight", dpi=300)
plt.savefig(f"{OUT_DIR}/fitness_heatmap.png", bbox_inches="tight", dpi=150)
print(f"Saved → {OUT_DIR}/fitness_heatmap.{{pdf,png}}")
