"""Cross-dataset × cross-model: Frozen LP vs FT comparison.

4 datasets × 3 FMs heatmap of Δ(FT − Frozen), plus BA overview.
"""
import json
import os
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import TwoSlopeNorm
matplotlib.rcParams.update({"font.size": 10, "font.family": "sans-serif"})

FEAT_DIR = "results/features_cache"
OUT_DIR = "results/studies/exp13_cross_dataset_model"

models = ["labram", "cbramod", "reve"]
model_labels = ["LaBraM", "CBraMod", "REVE"]
datasets = ["stress", "adftd", "tdbrain", "eegmat"]
dataset_labels = ["Stress", "ADFTD", "TDBRAIN", "EEGMAT"]


def frozen_lp_ba(feat_path):
    feat = np.load(feat_path)
    X, pids, labels = feat["features"], feat["patient_ids"], feat["labels"]
    bas = []
    for seed in [42, 123, 2024, 7, 0, 1, 99, 31337]:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        preds, trues = [], []
        for train_idx, test_idx in cv.split(X, labels, groups=pids):
            scaler = StandardScaler().fit(X[train_idx])
            clf = LogisticRegression(C=1.0, class_weight="balanced",
                                     max_iter=1000, random_state=seed)
            clf.fit(scaler.transform(X[train_idx]), labels[train_idx])
            preds.extend(clf.predict(scaler.transform(X[test_idx])))
            trues.extend(labels[test_idx])
        bas.append(balanced_accuracy_score(trues, preds))
    return np.mean(bas), np.std(bas)


# =====================================================================
# Collect all results
# =====================================================================
frozen_ba = {}   # frozen_ba[dataset][model] = (mean, std)
ft_ba = {}       # ft_ba[dataset][model] = (mean, std)
n_seeds = {}     # tracking seed count

# --- Frozen LP for all ---
for ds in datasets:
    frozen_ba[ds] = {}
    suffix = "30ch" if ds == "stress" else "19ch"
    for model in models:
        if ds == "stress":
            d = json.load(open(f"results/studies/exp03_stress_erosion/frozen_lp/{model}_multi_seed.json"))
            frozen_ba[ds][model] = (d["mean_8seed"], d["std_8seed"])
        else:
            path = f"{FEAT_DIR}/frozen_{model}_{ds}_{suffix}.npz"
            mean, std = frozen_lp_ba(path)
            frozen_ba[ds][model] = (mean, std)
    print(f"Frozen LP {ds}: done")

# --- FT for Stress (HP sweep best) ---
ft_ba["stress"] = {}
n_seeds["stress"] = {}
for model in models:
    configs = defaultdict(list)
    model_dir = f"results/hp_sweep/20260410_dass/{model}"
    for run in os.listdir(model_dir):
        s_path = os.path.join(model_dir, run, "summary.json")
        if os.path.exists(s_path):
            s = json.load(open(s_path))
            ba = s.get("subject_bal_acc", 0)
            parts = run.rsplit("_s", 1)
            configs[parts[0]].append(ba)
    best = max(configs.items(), key=lambda x: np.mean(x[1]))
    ft_ba["stress"][model] = (np.mean(best[1]), np.std(best[1]))
    n_seeds["stress"][model] = len(best[1])

# --- FT for ADFTD/TDBRAIN (exp07/exp08, 5s windows, 3 seeds) ---
summary = json.load(open("results/studies/exp07_adftd_multiseed/multiseed_variance_summary.json"))
for ds, ds_key in [("adftd", "adftd"), ("tdbrain", "tdbrain")]:
    ft_ba[ds] = {}
    n_seeds[ds] = {}
    for model in models:
        key = f"{model}_{ds_key}"
        if key in summary:
            d = summary[key]
            bas = [v["ba"] for v in d["per_seed"].values()]
            ft_ba[ds][model] = (np.mean(bas), np.std(bas))
            n_seeds[ds][model] = len(bas)

# --- FT for EEGMAT (single seed from features_cache) ---
ft_ba["eegmat"] = {}
n_seeds["eegmat"] = {}
for model in models:
    s = json.load(open(f"{FEAT_DIR}/ft_{model}_eegmat/summary.json"))
    ft_ba["eegmat"][model] = (s["subject_bal_acc"], 0.0)
    n_seeds["eegmat"][model] = 1

# =====================================================================
# Compute deltas
# =====================================================================
delta = np.zeros((len(datasets), len(models)))
frozen_mat = np.zeros((len(datasets), len(models)))
ft_mat = np.zeros((len(datasets), len(models)))

for i, ds in enumerate(datasets):
    for j, model in enumerate(models):
        frz = frozen_ba[ds][model][0]
        ft = ft_ba[ds][model][0]
        delta[i, j] = ft - frz
        frozen_mat[i, j] = frz
        ft_mat[i, j] = ft

# Print summary table
print("\n=== Full Table ===")
print(f"{'Dataset':<10} {'Model':<10} {'Frozen':>8} {'FT':>8} {'Δ':>8} {'Seeds':>6}")
for i, ds in enumerate(datasets):
    for j, model in enumerate(models):
        frz_m, frz_s = frozen_ba[ds][model]
        ft_m, ft_s = ft_ba[ds][model]
        ns = n_seeds.get(ds, {}).get(model, "?")
        print(f"{ds:<10} {model:<10} {frz_m:>8.3f} {ft_m:>8.3f} {delta[i,j]:>+8.3f} {ns:>6}")

# =====================================================================
# Figure: 2 panels
# Panel A: Grouped bar chart (frozen vs FT, grouped by dataset)
# Panel B: Δ heatmap
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                gridspec_kw={"width_ratios": [2, 1]})

# --- Panel A: Grouped bars ---
n_ds = len(datasets)
n_m = len(models)
group_width = 0.7
bar_width = group_width / (n_m * 2 + 1)  # frozen + ft per model + gap
model_colors_list = ["#4C72B0", "#DD8452", "#55A868"]

for j, (model, color) in enumerate(zip(models, model_colors_list)):
    x_frozen = np.arange(n_ds) + j * bar_width * 2 - group_width / 2 + bar_width / 2
    x_ft = x_frozen + bar_width

    frz_vals = [frozen_ba[ds][model][0] for ds in datasets]
    frz_errs = [frozen_ba[ds][model][1] for ds in datasets]
    ft_vals = [ft_ba[ds][model][0] for ds in datasets]
    ft_errs = [ft_ba[ds][model][1] for ds in datasets]

    ax1.bar(x_frozen, frz_vals, bar_width, yerr=frz_errs, capsize=2,
            color=color, alpha=0.9, edgecolor="white", linewidth=0.3,
            label=f"{model_labels[j]} frozen" if j == 0 else f"{model_labels[j]} frozen")
    ax1.bar(x_ft, ft_vals, bar_width, yerr=ft_errs, capsize=2,
            color=color, alpha=0.4, edgecolor="black", linewidth=0.5, hatch="//",
            label=f"{model_labels[j]} FT" if j == 0 else f"{model_labels[j]} FT")

ax1.axhline(0.5, color="gray", linestyle="--", linewidth=1)
ax1.set_xticks(np.arange(n_ds))
ax1.set_xticklabels(dataset_labels)
ax1.set_ylabel("Balanced Accuracy")
ax1.set_ylim(0.35, 0.80)
ax1.set_title("A. Frozen LP vs FT across datasets", fontweight="bold", fontsize=12)

# Custom legend: model color squares + frozen/FT distinction
from matplotlib.patches import Patch
legend_elements = []
for j, (model, color) in enumerate(zip(model_labels, model_colors_list)):
    legend_elements.append(Patch(facecolor=color, alpha=0.9, label=f"{model} frozen"))
    legend_elements.append(Patch(facecolor=color, alpha=0.4, edgecolor="black",
                                  hatch="//", label=f"{model} FT"))
ax1.legend(handles=legend_elements, fontsize=7, ncol=3, loc="upper left")

# --- Panel B: Δ heatmap ---
norm = TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)
im = ax2.imshow(delta, cmap="RdBu_r", norm=norm, aspect="auto")

for i in range(len(datasets)):
    for j in range(len(models)):
        d = delta[i, j]
        color = "white" if abs(d) > 0.05 else "black"
        ns = n_seeds.get(datasets[i], {}).get(models[j], "?")
        seed_marker = "" if ns >= 3 else "*"
        ax2.text(j, i, f"{d:+.1f}pp{seed_marker}", ha="center", va="center",
                 fontsize=10, fontweight="bold", color=color)

ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(model_labels)
ax2.set_yticks(range(len(datasets)))
ax2.set_yticklabels(dataset_labels)
ax2.set_title("B. Δ(FT − Frozen) pp", fontweight="bold", fontsize=12)
cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
cbar.set_label("Δ BA (pp)")

# Note about single-seed
ax2.text(0.5, -0.15, "* = single seed  |  Stress = 10s window, others = 5s window",
         transform=ax2.transAxes, ha="center", fontsize=8, style="italic")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/cross_dataset_model_comparison.pdf", bbox_inches="tight", dpi=300)
plt.savefig(f"{OUT_DIR}/cross_dataset_model_comparison.png", bbox_inches="tight", dpi=150)
print(f"\nSaved → {OUT_DIR}/cross_dataset_model_comparison.{{pdf,png}}")
