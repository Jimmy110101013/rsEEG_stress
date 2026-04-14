"""Matched subsample frozen LP BA curves: BA vs N for 3 FMs × 3 datasets.

Subsample each dataset at multiple N rungs (N=10,17,25,35,50,...),
run frozen LP at each rung × 100 draws, plot BA degradation curve.
Stress BA plotted as horizontal reference.
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 10, "font.family": "sans-serif"})

FEAT_DIR = "results/features_cache"
OUT_DIR = "results/studies/exp09_multimodel_matched"

models = ["labram", "cbramod", "reve"]
model_labels = ["LaBraM", "CBraMod", "REVE"]
model_colors = {"labram": "#4C72B0", "cbramod": "#DD8452", "reve": "#55A868"}
datasets = ["adftd", "tdbrain", "eegmat"]
dataset_labels = ["ADFTD", "TDBRAIN", "EEGMAT"]

N_DRAWS = 100

# Stress reference (8-seed frozen LP)
stress_ba = {}
for model in models:
    d = json.load(open(f"results/studies/exp03_stress_erosion/frozen_lp/{model}_multi_seed.json"))
    stress_ba[model] = {"mean": d["mean_8seed"], "std": d["std_8seed"]}


def frozen_lp_ba(X, labels, pids, seed=42):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    preds, trues = [], []
    for train_idx, test_idx in cv.split(X, labels, groups=pids):
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=1000, random_state=seed)
        clf.fit(scaler.transform(X[train_idx]), labels[train_idx])
        preds.extend(clf.predict(scaler.transform(X[test_idx])))
        trues.extend(labels[test_idx])
    return balanced_accuracy_score(trues, preds)


# =====================================================================
# Compute BA at each N rung
# =====================================================================
results = {}

for ds in datasets:
    suffix = "19ch"
    for model in models:
        key = f"{model}_{ds}"
        feat = np.load(f"{FEAT_DIR}/frozen_{model}_{ds}_{suffix}.npz")
        X = feat["features"]
        pids = feat["patient_ids"]
        labels = feat["labels"]

        unique_pids = np.unique(pids)
        n_subj = len(unique_pids)

        # Define N rungs: 10, 17, 25, 35, 50, ... up to full N
        rungs = sorted(set([10, 17, 25, 35, 50, 75, 100, 150, 200, 300, n_subj])
                       & set(range(10, n_subj + 1)))

        # Full-N BA (8-seed)
        full_bas = [frozen_lp_ba(X, labels, pids, s)
                    for s in [42, 123, 2024, 7, 0, 1, 99, 31337]]

        rung_results = {}
        for n in rungs:
            if n == n_subj:
                rung_results[n] = {
                    "mean": float(np.mean(full_bas)),
                    "std": float(np.std(full_bas)),
                    "ci95_lo": float(np.percentile(full_bas, 2.5)),
                    "ci95_hi": float(np.percentile(full_bas, 97.5)),
                    "n_draws": 8,
                }
                continue

            draw_bas = []
            for draw in range(N_DRAWS):
                rng = np.random.default_rng(draw)
                sampled = rng.choice(unique_pids, size=n, replace=False)
                mask = np.isin(pids, sampled)
                X_sub, labels_sub, pids_sub = X[mask], labels[mask], pids[mask]

                if len(np.unique(labels_sub)) < 2:
                    continue
                if min(np.bincount(labels_sub)) < 5:
                    continue

                draw_bas.append(frozen_lp_ba(X_sub, labels_sub, pids_sub, seed=draw))

            if len(draw_bas) < 10:
                continue

            draw_bas = np.array(draw_bas)
            rung_results[n] = {
                "mean": float(np.mean(draw_bas)),
                "std": float(np.std(draw_bas)),
                "ci95_lo": float(np.percentile(draw_bas, 2.5)),
                "ci95_hi": float(np.percentile(draw_bas, 97.5)),
                "n_draws": len(draw_bas),
            }

        results[key] = {
            "model": model,
            "dataset": ds,
            "full_n_subj": int(n_subj),
            "rungs": {str(k): v for k, v in rung_results.items()},
        }

        rung_str = ", ".join(f"N{n}={rung_results[n]['mean']:.3f}" for n in sorted(rung_results))
        print(f"{key}: {rung_str}")

# Save JSON
with open(f"{OUT_DIR}/matched_n_ba_curve.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {OUT_DIR}/matched_n_ba_curve.json")

# =====================================================================
# Figure: 1 row × 3 datasets, 3 FM curves each
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for j, (ds, dlabel) in enumerate(zip(datasets, dataset_labels)):
    ax = axes[j]

    for model, mlabel, color in zip(models, model_labels, model_colors.values()):
        key = f"{model}_{ds}"
        if key not in results:
            continue

        entry = results[key]
        rungs = entry["rungs"]
        ns = sorted([int(k) for k in rungs.keys()])
        means = [rungs[str(n)]["mean"] for n in ns]
        stds = [rungs[str(n)]["std"] for n in ns]

        ax.fill_between(ns,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.12, color=color)
        ax.plot(ns, means, "o-", color=color, linewidth=2,
                markersize=4, label=mlabel, zorder=3)

    # Stress reference bands
    for model, color in zip(models, model_colors.values()):
        m = stress_ba[model]["mean"]
        s = stress_ba[model]["std"]
        ax.axhspan(m - s, m + s, alpha=0.08, color=color)
        ax.axhline(m, color=color, linestyle=":", linewidth=1, alpha=0.6)

    # N=17 vertical line
    ax.axvline(17, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(17.5, ax.get_ylim()[0] + 0.01 if j == 0 else 0.42,
            "N=17\n(Stress)", fontsize=7, color="gray", va="bottom")

    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.set_xlabel("N (subjects)")
    ax.set_ylabel("Balanced Accuracy" if j == 0 else "")
    ax.set_title(dlabel, fontsize=12, fontweight="bold")
    ax.set_ylim(0.4, 0.80)

    if j == 0:
        ax.legend(fontsize=8, loc="lower right")

# Add annotation
axes[1].text(0.5, -0.18,
             "Solid lines: subsampled BA (100 draws) | Dotted lines: Stress frozen LP reference (8-seed) | Shaded: ±1 std",
             transform=axes[1].transAxes, ha="center", fontsize=8, style="italic")

plt.suptitle("Frozen LP BA vs N: Does performance degrade to Stress levels at matched sample size?",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/matched_n_ba_curve.pdf", bbox_inches="tight", dpi=300)
plt.savefig(f"{OUT_DIR}/matched_n_ba_curve.png", bbox_inches="tight", dpi=150)
print(f"Saved → {OUT_DIR}/matched_n_ba_curve.{{pdf,png}}")
