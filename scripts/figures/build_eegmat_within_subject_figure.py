"""EEGMAT within-subject condition contrast (rest vs math task).

Tests whether frozen FM representations can distinguish rest from task
within the same subject. Compared with Stress dataset (exp11) to show
that FMs capture strong task contrasts but not subtle stress variation.
"""
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import wilcoxon, spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

FEAT_DIR = "results/features_cache"
STRESS_EXP = "results/studies/exp11_longitudinal_dss"
OUT_DIR = "results/studies/exp11_longitudinal_dss"  # add to same experiment

models = [
    ("labram", "LaBraM"),
    ("cbramod", "CBraMod"),
    ("reve", "REVE"),
]
model_colors = {"labram": "#4C72B0", "cbramod": "#DD8452", "reve": "#55A868"}

# =====================================================================
# EEGMAT: within-subject rest vs task analysis
# =====================================================================
eegmat_results = {}

for key, label in models:
    feat = np.load(f"{FEAT_DIR}/frozen_{key}_eegmat_19ch.npz")
    X = feat["features"]
    pids = feat["patient_ids"]
    labels = feat["labels"]

    unique_pids = np.unique(pids)
    n_subj = len(unique_pids)

    # 1. Within-subject rest-task cosine distance
    distances = []
    for pid in unique_pids:
        mask = pids == pid
        x_rest = X[mask & (labels == 0)][0]
        x_task = X[mask & (labels == 1)][0]
        distances.append(cosine(x_rest, x_task))
    distances = np.array(distances)

    # 2. Leave-one-subject-out classification
    loo_preds = []
    loo_true = []
    for test_pid in unique_pids:
        test_mask = pids == test_pid
        train_mask = ~test_mask
        X_train, y_train = X[train_mask], labels[train_mask]
        X_test, y_test = X[test_mask], labels[test_mask]

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        loo_preds.extend(pred)
        loo_true.extend(y_test)

    ba = balanced_accuracy_score(loo_true, loo_preds)

    # 3. Wilcoxon test: are distances significantly > 0?
    stat, p_wilcox = wilcoxon(distances, alternative="greater")

    eegmat_results[key] = {
        "distances": distances,
        "loo_ba": ba,
        "dist_mean": float(distances.mean()),
        "dist_std": float(distances.std()),
        "wilcoxon_p": float(p_wilcox),
        "n_subjects": int(n_subj),
    }
    print(f"{label}: LOO BA={ba:.3f}, dist={distances.mean():.4f}±{distances.std():.4f}, Wilcoxon p={p_wilcox:.4f}")

# =====================================================================
# Load Stress within-subject distances for comparison
# =====================================================================
stress_dists = {}
stress_csv = pd.read_csv("data/comprehensive_labels.csv").dropna(subset=["Stress_Score"])

for key, label in models:
    feat = np.load(f"{FEAT_DIR}/frozen_{key}_stress_30ch.npz")
    X = feat["features"]
    pids = feat["patient_ids"]
    dss = stress_csv["Stress_Score"].values

    by_pid = {}
    for i in range(len(X)):
        if not np.isnan(dss[i]):
            by_pid.setdefault(pids[i], []).append(i)

    # For subjects with ≥3 recs: split into high-DSS vs low-DSS pairs
    all_dists = []
    for pid, idxs in by_pid.items():
        if len(idxs) < 2:
            continue
        scores = [dss[i] for i in idxs]
        median_s = np.median(scores)
        high_idx = [i for i in idxs if dss[i] >= median_s]
        low_idx = [i for i in idxs if dss[i] < median_s]
        if high_idx and low_idx:
            # Average distance between high and low groups
            for hi in high_idx:
                for lo in low_idx:
                    all_dists.append(cosine(X[hi], X[lo]))

    stress_dists[key] = np.array(all_dists)

# =====================================================================
# Figure: 3 panels
# =====================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

# --- Panel A: LOO classification BA ---
x = np.arange(len(models))
bars = ax1.bar(x, [eegmat_results[k]["loo_ba"] for k, _ in models],
               color=[model_colors[k] for k, _ in models],
               edgecolor="white", linewidth=0.5)
for bar, (k, _) in zip(bars, models):
    v = eegmat_results[k]["loo_ba"]
    ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
             f"{v:.2f}", ha="center", va="bottom", fontsize=10)

ax1.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
ax1.set_xticks(x)
ax1.set_xticklabels([l for _, l in models])
ax1.set_ylabel("Balanced Accuracy")
ax1.set_ylim(0, 1.05)
ax1.set_title("A. EEGMAT LOO classification\n(rest vs math task)", fontweight="bold", fontsize=11)
ax1.legend(fontsize=9)

# --- Panel B: Within-subject distance distribution ---
positions = np.arange(len(models))
for i, (key, label) in enumerate(models):
    dists = eegmat_results[key]["distances"]
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(dists))
    ax2.scatter([i] * len(dists) + jitter, dists,
                color=model_colors[key], alpha=0.5, s=25, edgecolors="white", linewidth=0.3)
    ax2.scatter(i, dists.mean(), color=model_colors[key], s=80, marker="D",
                edgecolors="black", linewidth=0.8, zorder=4)
    p = eegmat_results[key]["wilcoxon_p"]
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ax2.text(i, dists.max() + 0.005, stars, ha="center", fontsize=10)

ax2.set_xticks(positions)
ax2.set_xticklabels([l for _, l in models])
ax2.set_ylabel("Cosine distance (rest ↔ task)")
ax2.set_title("B. Within-subject rest-task\ndistance", fontweight="bold", fontsize=11)

# --- Panel C: EEGMAT vs Stress comparison ---
data_box = []
labels_box = []
colors_box = []
positions_box = []
pos = 0
for i, (key, label) in enumerate(models):
    # EEGMAT
    data_box.append(eegmat_results[key]["distances"])
    labels_box.append(f"{label}\nEEGMAT")
    colors_box.append(model_colors[key])
    positions_box.append(pos)
    pos += 1
    # Stress
    data_box.append(stress_dists[key])
    labels_box.append(f"{label}\nStress")
    colors_box.append(model_colors[key])
    positions_box.append(pos)
    pos += 1.5

bp = ax3.boxplot(data_box, positions=positions_box, widths=0.6,
                 patch_artist=True, showfliers=False)
for patch, color, pos_idx in zip(bp["boxes"], colors_box, range(len(data_box))):
    alpha = 0.8 if pos_idx % 2 == 0 else 0.3  # EEGMAT solid, Stress faded
    patch.set_facecolor(color)
    patch.set_alpha(alpha)

ax3.set_xticks(positions_box)
ax3.set_xticklabels(labels_box, fontsize=7)
ax3.set_ylabel("Cosine distance")
ax3.set_title("C. EEGMAT (task) vs Stress (DSS)\nwithin-subject distance", fontweight="bold", fontsize=11)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eegmat_within_subject.pdf", bbox_inches="tight", dpi=300)
plt.savefig(f"{OUT_DIR}/eegmat_within_subject.png", bbox_inches="tight", dpi=150)
print(f"\nSaved → {OUT_DIR}/eegmat_within_subject.{{pdf,png}}")

# Save results
out = {}
for key, label in models:
    r = eegmat_results[key]
    out[key] = {
        "loo_ba": r["loo_ba"],
        "dist_mean": r["dist_mean"],
        "dist_std": r["dist_std"],
        "wilcoxon_p": r["wilcoxon_p"],
        "n_subjects": r["n_subjects"],
        "stress_dist_mean": float(stress_dists[key].mean()),
        "stress_dist_std": float(stress_dists[key].std()),
    }
with open(f"{OUT_DIR}/eegmat_within_subject.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved → {OUT_DIR}/eegmat_within_subject.json")
