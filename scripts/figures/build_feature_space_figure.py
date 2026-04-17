"""Feature space analysis figure: EEGMAT vs Stress.

Panel A: UMAP of EEGMAT frozen features (LaBraM), rest vs task, within-subject arrows
Panel B: UMAP of Stress frozen features (LaBraM), colored by DSS, within-subject arrows
Panel C: Direction consistency comparison (all 3 FMs, EEGMAT vs Stress)
"""
import json
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.spatial.distance import cosine
from umap import UMAP
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.cm as cm
matplotlib.rcParams.update({"font.size": 10, "font.family": "sans-serif"})

FEAT_DIR = "results/features_cache"
OUT_DIR = "results/studies/exp11_longitudinal_dss"

models = [
    ("labram", "LaBraM"),
    ("cbramod", "CBraMod"),
    ("reve", "REVE"),
]
model_colors = {"labram": "#4C72B0", "cbramod": "#DD8452", "reve": "#55A868"}
UMAP_MODEL = "labram"  # representative FM for UMAP panels

# =====================================================================
# Load data
# =====================================================================
# EEGMAT
eegmat_feat = np.load(f"{FEAT_DIR}/frozen_{UMAP_MODEL}_eegmat_19ch.npz")
eegmat_X = eegmat_feat["features"]
eegmat_pids = eegmat_feat["patient_ids"]
eegmat_labels = eegmat_feat["labels"]

# Stress
stress_feat = np.load(f"{FEAT_DIR}/frozen_{UMAP_MODEL}_stress_30ch.npz")
stress_X = stress_feat["features"]
stress_pids = stress_feat["patient_ids"]
stress_csv = pd.read_csv("data/comprehensive_labels.csv").dropna(subset=["Stress_Score"])
stress_dss = stress_csv["Stress_Score"].values

# =====================================================================
# Compute direction consistency for all FMs
# =====================================================================
dir_results = {"eegmat": {}, "stress": {}}

for key, label in models:
    # EEGMAT
    feat = np.load(f"{FEAT_DIR}/frozen_{key}_eegmat_19ch.npz")
    X, pids, labs = feat["features"], feat["patient_ids"], feat["labels"]
    directions = []
    for pid in np.unique(pids):
        mask = pids == pid
        x_rest = X[mask & (labs == 0)][0]
        x_task = X[mask & (labs == 1)][0]
        diff = x_task - x_rest
        norm = np.linalg.norm(diff)
        if norm > 0:
            directions.append(diff / norm)
    dir_sims = [1 - cosine(directions[i], directions[j])
                for i, j in combinations(range(len(directions)), 2)]
    dir_results["eegmat"][key] = float(np.mean(dir_sims))

    # Stress
    feat = np.load(f"{FEAT_DIR}/frozen_{key}_stress_30ch.npz")
    X, pids = feat["features"], feat["patient_ids"]
    dss = stress_dss[:len(X)]
    by_pid = {}
    for i in range(len(X)):
        if not np.isnan(dss[i]):
            by_pid.setdefault(pids[i], []).append(i)
    directions = []
    for pid, idxs in by_pid.items():
        if len(idxs) < 3:
            continue
        scores = [(dss[i], i) for i in idxs]
        scores.sort()
        diff = X[scores[-1][1]] - X[scores[0][1]]
        norm = np.linalg.norm(diff)
        if norm > 0:
            directions.append(diff / norm)
    dir_sims = [1 - cosine(directions[i], directions[j])
                for i, j in combinations(range(len(directions)), 2)]
    dir_results["stress"][key] = float(np.mean(dir_sims))

print("Direction consistency:")
for key, label in models:
    print(f"  {label}: EEGMAT={dir_results['eegmat'][key]:.3f}, "
          f"Stress={dir_results['stress'][key]:.3f}")

# =====================================================================
# UMAP embeddings
# =====================================================================
umap_eegmat = UMAP(n_neighbors=15, min_dist=0.3, random_state=42).fit_transform(eegmat_X)
umap_stress = UMAP(n_neighbors=15, min_dist=0.3, random_state=42).fit_transform(stress_X)

# =====================================================================
# Figure: 1 row × 3 panels
# =====================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

# --- Panel A: EEGMAT UMAP ---
# Plot points
rest_mask = eegmat_labels == 0
task_mask = eegmat_labels == 1
ax1.scatter(umap_eegmat[rest_mask, 0], umap_eegmat[rest_mask, 1],
            c="#2196F3", s=35, alpha=0.7, edgecolors="white", linewidth=0.3,
            label="Rest", zorder=3)
ax1.scatter(umap_eegmat[task_mask, 0], umap_eegmat[task_mask, 1],
            c="#F44336", s=35, alpha=0.7, edgecolors="white", linewidth=0.3,
            label="Math task", zorder=3)

# Draw within-subject arrows (rest → task)
for pid in np.unique(eegmat_pids):
    mask = eegmat_pids == pid
    rest_idx = np.where(mask & rest_mask)[0][0]
    task_idx = np.where(mask & task_mask)[0][0]
    ax1.annotate("", xy=umap_eegmat[task_idx], xytext=umap_eegmat[rest_idx],
                 arrowprops=dict(arrowstyle="->", color="gray", alpha=0.4, lw=0.8))

ax1.set_title("A. EEGMAT — rest vs math task\n(LaBraM frozen)", fontweight="bold", fontsize=11)
ax1.set_xlabel("UMAP 1")
ax1.set_ylabel("UMAP 2")
ax1.legend(fontsize=9, loc="best")

# --- Panel B: Stress UMAP ---
# Color by DSS score
valid_mask = ~np.isnan(stress_dss[:len(stress_X)])
sc = ax2.scatter(umap_stress[valid_mask, 0], umap_stress[valid_mask, 1],
                 c=stress_dss[:len(stress_X)][valid_mask], cmap="RdYlBu_r",
                 s=35, alpha=0.7, edgecolors="white", linewidth=0.3, zorder=3)
cbar = plt.colorbar(sc, ax=ax2, shrink=0.8, pad=0.02)
cbar.set_label("DSS score", fontsize=9)

# Draw within-subject arrows (lowest → highest DSS)
by_pid = {}
for i in range(len(stress_X)):
    if valid_mask[i]:
        by_pid.setdefault(stress_pids[i], []).append(i)

for pid, idxs in by_pid.items():
    if len(idxs) < 3:
        continue
    scores = [(stress_dss[i], i) for i in idxs]
    scores.sort()
    lo_idx = scores[0][1]
    hi_idx = scores[-1][1]
    ax2.annotate("", xy=umap_stress[hi_idx], xytext=umap_stress[lo_idx],
                 arrowprops=dict(arrowstyle="->", color="gray", alpha=0.4, lw=0.8))

ax2.set_title("B. Stress — DSS score\n(LaBraM frozen)", fontweight="bold", fontsize=11)
ax2.set_xlabel("UMAP 1")
ax2.set_ylabel("UMAP 2")

# --- Panel C: Direction consistency ---
x = np.arange(len(models))
width = 0.35
bars1 = ax3.bar(x - width / 2,
                [dir_results["eegmat"][k] for k, _ in models],
                width, label="EEGMAT (rest→task)",
                color="#2196F3", alpha=0.8, edgecolor="white")
bars2 = ax3.bar(x + width / 2,
                [dir_results["stress"][k] for k, _ in models],
                width, label="Stress (low→high DSS)",
                color="#F44336", alpha=0.8, edgecolor="white")

for bar in bars1:
    v = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
             f"{v:.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    v = bar.get_height()
    y_pos = v + 0.005 if v >= 0 else v - 0.025
    va = "bottom" if v >= 0 else "top"
    ax3.text(bar.get_x() + bar.get_width() / 2, y_pos,
             f"{v:.3f}", ha="center", va=va, fontsize=9)

ax3.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels([l for _, l in models])
ax3.set_ylabel("Direction consistency\n(mean pairwise cosine sim.)")
ax3.set_ylim(-0.1, 0.25)
ax3.set_title("C. Direction consistency\n(frozen features)", fontweight="bold", fontsize=11)
ax3.legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_space_analysis.pdf", bbox_inches="tight", dpi=300)
plt.savefig(f"{OUT_DIR}/feature_space_analysis.png", bbox_inches="tight", dpi=150)
print(f"\nSaved → {OUT_DIR}/feature_space_analysis.{{pdf,png}}")

# Save results
out = {
    "umap_model": UMAP_MODEL,
    "direction_consistency": dir_results,
}
with open(f"{OUT_DIR}/feature_space_analysis.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved → {OUT_DIR}/feature_space_analysis.json")
