"""Build longitudinal DSS figure (exp11).

Three panels:
  A. Within-subject median-split classification BA (all below chance)
  B. Per-subject accuracy distribution
  C. Feature distance vs DSS score difference correlation (negative)
"""
import json
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

EXP_DIR = "results/studies/exp11_longitudinal_dss"
FEAT_DIR = "results/features_cache"
CSV_PATH = "data/comprehensive_labels.csv"

models = [
    ("labram", "LaBraM", "zscore"),
    ("cbramod", "CBraMod", "none"),
    ("reve", "REVE", "none"),
]
classifiers = ["centroid", "1nn"]
clf_labels = ["Centroid", "1-NN"]
model_colors = {"labram": "#4C72B0", "cbramod": "#DD8452", "reve": "#55A868"}

# =====================================================================
# Load classification data
# =====================================================================
cls_data = {}
per_subj = {}
for key, label, _ in models:
    with open(f"{EXP_DIR}/{key}/summary.json") as f:
        s = json.load(f)
    cls_data[key] = {c: s[c]["bal_acc"] for c in classifiers}
    per_subj[key] = {}
    for c in classifiers:
        accs = [
            subj[f"{c}_acc"]
            for subj in s["per_subject"]
            if f"{c}_acc" in subj
        ]
        per_subj[key][c] = accs

# =====================================================================
# Compute within-subject pairwise correlation (Panel C)
# =====================================================================
df = pd.read_csv(CSV_PATH)
# Only subjects with valid DSS scores
df = df.dropna(subset=["Stress_Score"])

corr_results = {}      # raw DSS
corr_results_z = {}    # within-subject z-scored DSS
scatter_data = {}
scatter_data_z = {}

for key, label, norm in models:
    feat_path = f"{FEAT_DIR}/frozen_{key}_stress_30ch.npz"
    feat = np.load(feat_path)
    X = feat["features"]       # (N, embed_dim)
    pids = feat["patient_ids"]  # (N,)

    # Match features to DSS scores by row order (both follow CSV order)
    assert len(X) == len(df), f"Feature count {len(X)} != CSV rows {len(df)}"
    dss_scores = df["Stress_Score"].values

    # Within-subject z-score normalization of DSS
    dss_z = np.full_like(dss_scores, np.nan)
    for pid in np.unique(pids):
        mask = pids == pid
        vals = dss_scores[mask]
        valid = ~np.isnan(vals)
        if valid.sum() >= 3 and np.std(vals[valid]) > 0:
            dss_z[mask] = (vals - np.mean(vals[valid])) / np.std(vals[valid])

    # Within-subject pairs (subjects with ≥3 recordings and valid DSS)
    by_pid = {}
    for i in range(len(X)):
        if not np.isnan(dss_scores[i]):
            by_pid.setdefault(pids[i], []).append(i)

    dists, dss_diffs, dss_diffs_z = [], [], []
    for pid, idxs in by_pid.items():
        if len(idxs) < 3:
            continue
        for i1, i2 in combinations(idxs, 2):
            d = cosine(X[i1], X[i2])
            dd = abs(dss_scores[i1] - dss_scores[i2])
            dists.append(d)
            dss_diffs.append(dd)
            # z-scored version (skip if z-score unavailable)
            if not np.isnan(dss_z[i1]) and not np.isnan(dss_z[i2]):
                dss_diffs_z.append(abs(dss_z[i1] - dss_z[i2]))

    dists = np.array(dists)
    dss_diffs = np.array(dss_diffs)
    dss_diffs_z = np.array(dss_diffs_z)
    r, p = spearmanr(dss_diffs, dists)
    rz, pz = spearmanr(dss_diffs_z, dists)
    corr_results[key] = {"r": r, "p": p, "n_pairs": len(dists)}
    corr_results_z[key] = {"r": rz, "p": pz, "n_pairs": len(dists)}
    scatter_data[key] = (dss_diffs, dists)
    scatter_data_z[key] = (dss_diffs_z, dists)
    print(f"{label}: raw r={r:.3f} p={p:.4f} | z-scored r={rz:.3f} p={pz:.4f} | n={len(dists)}")

# =====================================================================
# Figure: 3 panels
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                         gridspec_kw={"width_ratios": [1, 1.2, 1.3]})
ax_bar, ax_dot, ax_corr = axes

# --- Panel A: Overall BA bars ---
x = np.arange(len(models))
width = 0.3
bar_colors = ["#4C72B0", "#DD8452"]

for i, (clf, clf_label) in enumerate(zip(classifiers, clf_labels)):
    vals = [cls_data[k][clf] for k, _, _ in models]
    bars = ax_bar.bar(x + i * width, vals, width, label=clf_label,
                      color=bar_colors[i], edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)

ax_bar.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
ax_bar.set_xticks(x + width / 2)
ax_bar.set_xticklabels([l for _, l, _ in models])
ax_bar.set_ylabel("Balanced Accuracy")
ax_bar.set_ylim(0, 0.65)
ax_bar.set_title("A. Within-subject classification", fontweight="bold", fontsize=11)
ax_bar.legend(loc="upper left", fontsize=8)

# --- Panel B: Per-subject dot strip ---
positions = []
tick_labels = []
pos = 0
gap = 0.6
for mi, (key, model_label, _) in enumerate(models):
    for ci, (clf, clf_label) in enumerate(zip(classifiers, clf_labels)):
        accs = per_subj[key][clf]
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(accs))
        ax_dot.scatter(
            [pos] * len(accs) + jitter, accs,
            color=bar_colors[ci], alpha=0.6, s=25, edgecolors="white", linewidth=0.3,
            zorder=3,
        )
        ax_dot.scatter(pos, np.mean(accs), color=bar_colors[ci], s=70, marker="D",
                       edgecolors="black", linewidth=0.8, zorder=4)
        positions.append(pos)
        tick_labels.append(f"{model_label}\n{clf_label}")
        pos += 1
    pos += gap

ax_dot.axhline(0.5, color="gray", linestyle="--", linewidth=1)
ax_dot.set_xticks(positions)
ax_dot.set_xticklabels(tick_labels, fontsize=7)
ax_dot.set_ylabel("Per-subject accuracy")
ax_dot.set_ylim(-0.05, 1.05)
ax_dot.set_title("B. Per-subject distribution", fontweight="bold", fontsize=11)

n_subjects = len(per_subj[models[0][0]][classifiers[0]])
ax_dot.text(0.98, 0.02, f"N = {n_subjects} subjects\n(median-split, ≥4 rec.)",
            transform=ax_dot.transAxes, ha="right", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# --- Panel C: Feature distance vs z-scored DSS difference ---
for key, label, _ in models:
    dss_diffs_z, dists = scatter_data_z[key]
    ax_corr.scatter(dss_diffs_z, dists, alpha=0.3, s=15, color=model_colors[key],
                    edgecolors="none", label=None)
    # Trend line
    z = np.polyfit(dss_diffs_z, dists, 1)
    x_line = np.linspace(dss_diffs_z.min(), dss_diffs_z.max(), 50)
    cz = corr_results_z[key]
    ax_corr.plot(x_line, np.polyval(z, x_line), color=model_colors[key], linewidth=2,
                 label=f"{label} (r={cz['r']:.2f}, p={cz['p']:.3f})")

ax_corr.set_xlabel("|Δz(DSS)| (within-subject, z-normed)")
ax_corr.set_ylabel("Cosine distance")
ax_corr.set_title("C. Feature dist. vs DSS diff. (z-normed)", fontweight="bold", fontsize=11)
ax_corr.legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig(f"{EXP_DIR}/longitudinal_dss.pdf", bbox_inches="tight", dpi=300)
plt.savefig(f"{EXP_DIR}/longitudinal_dss.png", bbox_inches="tight", dpi=150)
print(f"\nSaved → {EXP_DIR}/longitudinal_dss.{{pdf,png}}")

# Save correlation results as JSON (both raw and z-scored)
corr_out = {}
for k in corr_results:
    corr_out[k] = {
        "raw": {"spearman_r": corr_results[k]["r"], "p_value": corr_results[k]["p"]},
        "z_scored": {"spearman_r": corr_results_z[k]["r"], "p_value": corr_results_z[k]["p"]},
        "n_pairs": corr_results[k]["n_pairs"],
    }
with open(f"{EXP_DIR}/dss_correlation.json", "w") as f:
    json.dump(corr_out, f, indent=2)
print(f"Saved → {EXP_DIR}/dss_correlation.json")

# Summary
print("\n=== Classification Summary ===")
print(f"{'Model':<10} {'Centroid BA':>12} {'1-NN BA':>10} {'Linear BA':>10}")
for key, label, _ in models:
    with open(f"{EXP_DIR}/{key}/summary.json") as f:
        s = json.load(f)
    print(f"{label:<10} {s['centroid']['bal_acc']:>12.3f} {s['1nn']['bal_acc']:>10.3f} {s['linear']['bal_acc']:>10.3f}")

print("\n=== Correlation Summary ===")
print(f"{'Model':<10} {'Raw r':>8} {'Raw p':>8} {'Z r':>8} {'Z p':>8} {'N':>6}")
for key, label, _ in models:
    c = corr_results[key]
    cz = corr_results_z[key]
    print(f"{label:<10} {c['r']:>8.3f} {c['p']:>8.4f} {cz['r']:>8.3f} {cz['p']:>8.4f} {c['n_pairs']:>6d}")
