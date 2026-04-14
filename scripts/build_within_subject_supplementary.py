"""Supplementary: Within-subject analysis — Frozen vs FT (2 rows × 3 cols).

Row 1: Frozen features
Row 2: Fine-tuned features
Col A: EEGMAT LOO classification BA
Col B: Stress LOO regression + 1-shot calibration (per-subject Spearman r)
Col C: Direction consistency (EEGMAT vs Stress, grouped bar)
"""
import json
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 10, "font.family": "sans-serif"})

FEAT_DIR = "results/features_cache"
OUT_DIR = "results/studies/exp11_longitudinal_dss"

models = [
    ("labram", "LaBraM"),
    ("cbramod", "CBraMod"),
    ("reve", "REVE"),
]
model_colors = {"labram": "#4C72B0", "cbramod": "#DD8452", "reve": "#55A868"}


def load_features(key, dataset, feat_type):
    if feat_type == "frozen":
        suffix = "30ch" if dataset == "stress" else "19ch"
        feat = np.load(f"{FEAT_DIR}/frozen_{key}_{dataset}_{suffix}.npz")
        return feat["features"], feat["patient_ids"]
    else:
        ft_dir = f"{FEAT_DIR}/ft_{key}_{dataset}"
        all_X, all_pids, all_idx = [], [], []
        for fold in range(1, 6):
            f = np.load(f"{ft_dir}/fold{fold}_features.npz")
            all_X.append(f["features"])
            all_pids.append(f["patient_ids"])
            all_idx.append(f["test_idx"])
        X_cat = np.concatenate(all_X)
        pids_cat = np.concatenate(all_pids)
        idx_cat = np.concatenate(all_idx)
        order = np.argsort(idx_cat)
        return X_cat[order], pids_cat[order]


def compute_eegmat_metrics(X, pids, labels):
    unique_pids = np.unique(pids)
    directions = []
    for pid in unique_pids:
        mask = pids == pid
        x_rest = X[mask & (labels == 0)][0]
        x_task = X[mask & (labels == 1)][0]
        diff = x_task - x_rest
        norm = np.linalg.norm(diff)
        if norm > 0:
            directions.append(diff / norm)
    dir_sims = [1 - cosine(directions[i], directions[j])
                for i, j in combinations(range(len(directions)), 2)]

    loo_preds, loo_true = [], []
    for test_pid in unique_pids:
        test_mask = pids == test_pid
        train_mask = ~test_mask
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X[train_mask], labels[train_mask])
        loo_preds.extend(clf.predict(X[test_mask]))
        loo_true.extend(labels[test_mask])

    return {
        "ba": balanced_accuracy_score(loo_true, loo_preds),
        "dir_consistency": float(np.mean(dir_sims)),
    }


def compute_stress_metrics(X, pids, dss):
    by_pid = {}
    for i in range(len(X)):
        if not np.isnan(dss[i]):
            by_pid.setdefault(pids[i], []).append(i)
    valid_pids = {pid: idxs for pid, idxs in by_pid.items() if len(idxs) >= 3}

    directions = []
    for pid, idxs in valid_pids.items():
        scores = [(dss[i], i) for i in idxs]
        scores.sort()
        diff = X[scores[-1][1]] - X[scores[0][1]]
        norm = np.linalg.norm(diff)
        if norm > 0:
            directions.append(diff / norm)
    dir_sims = [1 - cosine(directions[i], directions[j])
                for i, j in combinations(range(len(directions)), 2)]

    per_subject_r = []
    for test_pid in list(valid_pids.keys()):
        test_idxs = valid_pids[test_pid]
        train_idxs = [i for pid, idxs in valid_pids.items()
                      if pid != test_pid for i in idxs]
        reg = Ridge(alpha=1.0)
        reg.fit(X[train_idxs], dss[train_idxs])
        y_pred = reg.predict(X[test_idxs])
        offset = dss[test_idxs][0] - y_pred[0]
        y_pred_cal = y_pred + offset
        if len(test_idxs) >= 3 and np.std(dss[test_idxs][1:]) > 0:
            r, _ = spearmanr(dss[test_idxs][1:], y_pred_cal[1:])
            per_subject_r.append(r)

    return {
        "per_subject_r": np.array(per_subject_r),
        "mean_r": float(np.nanmean(per_subject_r)),
        "dir_consistency": float(np.mean(dir_sims)),
    }


# =====================================================================
# Compute all metrics
# =====================================================================
eegmat_labels = np.load(f"{FEAT_DIR}/frozen_labram_eegmat_19ch.npz")["labels"]
stress_csv = pd.read_csv("data/comprehensive_labels.csv").dropna(subset=["Stress_Score"])
dss_all = stress_csv["Stress_Score"].values

results = {}  # results[feat_type][dataset][model_key]
for feat_type in ["frozen", "ft"]:
    results[feat_type] = {"eegmat": {}, "stress": {}}
    for key, label in models:
        # EEGMAT
        X, pids = load_features(key, "eegmat", feat_type)
        labs = eegmat_labels[:len(X)]
        res = compute_eegmat_metrics(X, pids, labs)
        results[feat_type]["eegmat"][key] = res
        print(f"{feat_type.upper():6s} EEGMAT {label}: BA={res['ba']:.3f}, "
              f"dir={res['dir_consistency']:.3f}")

        # Stress
        X, pids = load_features(key, "stress", feat_type)
        res = compute_stress_metrics(X, pids, dss_all[:len(X)])
        results[feat_type]["stress"][key] = res
        print(f"{feat_type.upper():6s} Stress {label}: r={res['mean_r']:.3f}, "
              f"dir={res['dir_consistency']:.3f}")

# =====================================================================
# Figure: 2 rows × 3 columns
# =====================================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 6),
                         gridspec_kw={"hspace": 0.4, "wspace": 0.35})

x = np.arange(len(models))

for row_idx, feat_type in enumerate(["frozen", "ft"]):
    feat_label = "Frozen" if feat_type == "frozen" else "Fine-tuned"
    ax_eegmat = axes[row_idx, 0]
    ax_stress = axes[row_idx, 1]
    ax_dir = axes[row_idx, 2]

    # --- Col A: EEGMAT BA ---
    vals = [results[feat_type]["eegmat"][k]["ba"] for k, _ in models]
    bars = ax_eegmat.bar(x, vals, color=[model_colors[k] for k, _ in models],
                         edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax_eegmat.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                       f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax_eegmat.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax_eegmat.set_xticks(x)
    ax_eegmat.set_xticklabels([l for _, l in models])
    ax_eegmat.set_ylabel("Balanced Accuracy")
    ax_eegmat.set_ylim(0, 1.0)
    if row_idx == 0:
        ax_eegmat.set_title("EEGMAT LOO Classification", fontweight="bold", fontsize=11)

    # Row label
    ax_eegmat.text(-0.25, 0.5, feat_label, transform=ax_eegmat.transAxes,
                   fontsize=11, fontweight="bold", va="center", ha="right", rotation=90)

    # --- Col B: Stress per-subject r ---
    for i, (key, label) in enumerate(models):
        rs = results[feat_type]["stress"][key]["per_subject_r"]
        rs_valid = rs[~np.isnan(rs)]
        jitter = np.random.default_rng(42 + row_idx).uniform(-0.15, 0.15, len(rs_valid))
        ax_stress.scatter([i] * len(rs_valid) + jitter, rs_valid,
                          color=model_colors[key], alpha=0.5, s=30,
                          edgecolors="white", linewidth=0.3, zorder=3)
        mean_r = np.nanmean(rs_valid)
        ax_stress.scatter(i, mean_r, color=model_colors[key], s=80, marker="D",
                          edgecolors="black", linewidth=0.8, zorder=4)
        ax_stress.text(i, mean_r + 0.06, f"{mean_r:.2f}", ha="center", fontsize=10,
                       fontweight="bold")
    ax_stress.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax_stress.set_xticks(x)
    ax_stress.set_xticklabels([l for _, l in models])
    ax_stress.set_ylabel("Per-subject Spearman r")
    ax_stress.set_ylim(-1.1, 1.1)
    if row_idx == 0:
        ax_stress.set_title("Stress LOO Regression\n+ 1-shot calibration",
                            fontweight="bold", fontsize=11)

    # --- Col C: Direction consistency (grouped bar) ---
    width = 0.35
    eegmat_vals = [results[feat_type]["eegmat"][k]["dir_consistency"] for k, _ in models]
    stress_vals = [results[feat_type]["stress"][k]["dir_consistency"] for k, _ in models]

    bars1 = ax_dir.bar(x - width / 2, eegmat_vals, width,
                       label="EEGMAT" if row_idx == 0 else None,
                       color="#2196F3", alpha=0.8, edgecolor="white")
    bars2 = ax_dir.bar(x + width / 2, stress_vals, width,
                       label="Stress" if row_idx == 0 else None,
                       color="#F44336", alpha=0.8, edgecolor="white")

    for bar in bars1:
        v = bar.get_height()
        ax_dir.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        v = bar.get_height()
        y_pos = v + 0.005 if v >= 0 else v - 0.02
        va = "bottom" if v >= 0 else "top"
        ax_dir.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{v:.3f}", ha="center", va=va, fontsize=8)

    ax_dir.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_dir.set_xticks(x)
    ax_dir.set_xticklabels([l for _, l in models])
    ax_dir.set_ylabel("Direction consistency")
    ax_dir.set_ylim(-0.1, 0.25)
    if row_idx == 0:
        ax_dir.set_title("Direction Consistency\n(EEGMAT vs Stress)", fontweight="bold", fontsize=11)
        ax_dir.legend(fontsize=8, loc="upper right")

# Panel labels
for idx, lbl in enumerate("ABCDEF"):
    r, c = divmod(idx, 3)
    axes[r, c].text(-0.08, 1.08, lbl, transform=axes[r, c].transAxes,
                    fontsize=14, fontweight="bold", va="top")

plt.savefig(f"{OUT_DIR}/within_subject_supplementary.pdf", bbox_inches="tight", dpi=300)
plt.savefig(f"{OUT_DIR}/within_subject_supplementary.png", bbox_inches="tight", dpi=150)
print(f"\nSaved → {OUT_DIR}/within_subject_supplementary.{{pdf,png}}")

# Save JSON
out = {}
for feat_type in ["frozen", "ft"]:
    out[feat_type] = {}
    for ds in ["eegmat", "stress"]:
        out[feat_type][ds] = {}
        for k, _ in models:
            r = results[feat_type][ds][k]
            entry = {"dir_consistency": r["dir_consistency"]}
            if ds == "eegmat":
                entry["loo_ba"] = r["ba"]
            else:
                entry["mean_spearman_r"] = r["mean_r"]
                entry["per_subject_r"] = r["per_subject_r"].tolist()
            out[feat_type][ds][k] = entry
with open(f"{OUT_DIR}/within_subject_supplementary.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved → {OUT_DIR}/within_subject_supplementary.json")
