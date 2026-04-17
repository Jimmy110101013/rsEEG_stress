"""Classical vs FM comparison on Stress (exp02).

Bar chart: 5 classical methods vs 3 FM frozen LP, all under per-rec DASS.
Shows classical collapse while FM frozen features survive.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 10, "font.family": "sans-serif"})

OUT_DIR = "results/studies/exp02_classical_dass"

# --- Classical (70-rec rerun) ---
classical = json.load(open(f"{OUT_DIR}/rerun_70rec/summary.json"))

classical_methods = ["RF", "XGBoost", "SVM_RBF", "LogReg_L2", "LogReg_L1"]
classical_labels = ["RF", "XGBoost", "SVM-RBF", "LogReg-L2", "LogReg-L1"]
classical_bas = [classical["models"][m]["bal_acc"] for m in classical_methods]
classical_kappas = [classical["models"][m]["kappa"] for m in classical_methods]

# --- FM frozen LP (8-seed) ---
fm_data = {}
for key in ["labram", "cbramod", "reve"]:
    d = json.load(open(f"results/studies/exp03_stress_erosion/frozen_lp/{key}_multi_seed.json"))
    fm_data[key] = {"mean": d["mean_8seed"], "std": d["std_8seed"]}

fm_keys = ["labram", "cbramod", "reve"]
fm_labels = ["LaBraM", "CBraMod", "REVE"]
fm_bas = [fm_data[k]["mean"] for k in fm_keys]
fm_stds = [fm_data[k]["std"] for k in fm_keys]

# =====================================================================
# Figure: single panel
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 5))

n_classical = len(classical_labels)
n_fm = len(fm_labels)
n_total = n_classical + n_fm

x = np.arange(n_total)
colors = (["#999999"] * n_classical +
          ["#4C72B0", "#DD8452", "#55A868"])

all_bas = classical_bas + fm_bas
all_errs = [0] * n_classical + fm_stds
all_labels = classical_labels + [f"{l}\n(frozen LP)" for l in fm_labels]

bars = ax.bar(x, all_bas, yerr=all_errs, capsize=4,
              color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)

# Value labels
for bar, v, err in zip(bars, all_bas, all_errs):
    y = v + err + 0.015 if err > 0 else v + 0.015
    ax.text(bar.get_x() + bar.get_width() / 2, y,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Chance line
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (0.5)")

# Divider
ax.axvline(n_classical - 0.5, color="black", linestyle=":", linewidth=1, alpha=0.5)
ax.text(n_classical / 2 - 0.5, 0.72, "Classical\n(band power)",
        ha="center", fontsize=9, style="italic", color="#666")
ax.text(n_classical + n_fm / 2 - 0.5, 0.72, "FM frozen LP",
        ha="center", fontsize=9, style="italic", color="#666")

ax.set_xticks(x)
ax.set_xticklabels(all_labels, fontsize=9)
ax.set_ylabel("Balanced Accuracy")
ax.set_ylim(0.3, 0.78)
ax.set_title("Classical vs FM features on Stress (per-rec DASS, 70 recordings)",
             fontweight="bold", fontsize=12)
ax.legend(fontsize=9, loc="upper right")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/classical_vs_fm.pdf", bbox_inches="tight", dpi=300)
plt.savefig(f"{OUT_DIR}/classical_vs_fm.png", bbox_inches="tight", dpi=150)
print(f"Saved → {OUT_DIR}/classical_vs_fm.{{pdf,png}}")
