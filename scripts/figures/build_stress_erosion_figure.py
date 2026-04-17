"""Stress erosion figure (exp03).

Panel A: Frozen LP vs Best FT BA per model (3 FMs, error bars = seed std)
Panel B: LaBraM frozen LP vs FT vs permutation null distribution
Panel C: Δ(FT − Frozen) per model showing erosion vs injection
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 10, "font.family": "sans-serif"})

EXP_DIR = "results/studies/exp03_stress_erosion"
SWEEP_DIR = "results/hp_sweep/20260410_dass"
OUT_DIR = EXP_DIR

model_colors = {"labram": "#4C72B0", "cbramod": "#DD8452", "reve": "#55A868"}

# =====================================================================
# Load data
# =====================================================================
# Frozen LP (8-seed)
frozen = {}
for key in ["labram", "cbramod", "reve"]:
    with open(f"{EXP_DIR}/frozen_lp/{key}_multi_seed.json") as f:
        d = json.load(f)
    frozen[key] = {
        "mean": d["mean_8seed"],
        "std": d["std_8seed"],
        "per_seed": list(d["per_seed_ba"].values()),
    }

# Best FT from HP sweep (3-seed)
import os
from collections import defaultdict

ft_best = {}
for key in ["labram", "cbramod", "reve"]:
    model_dir = os.path.join(SWEEP_DIR, key)
    configs = defaultdict(list)
    for run in os.listdir(model_dir):
        summary = os.path.join(model_dir, run, "summary.json")
        if os.path.exists(summary):
            s = json.load(open(summary))
            ba = s.get("subject_bal_acc", 0)
            parts = run.rsplit("_s", 1)
            config = parts[0]
            configs[config].append(ba)

    best_mean, best_std, best_seeds = -1, 0, []
    for config, bas in configs.items():
        mean = np.mean(bas)
        if mean > best_mean:
            best_mean = mean
            best_std = np.std(bas)
            best_seeds = bas
    ft_best[key] = {"mean": best_mean, "std": best_std, "per_seed": best_seeds}

# Permutation null (LaBraM only, 10 perms)
analysis = json.load(open(f"{EXP_DIR}/analysis.json"))
null_bas = list(analysis["ft_null"]["per_perm_ba"].values())
null_mean = analysis["ft_null"]["mean"]
null_std = analysis["ft_null"]["std"]

# LaBraM canonical FT (3-seed)
labram_canonical = analysis["ft_real"]["primary_3seed_llrd1.0"]

print("=== Summary ===")
for key, label in [("labram", "LaBraM"), ("cbramod", "CBraMod"), ("reve", "REVE")]:
    delta = ft_best[key]["mean"] - frozen[key]["mean"]
    mode = "erosion" if delta < 0 else "injection"
    print(f"{label}: Frozen={frozen[key]['mean']:.3f}±{frozen[key]['std']:.3f}, "
          f"FT={ft_best[key]['mean']:.3f}±{ft_best[key]['std']:.3f}, "
          f"Δ={delta:+.3f} ({mode})")

# =====================================================================
# Figure: 3 panels
# =====================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

models = [("labram", "LaBraM"), ("cbramod", "CBraMod"), ("reve", "REVE")]
x = np.arange(len(models))
width = 0.3

# --- Panel A: Frozen LP vs Best FT ---
frozen_vals = [frozen[k]["mean"] for k, _ in models]
frozen_errs = [frozen[k]["std"] for k, _ in models]
ft_vals = [ft_best[k]["mean"] for k, _ in models]
ft_errs = [ft_best[k]["std"] for k, _ in models]

bars1 = ax1.bar(x - width / 2, frozen_vals, width, yerr=frozen_errs,
                capsize=4, label="Frozen LP (8-seed)",
                color=[model_colors[k] for k, _ in models],
                edgecolor="white", linewidth=0.5, alpha=0.9)
bars2 = ax1.bar(x + width / 2, ft_vals, width, yerr=ft_errs,
                capsize=4, label="Best FT (3-seed)",
                color=[model_colors[k] for k, _ in models],
                edgecolor="black", linewidth=0.8, alpha=0.5,
                hatch="//")

for bar, v in zip(bars1, frozen_vals):
    ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.04,
             f"{v:.2f}", ha="center", va="bottom", fontsize=9)
for bar, v in zip(bars2, ft_vals):
    ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.04,
             f"{v:.2f}", ha="center", va="bottom", fontsize=9)

ax1.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
ax1.set_xticks(x)
ax1.set_xticklabels([l for _, l in models])
ax1.set_ylabel("Balanced Accuracy")
ax1.set_ylim(0.3, 0.75)
ax1.set_title("A. Frozen LP vs Best FT", fontweight="bold", fontsize=12)
ax1.legend(fontsize=8, loc="upper right")

# --- Panel B: LaBraM detail — frozen vs FT vs null distribution ---
# Null histogram
ax2.hist(null_bas, bins=8, color="lightgray", edgecolor="gray",
         alpha=0.7, label=f"Permutation null (n=10)")

# Real FT marker
ax2.axvline(labram_canonical["mean"], color="#F44336", linewidth=2.5,
            linestyle="-", label=f"Real FT: {labram_canonical['mean']:.3f}")

# Frozen LP marker
ax2.axvline(frozen["labram"]["mean"], color="#4C72B0", linewidth=2.5,
            linestyle="-", label=f"Frozen LP: {frozen['labram']['mean']:.3f}")

# Individual FT seeds
for ba in labram_canonical["bas"]:
    ax2.axvline(ba, color="#F44336", linewidth=1, linestyle=":", alpha=0.5)

# Individual frozen seeds
for ba in frozen["labram"]["per_seed"][:3]:
    ax2.axvline(ba, color="#4C72B0", linewidth=1, linestyle=":", alpha=0.5)

ax2.axvline(0.5, color="gray", linestyle="--", linewidth=1)
ax2.set_xlabel("Balanced Accuracy")
ax2.set_ylabel("Count")
ax2.set_title("B. LaBraM: FT vs permutation null", fontweight="bold", fontsize=12)
ax2.legend(fontsize=8, loc="upper left")

# --- Panel C: Δ(FT − Frozen) ---
deltas = [ft_best[k]["mean"] - frozen[k]["mean"] for k, _ in models]
# Error propagation: sqrt(std_ft² + std_frozen²)
delta_errs = [np.sqrt(ft_best[k]["std"]**2 + frozen[k]["std"]**2) for k, _ in models]

bars = ax3.bar(x, deltas, yerr=delta_errs, capsize=4,
               color=[model_colors[k] for k, _ in models],
               edgecolor="white", linewidth=0.5, alpha=0.8)

for bar, d in zip(bars, deltas):
    y_pos = d + 0.01 if d >= 0 else d - 0.025
    va = "bottom" if d >= 0 else "top"
    label = f"{d:+.1f}pp"
    ax3.text(bar.get_x() + bar.get_width() / 2, y_pos,
             label, ha="center", va=va, fontsize=11, fontweight="bold")

ax3.axhline(0, color="gray", linestyle="--", linewidth=1)
ax3.set_xticks(x)
ax3.set_xticklabels([l for _, l in models])
ax3.set_ylabel("Δ BA (FT − Frozen)")
ax3.set_ylim(-0.15, 0.15)
ax3.set_title("C. FT effect: erosion vs injection", fontweight="bold", fontsize=12)

# Annotate erosion/injection zones
ax3.text(0.02, 0.95, "← erosion", transform=ax3.transAxes,
         fontsize=9, color="red", va="top", style="italic")
ax3.text(0.02, 0.05, "injection →", transform=ax3.transAxes,
         fontsize=9, color="green", va="bottom", style="italic")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/stress_erosion.pdf", bbox_inches="tight", dpi=300)
plt.savefig(f"{OUT_DIR}/stress_erosion.png", bbox_inches="tight", dpi=150)
print(f"\nSaved → {OUT_DIR}/stress_erosion.{{pdf,png}}")
