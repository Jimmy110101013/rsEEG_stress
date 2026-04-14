"""Statistical hardening for UCSD Stress findings.

1. Bootstrap CIs on per-seed BA for all 3 models (frozen LP + best FT)
2. Sign test: does trial-level BA consistently exceed subject-level BA?
3. Permutation test on frozen LP vs best FT difference

Usage:
    python scripts/stat_hardening.py
"""

import json
import os
import numpy as np
from scipy import stats

RESULTS_ROOT = "results"
FROZEN_LP_DIR = os.path.join(RESULTS_ROOT, "studies/exp03_stress_erosion/frozen_lp")
HP_SWEEP_DIR = os.path.join(RESULTS_ROOT, "hp_sweep/20260410_dass")
OUTPUT_DIR = os.path.join(RESULTS_ROOT, "studies/exp10_stat_hardening")

# Best FT configs per model (from sweep summary)
BEST_FT = {
    "labram": {"lr": "1e-4", "elrs": "1.0"},
    "cbramod": {"lr": "1e-5", "elrs": "0.1"},
    "reve": {"lr": "3e-5", "elrs": "0.1"},
}
SEEDS = [42, 123, 2024]


def load_frozen_lp_seeds(model):
    path = os.path.join(FROZEN_LP_DIR, f"{model}_multi_seed.json")
    with open(path) as f:
        data = json.load(f)
    return data["per_seed_ba"], data["seeds"]


def load_ft_seed_bas(model, lr, elrs):
    bas = []
    for seed in SEEDS:
        run_dir = os.path.join(
            HP_SWEEP_DIR, model,
            f"{model}_encoderlrscale{elrs}_lr{lr}_s{seed}",
            "summary.json"
        )
        with open(run_dir) as f:
            data = json.load(f)
        bas.append(data["subject_bal_acc"])
    return bas


def bootstrap_ci(values, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    values = np.array(values)
    n = len(values)
    boot_means = np.array([
        rng.choice(values, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, 100 * alpha)
    hi = np.percentile(boot_means, 100 * (1 - alpha))
    return float(lo), float(hi), float(np.mean(boot_means))


def permutation_test_paired(a, b, n_perm=10000, seed=42):
    """Two-sided permutation test for paired differences."""
    rng = np.random.RandomState(seed)
    a, b = np.array(a), np.array(b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    obs_diff = a.mean() - b.mean()
    count = 0
    for _ in range(n_perm):
        swap = rng.randint(0, 2, size=n).astype(bool)
        perm_a = np.where(swap, b, a)
        perm_b = np.where(swap, a, b)
        if abs(perm_a.mean() - perm_b.mean()) >= abs(obs_diff):
            count += 1
    return float(obs_diff), count / n_perm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    # ──────────────────────────────────────────────
    # 1. Bootstrap CIs on BA
    # ──────────────────────────────────────────────
    print("=" * 60)
    print("1. Bootstrap 95% CIs on Balanced Accuracy")
    print("=" * 60)

    ci_results = {}
    for model in ["labram", "cbramod", "reve"]:
        # Frozen LP (8 seeds)
        seed_bas, seeds = load_frozen_lp_seeds(model)
        frozen_vals = [seed_bas[str(s)] for s in seeds]
        lo, hi, boot_mean = bootstrap_ci(frozen_vals)
        ci_results[f"{model}_frozen"] = {
            "mean": float(np.mean(frozen_vals)),
            "std": float(np.std(frozen_vals)),
            "n_seeds": len(frozen_vals),
            "ci_95_lo": lo, "ci_95_hi": hi,
            "values": frozen_vals,
        }
        print(f"  {model} Frozen LP: {np.mean(frozen_vals):.4f} [{lo:.4f}, {hi:.4f}] (n={len(frozen_vals)})")

        # Best FT (3 seeds)
        cfg = BEST_FT[model]
        ft_vals = load_ft_seed_bas(model, cfg["lr"], cfg["elrs"])
        lo, hi, boot_mean = bootstrap_ci(ft_vals)
        ci_results[f"{model}_ft"] = {
            "mean": float(np.mean(ft_vals)),
            "std": float(np.std(ft_vals)),
            "n_seeds": len(ft_vals),
            "ci_95_lo": lo, "ci_95_hi": hi,
            "config": cfg,
            "values": ft_vals,
        }
        print(f"  {model} Best FT:   {np.mean(ft_vals):.4f} [{lo:.4f}, {hi:.4f}] (n={len(ft_vals)})")

    results["bootstrap_ci"] = ci_results

    # ──────────────────────────────────────────────
    # 2. Permutation test: Frozen LP vs Best FT
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. Permutation test: Frozen LP vs Best FT (paired by seed)")
    print("=" * 60)

    perm_results = {}
    for model in ["labram", "cbramod", "reve"]:
        seed_bas, _ = load_frozen_lp_seeds(model)
        frozen_3seed = [seed_bas[str(s)] for s in SEEDS]
        cfg = BEST_FT[model]
        ft_vals = load_ft_seed_bas(model, cfg["lr"], cfg["elrs"])

        diff, p = permutation_test_paired(frozen_3seed, ft_vals)
        perm_results[model] = {
            "frozen_3seed": frozen_3seed,
            "ft_3seed": ft_vals,
            "mean_diff_frozen_minus_ft": diff,
            "p_value_two_sided": p,
        }
        direction = "frozen > FT (erosion)" if diff > 0 else "FT > frozen (injection)"
        print(f"  {model}: Δ = {diff:+.4f} ({direction}), p = {p:.4f}")

    results["permutation_frozen_vs_ft"] = perm_results

    # ──────────────────────────────────────────────
    # 3. Sign test: trial-level > subject-level BA
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. Sign test: trial-level BA > subject-level BA")
    print("=" * 60)

    # Historical trial-level BAs (subject-dass, single seed — only reference)
    # and subject-level BAs (subject-dass, single seed)
    # These are the known gaps from F01 (docs/findings.md)
    trial_bas = {"labram": 0.862, "reve": 0.770, "cbramod": 0.712}
    subj_bas = {"labram": 0.656, "reve": 0.553, "cbramod": 0.488}

    gaps = []
    for model in ["labram", "cbramod", "reve"]:
        gap = trial_bas[model] - subj_bas[model]
        gaps.append(gap)
        print(f"  {model}: trial={trial_bas[model]:.3f} - subj={subj_bas[model]:.3f} = gap={gap:+.3f}")

    # Sign test: all 3 gaps are positive
    n_positive = sum(1 for g in gaps if g > 0)
    n_total = len(gaps)
    # Under H0 (no consistent direction), P(all 3 positive) = 0.5^3 = 0.125
    p_sign = stats.binomtest(n_positive, n_total, 0.5, alternative="greater").pvalue
    print(f"\n  Sign test: {n_positive}/{n_total} positive, p = {p_sign:.4f}")
    print(f"  Mean gap: {np.mean(gaps):+.3f} pp")

    # Also: Wilcoxon signed-rank (n=3 is minimum)
    stat_w, p_w = stats.wilcoxon(
        [trial_bas[m] for m in ["labram", "cbramod", "reve"]],
        [subj_bas[m] for m in ["labram", "cbramod", "reve"]],
        alternative="greater"
    )
    print(f"  Wilcoxon signed-rank: W={stat_w:.1f}, p={p_w:.4f}")

    results["sign_test_trial_vs_subject"] = {
        "trial_bas": trial_bas,
        "subj_bas": subj_bas,
        "gaps": {m: trial_bas[m] - subj_bas[m] for m in trial_bas},
        "n_positive": n_positive,
        "n_total": n_total,
        "sign_test_p": float(p_sign),
        "wilcoxon_W": float(stat_w),
        "wilcoxon_p": float(p_w),
        "note": "Both BAs from subject-dass single-seed (historical reference). Gap direction is the claim, not absolute values.",
    }

    # ──────────────────────────────────────────────
    # 4. Effect size: Cohen's d for frozen vs FT
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. Cohen's d: Frozen LP vs Best FT")
    print("=" * 60)

    cohens_d = {}
    for model in ["labram", "cbramod", "reve"]:
        seed_bas, _ = load_frozen_lp_seeds(model)
        frozen_vals = np.array([seed_bas[str(s)] for s in SEEDS])
        cfg = BEST_FT[model]
        ft_vals = np.array(load_ft_seed_bas(model, cfg["lr"], cfg["elrs"]))
        pooled_std = np.sqrt((frozen_vals.std()**2 + ft_vals.std()**2) / 2)
        d = (frozen_vals.mean() - ft_vals.mean()) / pooled_std if pooled_std > 0 else 0
        cohens_d[model] = {
            "d": float(d),
            "frozen_mean": float(frozen_vals.mean()),
            "ft_mean": float(ft_vals.mean()),
            "pooled_std": float(pooled_std),
        }
        direction = "erosion" if d > 0 else "injection"
        print(f"  {model}: d = {d:+.2f} ({direction})")

    results["cohens_d"] = cohens_d

    # Save
    out_path = os.path.join(OUTPUT_DIR, "stat_hardening.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
