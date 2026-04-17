"""Multi-model matched-subsample variance analysis.

Extends the LaBraM-only matched-subsample analysis (F04) to CBraMod and REVE
on ADFTD and TDBRAIN. Uses the same paired frozen-vs-FT subsampling design
with 100 draws per rung and a subject-level label-permutation null.

Usage:
    python scripts/run_multimodel_matched.py
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from src import variance_analysis as va

FEATURES_CACHE = "results/features_cache"
OUTPUT = "results/studies/exp09_multimodel_matched"

MODELS = ["labram", "cbramod", "reve"]

DATASETS = {
    "adftd": {
        "rungs": [
            {"name": "N17", "n_per_label": {0: 8, 1: 9}},
            {"name": "N25", "n_per_label": {0: 11, 1: 14}},
            {"name": "N35", "n_per_label": {0: 16, 1: 19}},
            {"name": "N50", "n_per_label": {0: 22, 1: 28}},
        ],
    },
    "tdbrain": {
        "rungs": [
            {"name": "N35", "n_per_label": {0: 5, 1: 30}},
            {"name": "N65", "n_per_label": {0: 9, 1: 56}},
            {"name": "N150", "n_per_label": {0: 20, 1: 130}},
            {"name": "N300", "n_per_label": {0: 39, 1: 261}},
        ],
    },
}

N_DRAWS = 100
SEED = 42
PERM_SEED = 777


def load_frozen(model, dataset):
    """Load frozen features as (features, pids, labels)."""
    path = os.path.join(FEATURES_CACHE, f"frozen_{model}_{dataset}_19ch.npz")
    data = np.load(path)
    return data["features"], data["patient_ids"], data["labels"]


def load_ft_pooled(model, dataset):
    """Load and concatenate FT fold features as (features, pids, labels)."""
    ft_dir = os.path.join(FEATURES_CACHE, f"ft_{model}_{dataset}")
    all_f, all_p, all_l = [], [], []
    for fold in range(1, 6):
        path = os.path.join(ft_dir, f"fold{fold}_features.npz")
        data = np.load(path)
        all_f.append(data["features"])
        all_p.append(data["patient_ids"])
        all_l.append(data["labels"])
    return np.concatenate(all_f), np.concatenate(all_p), np.concatenate(all_l)


def run_matched_with_null(frozen, ft_pooled, n_per_label, n_draws, seed, perm_seed):
    """Run matched subsample + permuted null."""
    observed = va.analyze_matched_subsample(
        frozen, ft_pooled, n_per_label, n_draws=n_draws, seed=seed
    )

    # Permuted null: shuffle labels at subject level
    f_fz, s_fz, y_fz = frozen
    f_ft, s_ft, y_ft = ft_pooled
    rng = np.random.RandomState(perm_seed)

    # Build subject-level permutation
    unique_pids = np.unique(s_fz)
    pid_to_label = {pid: y_fz[s_fz == pid][0] for pid in unique_pids}
    perm_labels = rng.permutation(list(pid_to_label.values()))
    perm_map = dict(zip(unique_pids, perm_labels))

    y_fz_perm = np.array([perm_map[p] for p in s_fz])
    y_ft_perm = np.array([perm_map[p] for p in s_ft])

    null = va.analyze_matched_subsample(
        (f_fz, s_fz, y_fz_perm),
        (f_ft, s_ft, y_ft_perm),
        n_per_label, n_draws=n_draws, seed=seed
    )

    return observed, null


def summarize(result):
    """Extract key stats from analyze_matched_subsample result."""
    d = result["delta_pooled_label_fraction"]
    f = result["pooled_label_fraction_frozen"]
    t = result["pooled_label_fraction_ft"]
    return {
        "frozen_mean": round(f["mean"], 4),
        "frozen_std": round(f["std"], 4),
        "ft_mean": round(t["mean"], 4),
        "ft_std": round(t["std"], 4),
        "delta_mean": round(d["mean"], 4),
        "delta_std": round(d["std"], 4),
        "frac_ft_gt_frozen": round(d.get("frac_positive", 0), 4),
        "n_draws": result["config"]["n_draws"],
    }


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    all_results = {}

    for model in MODELS:
        for ds_name, ds_cfg in DATASETS.items():
            print(f"\n{'='*60}")
            print(f"  {model} × {ds_name}")
            print(f"{'='*60}")

            t0 = time.time()
            frozen = load_frozen(model, ds_name)
            ft_pooled = load_ft_pooled(model, ds_name)

            print(f"  Frozen: {frozen[0].shape}, FT: {ft_pooled[0].shape}")

            # Full-N label fraction (pooled SS_label / SS_total)
            fz_ss = va.nested_ss(frozen[0], frozen[1], frozen[2])
            ft_ss = va.nested_ss(ft_pooled[0], ft_pooled[1], ft_pooled[2])
            full_n_frozen = 100 * float(fz_ss["label"].sum() / max(fz_ss["total"].sum(), 1e-18))
            full_n_ft = 100 * float(ft_ss["label"].sum() / max(ft_ss["total"].sum(), 1e-18))

            key = f"{model}_{ds_name}"
            all_results[key] = {
                "model": model,
                "dataset": ds_name,
                "full_n": {
                    "frozen_label_frac": round(full_n_frozen, 4),
                    "ft_label_frac": round(full_n_ft, 4),
                    "delta": round(full_n_ft - full_n_frozen, 4),
                    "n_rec": int(frozen[0].shape[0]),
                    "n_subj": int(len(np.unique(frozen[1]))),
                },
                "rungs": {},
            }

            print(f"  Full-N: frozen={full_n_frozen:.2f}%, ft={full_n_ft:.2f}%, "
                  f"Δ={full_n_ft - full_n_frozen:+.2f}pp")

            for rung in ds_cfg["rungs"]:
                rung_name = rung["name"]
                n_per = rung["n_per_label"]
                print(f"\n  Rung {rung_name} (n_per_label={n_per})...", end=" ", flush=True)

                try:
                    obs, null = run_matched_with_null(
                        frozen, ft_pooled, n_per, N_DRAWS, SEED, PERM_SEED
                    )
                    obs_s = summarize(obs)
                    null_s = summarize(null)

                    all_results[key]["rungs"][rung_name] = {
                        "n_per_label": {str(k): v for k, v in n_per.items()},
                        "observed": obs_s,
                        "null": null_s,
                    }

                    print(f"Δ={obs_s['delta_mean']:+.2f}±{obs_s['delta_std']:.2f}pp "
                          f"(null Δ={null_s['delta_mean']:+.2f}pp) "
                          f"frac+={obs_s['frac_ft_gt_frozen']:.0%}")
                except Exception as e:
                    print(f"FAILED: {e}")
                    all_results[key]["rungs"][rung_name] = {"error": str(e)}

            elapsed = time.time() - t0
            print(f"\n  [{model}×{ds_name} done in {elapsed:.1f}s]")

    # Save
    out_path = os.path.join(OUTPUT, "matched_subsample_multimodel.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Results saved to {out_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY — Full-N label fraction delta (FT − Frozen)")
    print(f"{'='*60}")
    print(f"{'Model':>10s}  {'Dataset':>10s}  {'Frozen%':>8s}  {'FT%':>8s}  {'Δ pp':>8s}")
    print("-" * 55)
    for key, r in sorted(all_results.items()):
        fn = r["full_n"]
        print(f"{r['model']:>10s}  {r['dataset']:>10s}  "
              f"{fn['frozen_label_frac']:8.2f}  {fn['ft_label_frac']:8.2f}  "
              f"{fn['delta']:+8.2f}")


if __name__ == "__main__":
    main()
