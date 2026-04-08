"""Run the variance-decomposition analysis on cached features and dump JSON.

Loads frozen-FM features and per-fold fine-tuned features for the cross-dataset
benchmark (UCSD Stress, ADFTD, TDBRAIN), runs the full reviewer-defensible
analysis pipeline (nested ω², mixed-effects ICC, cluster bootstrap, PERMANOVA),
and writes one JSON file consumed downstream by `build_cross_dataset_figure.py`
and the paper.

# IMPORTANT: env

This script imports `statsmodels` (via `mixed_effects_variance`), which is
broken in `timm_eeg` due to a scipy ABI mismatch. Run from `stats_env`:

    conda run -n stats_env python scripts/run_variance_analysis.py

For training/feature extraction continue to use `timm_eeg`. The two envs share
data only via on-disk `.npz`/`.json` files (see memory:feedback_stats_env).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src import variance_analysis as va  # noqa: E402


# ----------------------------------------------------------------------
# Path constants (matched to scripts/build_cross_dataset_figure.py)
# ----------------------------------------------------------------------
RESULTS = "results"
CROSS_DIR = f"{RESULTS}/cross_dataset"

DATASETS = {
    "Stress": {
        "ft_dir": f"{RESULTS}/20260406_0419_ft_subjectdass_aug75_labram_feat",
        "frozen": f"{CROSS_DIR}/features_stress_19ch.npz",
    },
    "ADFTD": {
        "ft_dir": f"{RESULTS}/20260406_0935_ft_dass_aug75_labram_adftd_feat",
        "frozen": f"{CROSS_DIR}/features_adftd_19ch.npz",
    },
    "TDBRAIN": {
        "ft_dir": f"{RESULTS}/20260407_1533_ft_dass_aug75_labram_tdbrain_feat",
        "frozen": f"{CROSS_DIR}/features_tdbrain_19ch.npz",
    },
}


# ----------------------------------------------------------------------
def load_ba_from_summary(run_dir: str) -> float | None:
    """Read the recording-level BA from the run's summary.json (best-effort)."""
    path = os.path.join(run_dir, "summary.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            s = json.load(f)
        return float(s.get("subject_bal_acc", s.get("bal_acc")))
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets", nargs="+", default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        help="Which datasets to analyze",
    )
    p.add_argument("--n-boot", type=int, default=1000,
                   help="Cluster bootstrap iterations")
    p.add_argument("--n-perm", type=int, default=999,
                   help="PERMANOVA permutations")
    p.add_argument("--mixed-dims", type=int, default=None,
                   help="Number of feature dims for mixed-effects "
                        "(default: all 200)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="paper/figures/variance_analysis.json")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if output exists")
    p.add_argument("--skip-permanova", action="store_true")
    p.add_argument("--skip-mixed-effects", action="store_true")
    args = p.parse_args()

    if os.path.isfile(args.out) and not args.force:
        print(f"[skip] {args.out} exists; pass --force to recompute")
        return

    results = {
        "config": {
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "mixed_effects_max_dims": args.mixed_dims,
            "seed": args.seed,
            "datasets": args.datasets,
        },
        "datasets": {},
    }

    for name in args.datasets:
        cfg = DATASETS[name]
        print(f"\n=== {name} ===")
        print(f"  ft_dir: {cfg['ft_dir']}")
        print(f"  frozen: {cfg['frozen']}")

        if not os.path.isdir(cfg["ft_dir"]):
            print(f"  [skip] FT dir missing")
            continue
        if not os.path.isfile(cfg["frozen"]):
            print(f"  [skip] frozen npz missing")
            continue

        ba = load_ba_from_summary(cfg["ft_dir"])
        if ba is not None:
            print(f"  BA (FT, subject-level): {ba:.4f}")

        t0 = time.time()
        ft_per_fold = va.load_ft_features_per_fold(cfg["ft_dir"])
        print(f"  FT folds: {len(ft_per_fold)}")
        for k, (f, y, p_) in enumerate(ft_per_fold):
            print(f"    fold {k+1}: feats={f.shape} subjects={len(set(p_.tolist()))}")

        frozen = va.load_frozen_features(cfg["frozen"], cfg["ft_dir"])
        print(f"  frozen: feats={frozen[0].shape} "
              f"subjects={len(set(frozen[2].tolist()))}")

        # Note: nested_ss expects (features, subject, label).
        # load_*_features returns (features, labels, pids).
        # Repack to (features, subject, label) for analyze_dataset.
        frozen_repack = (frozen[0], frozen[2], frozen[1])
        ft_repack = [(f, p_, y) for (f, y, p_) in ft_per_fold]

        analysis = va.analyze_dataset(
            frozen_repack, ft_repack,
            n_boot=args.n_boot,
            n_perm=args.n_perm,
            mixed_effects_max_dims=args.mixed_dims,
            do_permanova=not args.skip_permanova,
            do_mixed_effects=not args.skip_mixed_effects,
            seed=args.seed,
        )
        results["datasets"][name] = {
            "ba": ba,
            "analysis": analysis,
        }

        # Quick console summary.
        def _print_regime(label: str, r: dict):
            o = r["nested_omega2"]
            print(f"  {label}  ω²_label={o['label']:.3f}  "
                  f"ω²_subject|label={o['subject_within_label']:.3f}  "
                  f"ratio={o['ratio_subject_to_label']:.2f}")
            if "mixed_effects" in r and "error" not in r["mixed_effects"]:
                me = r["mixed_effects"]
                print(f"  {label}  ICC={me['icc_subject_mean']:.3f}  "
                      f"frac_lbl={me['frac_label_mean']:.3f}  "
                      f"conv={me['n_converged']}/{me['n_dims_tried']}")
            if "permanova" in r:
                pn = r["permanova"]
                print(f"  {label}  PERMANOVA F={pn['pseudo_F']:.2f} "
                      f"R²={pn['R2']:.3f} p={pn['p_value']:.4f}")

        _print_regime("FROZEN ", analysis["frozen"])
        _print_regime("FT_POOL", analysis["ft_pooled"])
        ft = analysis["ft_aggregated"]
        print(f"  FT_FOLD (mean over {ft['omega2_label']['n_folds']} folds)  "
              f"ω²_lbl={ft['omega2_label']['mean']:.3f}  "
              f"ω²_subj={ft['omega2_subject_within_label']['mean']:.3f}  "
              f"ratio={ft['ratio_subject_to_label']['mean']:.2f}")
        if "icc_subject" in ft:
            print(f"  FT_FOLD ICC={ft['icc_subject']['mean']:.3f}  "
                  f"[{ft['icc_subject']['ci_low']:.3f}, {ft['icc_subject']['ci_high']:.3f}]")

        print(f"  elapsed: {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
