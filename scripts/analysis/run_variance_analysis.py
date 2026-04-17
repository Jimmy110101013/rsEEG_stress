"""Run the variance-decomposition analysis on cached features and dump JSON.

Loads frozen-FM features and per-fold fine-tuned features for the cross-dataset
benchmark (UCSD Stress, ADFTD, TDBRAIN), runs the full reviewer-defensible
analysis pipeline (nested ω², mixed-effects ICC, cluster bootstrap, PERMANOVA),
and writes one JSON file consumed downstream by `build_cross_dataset_figure.py`
and the paper.

# IMPORTANT: env

This script imports `statsmodels` (via `mixed_effects_variance`), which is
broken in `timm_eeg` due to a scipy ABI mismatch. Run from `stress`:

    /raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python scripts/run_variance_analysis.py

`stress` is the unified env that handles both training/feature extraction
and stats (scipy 1.17 + statsmodels 0.14 + sklearn 1.8). The legacy
`stats_env` was removed on 2026-04-08 after verifying bit-exact reproduction
of all pooled label fractions under `stress`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src import variance_analysis as va  # noqa: E402


# ----------------------------------------------------------------------
# Path constants
# ----------------------------------------------------------------------
RESULTS = "results"
FEATURES_CACHE = f"{RESULTS}/features_cache"
FT_LABRAM = f"{FEATURES_CACHE}/ft_labram"

# Stress per-rec dass: 3/17 subjects (pid 3, 11, 17) have recordings in
# both DASS classes (increase + normal). nested_ss requires pure-label
# subjects, so we drop these 3 (15 recordings), leaving 55 rec / 14 subj.
# The pooled label fraction is computed on the pure subset only.
DATASETS = {
    "Stress": {
        "ft_dir": "results/studies/exp05_stress_feat_multiseed/s42_llrd1.0",
        "frozen": f"{FEATURES_CACHE}/frozen_labram_stress_30ch.npz",
        "drop_mixed_subjects": True,  # 3 mixed-label subjects under per-rec dass
    },
    "ADFTD": {
        "ft_dir": f"{FT_LABRAM}/adftd_2026-04-06",
        "frozen": f"{FEATURES_CACHE}/frozen_labram_adftd_19ch.npz",
    },
    "TDBRAIN": {
        "ft_dir": f"{FT_LABRAM}/tdbrain_2026-04-07",
        "frozen": f"{FEATURES_CACHE}/frozen_labram_tdbrain_19ch.npz",
    },
    # EEGMAT positive control: within-subject task labels (rest vs arithmetic).
    # Hypothesis: FT cleanly rewrites the LaBraM representation here even on
    # small data, in contrast to the projection-only failure on Stress.
    "EEGMAT": {
        "ft_dir": f"{FT_LABRAM}/eegmat_2026-04-08",
        "frozen": f"{FEATURES_CACHE}/frozen_labram_eegmat_19ch.npz",
    },
}


# ----------------------------------------------------------------------
# Matched-subsample regimes for the N-controlled frozen-vs-FT comparison.
#
# Motivation: the canonical ADFTD FT-rewrite multiplier (2.79% -> 7.70%,
# ~2.76x) compares a frozen baseline computed on 65 subjects to an FT value
# on 65 subjects. Pooled label fraction is sensitive to the subject
# denominator: at small N, SS_label is inflated relative to SS_total because
# there are fewer between-subject residual directions to absorb variance.
# To isolate N as the only variable, we subsample ADFTD to smaller subject
# counts and recompute the paired frozen-vs-FT delta across many draws.
#
# Five rungs (N=17, 25, 35, 50 preserving ADFTD's own 36:29 class ratio;
# plus N=17 with Stress's 10:7 ratio for direct comparison to the legacy
# notebook experiment).
# ----------------------------------------------------------------------
MATCHED_REGIMES = {
    # Main rungs: preserve ADFTD's own 36:29 class ratio so the only variable
    # that changes across rungs is the subject count.
    "ADFTD_matched_N17_adftdRatio": {
        "source": "ADFTD",
        "n_per_label": {0: 8, 1: 9},  # 8 HC + 9 AD = 17
        "description": "ADFTD subsampled to N=17 with ADFTD class ratio",
    },
    "ADFTD_matched_N25_adftdRatio": {
        "source": "ADFTD",
        "n_per_label": {0: 11, 1: 14},  # 11 HC + 14 AD = 25
        "description": "ADFTD subsampled to N=25 with ADFTD class ratio",
    },
    "ADFTD_matched_N35_adftdRatio": {
        "source": "ADFTD",
        "n_per_label": {0: 16, 1: 19},  # 16 HC + 19 AD = 35 (midpoint rung)
        "description": "ADFTD subsampled to N=35 with ADFTD class ratio",
    },
    "ADFTD_matched_N50_adftdRatio": {
        "source": "ADFTD",
        "n_per_label": {0: 22, 1: 28},  # 22 HC + 28 AD = 50
        "description": "ADFTD subsampled to N=50 with ADFTD class ratio",
    },
    # Bonus: N=17 with Stress's 10:7 class ratio, for direct comparison to
    # the legacy notebook experiment in notebooks/Cross_Dataset_Signal_Strength.ipynb.
    "ADFTD_matched_N17_stressRatio": {
        "source": "ADFTD",
        "n_per_label": {0: 10, 1: 7},  # 10 HC + 7 AD = 17 (matches old notebook)
        "description": "ADFTD subsampled to N=17 with Stress 10:7 ratio (legacy comparison)",
    },
    # TDBRAIN rungs (preserving its 47:312 HC:MDD ratio = ~6.6:1).
    # Purpose: check whether TDBRAIN's full-N fold-drift dilution (2.97% -> 1.47%)
    # persists or collapses under smaller N. Starts at N=35 because a 6.6:1 ratio
    # would put only 2 HC in an N=17 draw (identifiability boundary).
    "TDBRAIN_matched_N35_tdbrainRatio": {
        "source": "TDBRAIN",
        "n_per_label": {0: 5, 1: 30},  # 5 HC + 30 MDD = 35
        "description": "TDBRAIN subsampled to N=35 with TDBRAIN 6:1 ratio",
    },
    "TDBRAIN_matched_N65_tdbrainRatio": {
        "source": "TDBRAIN",
        "n_per_label": {0: 9, 1: 56},  # 9 HC + 56 MDD = 65 (ADFTD-full comparable)
        "description": "TDBRAIN subsampled to N=65 with TDBRAIN 6:1 ratio (ADFTD-full comparable)",
    },
    "TDBRAIN_matched_N150_tdbrainRatio": {
        "source": "TDBRAIN",
        "n_per_label": {0: 20, 1: 130},  # 20 HC + 130 MDD = 150
        "description": "TDBRAIN subsampled to N=150 with TDBRAIN 6:1 ratio",
    },
    "TDBRAIN_matched_N300_tdbrainRatio": {
        "source": "TDBRAIN",
        "n_per_label": {0: 39, 1: 261},  # 39 HC + 261 MDD = 300 (near-full)
        "description": "TDBRAIN subsampled to N=300 with TDBRAIN 6:1 ratio",
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
    p.add_argument(
        "--matched", nargs="*", default=None,
        help="Run matched-subsample regimes in addition to the main analysis. "
        "Pass regime names (e.g. ADFTD_matched_N17_adftdRatio) or 'all' for "
        "every regime defined in MATCHED_REGIMES. If omitted, matched "
        "analysis is skipped and the main output JSON is unchanged.",
    )
    p.add_argument("--matched-n-draws", type=int, default=100,
                   help="Number of random draws per matched regime")
    p.add_argument("--matched-seed", type=int, default=42,
                   help="Master RNG seed for matched subsampling")
    p.add_argument("--matched-out", default=None,
                   help="Optional separate JSON path for matched results. "
                   "If omitted, matched results are merged into --out under "
                   "key 'matched_subsamples'.")
    p.add_argument("--permute-labels-check", action="store_true",
                   help="Also run each matched regime with labels permuted "
                   "at the subject level (paired — same permutation on "
                   "frozen and FT). This is the null test: any FT-frozen "
                   "delta should collapse to ~0 under label permutation. "
                   "Results stored under the key 'permuted_matched_subsamples'.")
    p.add_argument("--permute-seed", type=int, default=777,
                   help="Seed for the subject-level label permutation.")
    args = p.parse_args()

    # Resolve which matched regimes to run (None = skip).
    matched_regimes: list[str] = []
    if args.matched is not None:
        if len(args.matched) == 0 or "all" in args.matched:
            matched_regimes = list(MATCHED_REGIMES.keys())
        else:
            unknown = [m for m in args.matched if m not in MATCHED_REGIMES]
            if unknown:
                raise SystemExit(
                    f"Unknown matched regime(s): {unknown}. "
                    f"Available: {list(MATCHED_REGIMES.keys())}"
                )
            matched_regimes = list(args.matched)

    # Decide whether to run the main per-dataset analysis.
    main_exists = os.path.isfile(args.out)
    run_main = args.force or not main_exists

    if not run_main and not matched_regimes:
        print(f"[skip] {args.out} exists; pass --force to recompute "
              "or --matched to add matched-subsample analysis")
        return

    if run_main:
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
    else:
        # Matched-only mode: load the existing main JSON so we can merge
        # matched results into it (or just keep a reference for separate out).
        print(f"[matched-only] loading existing {args.out}")
        with open(args.out) as f:
            results = json.load(f)

    # Source-dataset feature cache: avoid re-loading frozen+FT if multiple
    # matched regimes share a source dataset.
    source_cache: dict = {}

    def _load_source(src_name: str):
        if src_name in source_cache:
            return source_cache[src_name]
        src_cfg = DATASETS[src_name]
        ft_per_fold_ = va.load_ft_features_per_fold(src_cfg["ft_dir"])
        frozen_ = va.load_frozen_features(src_cfg["frozen"], src_cfg["ft_dir"])
        frozen_repack_ = (frozen_[0], frozen_[2], frozen_[1])
        ft_f_ = np.concatenate([t[0] for t in ft_per_fold_])
        ft_y_ = np.concatenate([t[1] for t in ft_per_fold_])
        ft_p_ = np.concatenate([t[2] for t in ft_per_fold_])
        ft_pooled_repack_ = (ft_f_, ft_p_, ft_y_)
        source_cache[src_name] = (frozen_repack_, ft_pooled_repack_, ft_per_fold_)
        return source_cache[src_name]

    if not run_main:
        # Seed the cache from disk — main loop won't populate it.
        pass

    for name in args.datasets if run_main else []:
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

        # Drop mixed-label subjects if flagged (e.g. Stress per-rec dass
        # has 3 subjects with recordings in both classes).
        if cfg.get("drop_mixed_subjects", False):
            def _find_mixed(pids, labs):
                mixed = set()
                for p in np.unique(pids):
                    if len(np.unique(labs[pids == p])) > 1:
                        mixed.add(p)
                return mixed

            mixed_pids = _find_mixed(frozen[2], frozen[1])
            if mixed_pids:
                print(f"  Dropping {len(mixed_pids)} mixed-label subjects: {sorted(mixed_pids)}")
                keep_fz = np.array([p not in mixed_pids for p in frozen[2]])
                frozen = (frozen[0][keep_fz], frozen[1][keep_fz], frozen[2][keep_fz])
                new_ft = []
                for f, y, p_ in ft_per_fold:
                    keep = np.array([p not in mixed_pids for p in p_])
                    new_ft.append((f[keep], y[keep], p_[keep]))
                ft_per_fold = new_ft
                print(f"  After drop: frozen {frozen[0].shape[0]} rec, "
                      f"FT {sum(t[0].shape[0] for t in ft_per_fold)} rec (pooled)")

        # Note: nested_ss expects (features, subject, label).
        # load_*_features returns (features, labels, pids).
        # Repack to (features, subject, label) for analyze_dataset.
        frozen_repack = (frozen[0], frozen[2], frozen[1])
        ft_repack = [(f, p_, y) for (f, y, p_) in ft_per_fold]

        # Populate the source cache so matched regimes that target this
        # dataset don't re-load from disk.
        ft_f_all = np.concatenate([t[0] for t in ft_per_fold])
        ft_y_all = np.concatenate([t[1] for t in ft_per_fold])
        ft_p_all = np.concatenate([t[2] for t in ft_per_fold])
        source_cache[name] = (
            frozen_repack,
            (ft_f_all, ft_p_all, ft_y_all),
            ft_per_fold,
        )

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

    # ------------------------------------------------------------------
    # Matched-subsample regimes (N-controlled frozen-vs-FT comparison).
    # ------------------------------------------------------------------
    if matched_regimes:
        print(f"\n=== Matched-subsample regimes ({len(matched_regimes)}) ===")
        print(f"  n_draws={args.matched_n_draws}  seed={args.matched_seed}")
        if args.permute_labels_check:
            print(f"  [null test] subject-level label permutation "
                  f"seed={args.permute_seed}")
        matched_results: dict = {}
        permuted_results: dict = {}
        # Cache permuted tuples per source (one permutation per source, shared
        # across all regimes targeting the same source so permuted results are
        # directly comparable across rungs).
        permuted_source_cache: dict = {}
        matched_t0 = time.time()
        for regime_name in matched_regimes:
            cfg = MATCHED_REGIMES[regime_name]
            src = cfg["source"]
            print(f"\n  --- {regime_name} ---")
            print(f"      source={src}  n_per_label={cfg['n_per_label']}")
            print(f"      {cfg['description']}")

            try:
                frozen_pack, ft_pooled_pack, _ = _load_source(src)
            except Exception as exc:
                print(f"      [skip] failed to load source {src}: {exc}")
                continue

            rt0 = time.time()
            regime_out = va.analyze_matched_subsample(
                frozen_pack, ft_pooled_pack,
                n_per_label=cfg["n_per_label"],
                n_draws=args.matched_n_draws,
                seed=args.matched_seed,
            )
            regime_out["description"] = cfg["description"]
            regime_out["source"] = src
            matched_results[regime_name] = regime_out

            fz = regime_out["pooled_label_fraction_frozen"]
            ft_ = regime_out["pooled_label_fraction_ft"]
            dl = regime_out["delta_pooled_label_fraction"]
            print(f"      frozen pooled label frac: "
                  f"{fz['mean']:.4f} ± {fz['std']:.4f}  "
                  f"[{fz['ci_low']:.4f}, {fz['ci_high']:.4f}]")
            print(f"      FT     pooled label frac: "
                  f"{ft_['mean']:.4f} ± {ft_['std']:.4f}  "
                  f"[{ft_['ci_low']:.4f}, {ft_['ci_high']:.4f}]")
            print(f"      delta (FT - frozen):      "
                  f"{dl['mean']:.4f} ± {dl['std']:.4f}  "
                  f"[{dl['ci_low']:.4f}, {dl['ci_high']:.4f}]  "
                  f"frac_positive={dl['frac_positive']:.3f}")
            print(f"      identifiable draws: "
                  f"{regime_out['identifiable_draws']}/{args.matched_n_draws}")
            print(f"      elapsed: {time.time() - rt0:.1f}s")

            # ---- Null: subject-level label permutation ----
            if args.permute_labels_check:
                if src not in permuted_source_cache:
                    # Build a permuted label vector at subject level, apply
                    # the SAME permutation to frozen and FT (subject sets are
                    # identical between the two regimes, so the subject->label
                    # remap is shared).
                    f_fz, s_fz, y_fz = frozen_pack
                    f_ft, s_ft, y_ft = ft_pooled_pack
                    y_fz_perm = va.permute_labels_by_subject(
                        s_fz, y_fz, seed=args.permute_seed,
                    )
                    # For FT we need to apply the *same* subject->label map.
                    # Re-build the map from (s_fz, y_fz_perm) and apply to s_ft.
                    pid_remap = {
                        sid: lab for sid, lab in zip(s_fz, y_fz_perm)
                    }
                    y_ft_perm = np.array(
                        [pid_remap[sid] for sid in s_ft], dtype=y_ft.dtype
                    )
                    permuted_source_cache[src] = (
                        (f_fz, s_fz, y_fz_perm),
                        (f_ft, s_ft, y_ft_perm),
                    )
                frozen_perm_pack, ft_perm_pack = permuted_source_cache[src]

                pt0 = time.time()
                perm_out = va.analyze_matched_subsample(
                    frozen_perm_pack, ft_perm_pack,
                    n_per_label=cfg["n_per_label"],
                    n_draws=args.matched_n_draws,
                    seed=args.matched_seed,
                )
                perm_out["description"] = cfg["description"] + " [PERMUTED NULL]"
                perm_out["source"] = src
                permuted_results[regime_name] = perm_out

                pfz = perm_out["pooled_label_fraction_frozen"]
                pft = perm_out["pooled_label_fraction_ft"]
                pdl = perm_out["delta_pooled_label_fraction"]
                print(f"      [null] frozen: {pfz['mean']:.4f} ± {pfz['std']:.4f}  "
                      f"FT: {pft['mean']:.4f} ± {pft['std']:.4f}  "
                      f"Δ: {pdl['mean']:+.4f} ± {pdl['std']:.4f}  "
                      f"frac+={pdl['frac_positive']:.3f}  ({time.time() - pt0:.1f}s)")

        print(f"\n  Total matched analysis: {time.time() - matched_t0:.1f}s")

        matched_wrapper = {
            "config": {
                "n_draws": args.matched_n_draws,
                "seed": args.matched_seed,
                "regimes": matched_regimes,
                "permute_labels_check": bool(args.permute_labels_check),
                "permute_seed": int(args.permute_seed) if args.permute_labels_check else None,
            },
            "regimes": matched_results,
        }
        if args.permute_labels_check:
            matched_wrapper["permuted_regimes"] = permuted_results

        if args.matched_out:
            # Write matched results to a separate file; leave main JSON alone.
            os.makedirs(os.path.dirname(args.matched_out) or ".", exist_ok=True)
            with open(args.matched_out, "w") as f:
                json.dump(matched_wrapper, f, indent=2)
            print(f"Saved matched → {args.matched_out}")
        else:
            # Merge into the main results dict under 'matched_subsamples'.
            results["matched_subsamples"] = matched_wrapper

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # In matched-only mode we still rewrite args.out (the existing file is
    # already loaded into `results` and merged). Separate --matched-out was
    # already written above; rewrite main only if we added matched content
    # into `results` or re-ran the main analysis.
    if run_main or (matched_regimes and not args.matched_out):
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
