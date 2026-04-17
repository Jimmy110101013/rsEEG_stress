"""Variance decomposition for non-FM baseline features on Stress.

Loads trained EEGNet/ShallowConvNet features from exp15 multi-seed runs
and computes pooled label fraction + bootstrap CIs. Compares against
FM frozen features to answer: "Is subject dominance FM-specific?"

Usage:
    /raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python \
        scripts/analyze_nonfm_variance.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src import variance_analysis as va

STUDY_ROOT = Path("results/studies/exp15_nonfm_baselines")
FEATURES_CACHE = Path("results/features_cache")

# FM frozen feature references for comparison
FM_FROZEN_REFS = {
    "LaBraM (frozen)":  FEATURES_CACHE / "frozen_labram_stress_30ch.npz",
    "CBraMod (frozen)": FEATURES_CACHE / "frozen_cbramod_stress_30ch.npz",
    "REVE (frozen)":    FEATURES_CACHE / "frozen_reve_stress_30ch.npz",
}

# Per-rec dass labels: 3/17 subjects (pid 3, 11, 17) have recordings in
# both DASS classes. nested_ss requires pure-label subjects, so we drop them.
MIXED_PIDS = {3, 11, 17}
SEEDS = [42, 123, 2024]


def _drop_mixed(features, labels, pids):
    """Drop recordings from mixed-label subjects."""
    mask = ~np.isin(pids, list(MIXED_PIDS))
    return features[mask], labels[mask], pids[mask]


def _compute_plf(features, labels, pids) -> dict:
    """Compute pooled label fraction with bootstrap CI."""
    features, labels, pids = _drop_mixed(features, labels, pids)

    n_subj = len(np.unique(pids))
    n_rec = len(features)
    n_pos = int(labels.sum())

    # Basic pooled label fraction
    ss = va.nested_ss(features, pids, labels)
    plf = float(ss["label"].sum() / max(ss["total"].sum(), 1e-18))

    # Bootstrap CI (resample at subject level)
    def plf_stat(f, s, y):
        ss_ = va.nested_ss(f, s, y)
        return float(ss_["label"].sum() / max(ss_["total"].sum(), 1e-18))

    boot = va.cluster_bootstrap(features, pids, labels,
                                plf_stat, n_boot=2000, seed=42)

    return {
        "pooled_label_fraction": plf,
        "bootstrap_mean": float(boot["mean"]),
        "bootstrap_std": float(boot["std"]),
        "ci_low": float(boot["ci_low"]),
        "ci_high": float(boot["ci_high"]),
        "n_subjects": n_subj,
        "n_recordings": n_rec,
        "n_positive": n_pos,
        "embed_dim": features.shape[1],
    }


def analyze_fm_frozen() -> dict:
    """Compute PLF for FM frozen features as reference."""
    results = {}
    for name, path in FM_FROZEN_REFS.items():
        if not path.exists():
            print(f"  [skip] {name}: {path} not found")
            continue
        d = np.load(path)
        feats = d["features"]
        # Frozen npz may lack labels/pids — recover from any FT run
        if "labels" in d.files and "patient_ids" in d.files:
            labels, pids = d["labels"], d["patient_ids"]
        else:
            # Try to recover from LaBraM FT features
            ft_dir = "results/studies/exp05_stress_feat_multiseed/s42_llrd1.0"
            if os.path.isdir(ft_dir):
                _, labels, pids = va.load_ft_features(ft_dir)
            else:
                print(f"  [skip] {name}: no labels available")
                continue

        r = _compute_plf(feats, labels, pids)
        results[name] = r
        print(f"  {name}: PLF={r['pooled_label_fraction']:.4f} "
              f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}] "
              f"({r['n_subjects']}subj, {r['embed_dim']}d)")
    return results


def analyze_nonfm_trained() -> dict:
    """Compute PLF for trained non-FM features (per seed and mean)."""
    results = {}
    ms_root = STUDY_ROOT / "multiseed"

    if not ms_root.exists():
        print("  [skip] multiseed directory not found")
        return results

    for model in ["eegnet", "shallowconvnet"]:
        model_results = {"per_seed": {}, "mean_plf": None, "mean_ba": None}
        plfs = []
        bas = []

        for seed in SEEDS:
            # Find the run dir (need to match the lr in the name)
            matching = list(ms_root.glob(f"{model}_lr*_s{seed}"))
            if not matching:
                print(f"  [skip] {model} s{seed}: no run dir found")
                continue
            run_dir = matching[0]

            # Load features
            try:
                feats, labels, pids = va.load_ft_features(str(run_dir))
            except FileNotFoundError:
                print(f"  [skip] {model} s{seed}: no fold features")
                continue

            # Load BA
            summary_path = run_dir / "summary.json"
            ba = None
            if summary_path.exists():
                m = json.loads(summary_path.read_text())
                ba = m.get("subject_bal_acc")
                if ba is not None:
                    bas.append(ba)

            r = _compute_plf(feats, labels, pids)
            r["subject_bal_acc"] = ba
            model_results["per_seed"][str(seed)] = r
            plfs.append(r["pooled_label_fraction"])

            print(f"  {model} s{seed}: PLF={r['pooled_label_fraction']:.4f} "
                  f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}] "
                  f"BA={ba:.4f if ba else 'N/A'}")

        if plfs:
            model_results["mean_plf"] = float(np.mean(plfs))
            model_results["std_plf"] = float(np.std(plfs))
        if bas:
            model_results["mean_ba"] = float(np.mean(bas))
            model_results["std_ba"] = float(np.std(bas))

        results[model] = model_results
    return results


def main():
    print("=" * 60)
    print("Non-FM Baseline Variance Analysis (Stress per-rec DASS)")
    print("=" * 60)

    print("\n--- FM Frozen References ---")
    fm_results = analyze_fm_frozen()

    print("\n--- Non-FM Trained Features ---")
    nonfm_results = analyze_nonfm_trained()

    # Summary comparison table
    print(f"\n{'='*60}")
    print("Comparison: Pooled Label Fraction (higher = more label signal)")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'PLF':>8} {'95% CI':>18} {'BA':>8}")
    print("-" * 60)

    for name, r in fm_results.items():
        print(f"{name:<25} {r['pooled_label_fraction']:>8.4f} "
              f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]  {'N/A':>8}")

    for model, mr in nonfm_results.items():
        if mr["mean_plf"] is not None:
            plf = mr["mean_plf"]
            ba_str = f"{mr['mean_ba']:.4f}" if mr["mean_ba"] else "N/A"
            # Use first seed's CI as representative
            first = next(iter(mr["per_seed"].values()), None)
            ci = f"[{first['ci_low']:.4f}, {first['ci_high']:.4f}]" if first else ""
            print(f"{model + ' (trained)':<25} {plf:>8.4f} {ci:>18}  {ba_str:>8}")

    # Save results
    out_path = STUDY_ROOT / "variance_analysis.json"
    output = {
        "fm_frozen_references": fm_results,
        "nonfm_trained": nonfm_results,
        "interpretation": (
            "If non-FM trained PLF is similar to FM frozen PLF (~3-7%), "
            "subject dominance is a property of the EEG signal itself. "
            "If non-FM trained PLF is significantly higher, FMs are "
            "especially subject-dominated due to their pretraining."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
