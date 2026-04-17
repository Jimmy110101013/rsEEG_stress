"""WSCI analysis from pre-computed holospectra.

Loads per-recording holospectrum files, builds per-subject condition
arrays, runs WSCI cluster permutation, and outputs comparison figures.

EEGMAT: Subject{NN}_1 = rest (cond0), Subject{NN}_2 = task (cond1)
Stress: median-split on DSS per subject (above = cond1, below = cond0)

Usage:
    python scripts/run_wsci_from_holospectra.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.wsci import wsci_subject, wsci_dataset

HOLO_DIR = "results/hhsa/holospectra"
OUT_DIR = "results/hhsa/wsci"


def load_eegmat_subjects():
    """Return dict {subject_id: (H_rest, H_task)} with channel-aggregated holospectra."""
    holo_dir = os.path.join(HOLO_DIR, "eegmat")
    subjects = {}
    for f in sorted(os.listdir(holo_dir)):
        if not f.endswith(".npz"):
            continue
        stem = f[:-4]  # Subject01_1
        parts = stem.rsplit("_", 1)
        subj_id, cond_idx = parts[0], int(parts[1])

        d = np.load(os.path.join(holo_dir, f))
        H_agg = d["H_chan_agg"]  # (n_win, n_fc, n_fa)

        if subj_id not in subjects:
            subjects[subj_id] = [None, None]
        subjects[subj_id][cond_idx - 1] = H_agg  # idx 0=rest, 1=task

    # Filter to subjects with both conditions
    valid = {}
    for sid, (h0, h1) in subjects.items():
        if h0 is not None and h1 is not None:
            valid[sid] = (h0, h1)
    return valid


def load_stress_subjects():
    """Return dict {subject_id: (H_low_stress, H_high_stress)} via DSS median split."""
    import pandas as pd
    df = pd.read_csv("data/comprehensive_labels.csv")

    holo_dir = os.path.join(HOLO_DIR, "stress")

    # Group recordings by subject
    subj_recs = {}
    for _, row in df.iterrows():
        pid = int(row["Patient_ID"])
        rid = int(row["Recording_ID"])
        group = row["Group"]
        dss = float(row["Stress_Score"])
        rec_id = f"{group}_p{pid:02d}_rec{rid}"
        holo_path = os.path.join(holo_dir, f"{rec_id}.npz")

        if not os.path.exists(holo_path):
            continue

        if pid not in subj_recs:
            subj_recs[pid] = []
        subj_recs[pid].append({"dss": dss, "path": holo_path})

    # Median split per subject
    subjects = {}
    for pid, recs in subj_recs.items():
        if len(recs) < 4:  # need >=2 per condition
            continue
        median_dss = np.median([r["dss"] for r in recs])
        lo_paths = [r["path"] for r in recs if r["dss"] <= median_dss]
        hi_paths = [r["path"] for r in recs if r["dss"] > median_dss]
        if len(lo_paths) < 2 or len(hi_paths) < 2:
            continue

        H_lo = np.concatenate([np.load(p)["H_chan_agg"] for p in lo_paths], axis=0)
        H_hi = np.concatenate([np.load(p)["H_chan_agg"] for p in hi_paths], axis=0)
        subjects[f"p{pid:02d}"] = (H_lo, H_hi)

    return subjects


def run_wsci_for_dataset(subjects, dataset_name):
    """Run per-subject WSCI + dataset summary. Returns DatasetWSCI."""
    print(f"\n{'='*60}")
    print(f"WSCI: {dataset_name} ({len(subjects)} subjects)")
    print(f"{'='*60}")

    wsci_values = []
    details = []
    for sid in sorted(subjects.keys()):
        H0, H1 = subjects[sid]
        t0 = time.time()
        result = wsci_subject(H0, H1, seed=42)
        dt = time.time() - t0
        wsci_values.append(result.wsci)
        details.append(result)
        print(f"  {sid}: WSCI={result.wsci:+.4f}  "
              f"({result.n_epochs_cond0}/{result.n_epochs_cond1} epochs, "
              f"{result.n_surviving_clusters} clusters, {dt:.1f}s)")

    ds_result = wsci_dataset(np.array(wsci_values), seed=42)
    print(f"\n  Dataset median: {ds_result.median:+.4f} "
          f"[{ds_result.ci_lower:+.4f}, {ds_result.ci_upper:+.4f}]")
    return ds_result, details


def save_results(ds_result, details, subjects, dataset_name):
    """Save WSCI results to disk."""
    out_path = os.path.join(OUT_DIR, f"{dataset_name}_wsci.npz")
    np.savez_compressed(
        out_path,
        wsci_per_subject=ds_result.per_subject,
        median=ds_result.median,
        ci_lower=ds_result.ci_lower,
        ci_upper=ds_result.ci_upper,
        subject_ids=np.array(sorted(subjects.keys())),
        g_maps=np.stack([d.g_map for d in details]),
        surviving_masks=np.stack([d.surviving_cluster_mask for d in details]),
    )
    print(f"  Saved: {out_path}")


def plot_comparison(eegmat_ds, stress_ds):
    """Bar plot comparing EEGMAT vs Stress WSCI distributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: per-subject WSCI distributions
    ax = axes[0]
    for i, (ds, name, color) in enumerate([
        (eegmat_ds, "EEGMAT", "#2196F3"),
        (stress_ds, "Stress-DSS", "#FF5722"),
    ]):
        vals = ds.per_subject
        ax.boxplot(vals, positions=[i], widths=0.5,
                   boxprops=dict(color=color), medianprops=dict(color=color),
                   whiskerprops=dict(color=color), capprops=dict(color=color),
                   flierprops=dict(markeredgecolor=color))
        ax.scatter(np.full(len(vals), i) + np.random.default_rng(42).uniform(-0.1, 0.1, len(vals)),
                   vals, alpha=0.5, color=color, s=20, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["EEGMAT\n(rest/task)", "Stress\n(DSS split)"])
    ax.set_ylabel("WSCI")
    ax.set_title("Per-Subject WSCI")
    ax.axhline(0, color="gray", ls="--", lw=0.5)

    # Panel B: dataset-level summary with bootstrap CI
    ax = axes[1]
    for i, (ds, name, color) in enumerate([
        (eegmat_ds, "EEGMAT", "#2196F3"),
        (stress_ds, "Stress-DSS", "#FF5722"),
    ]):
        ax.bar(i, ds.median, color=color, alpha=0.7, width=0.5)
        ax.errorbar(i, ds.median,
                    yerr=[[ds.median - ds.ci_lower], [ds.ci_upper - ds.median]],
                    color="black", capsize=5, capthick=1.5, lw=1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["EEGMAT", "Stress-DSS"])
    ax.set_ylabel("WSCI (median)")
    ax.set_title("Dataset-Level WSCI + 95% CI")
    ax.axhline(0, color="gray", ls="--", lw=0.5)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "wsci_comparison.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure: {fig_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    # Check holospectra exist
    for ds in ["eegmat", "stress"]:
        d = os.path.join(HOLO_DIR, ds)
        n = len([f for f in os.listdir(d) if f.endswith(".npz")]) if os.path.isdir(d) else 0
        print(f"  {ds}: {n} holospectrum files")
    print()

    # EEGMAT
    eegmat_subj = load_eegmat_subjects()
    eegmat_ds, eegmat_details = run_wsci_for_dataset(eegmat_subj, "eegmat")
    save_results(eegmat_ds, eegmat_details, eegmat_subj, "eegmat")

    # Stress
    stress_subj = load_stress_subjects()
    stress_ds, stress_details = run_wsci_for_dataset(stress_subj, "stress_dss")
    save_results(stress_ds, stress_details, stress_subj, "stress_dss")

    # Comparison figure
    plot_comparison(eegmat_ds, stress_ds)

    # Summary
    print(f"\n{'='*60}")
    print(f"WSCI Comparison Summary")
    print(f"{'='*60}")
    print(f"  EEGMAT (rest/task):  median={eegmat_ds.median:+.4f} "
          f"[{eegmat_ds.ci_lower:+.4f}, {eegmat_ds.ci_upper:+.4f}] (n={eegmat_ds.n_subjects})")
    print(f"  Stress (DSS split):  median={stress_ds.median:+.4f} "
          f"[{stress_ds.ci_lower:+.4f}, {stress_ds.ci_upper:+.4f}] (n={stress_ds.n_subjects})")
    overlap = not (eegmat_ds.ci_lower > stress_ds.ci_upper or
                   stress_ds.ci_lower > eegmat_ds.ci_upper)
    print(f"  CIs overlap: {overlap}")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
