"""Re-run FOOOF ablation probes on ADFTD using original 195-rec meta.

The ablation cache stores only 65 recordings (1/subject), making
subject_probe degenerate (< 2 recs/subject → NaN). The original
per-window cache stores the same windows but with a 195-rec meta
(3 recordings per subject — epoch-group split). Since window ordering
is identical across the two caches, we simply splice the original's
meta onto the ablated features and re-run the probes.

Overwrites results/studies/fooof_ablation/adftd_probes.json with
subject-probe values filled in for all 4 conditions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.experiments.fooof_ablation_probes import (  # noqa: E402
    SEEDS, state_probe, subject_probe,
)


def main():
    out_path = REPO / "results/studies/fooof_ablation/adftd_probes.json"
    conditions = ["original", "aperiodic_removed", "periodic_removed", "both_removed"]
    models = ["labram", "cbramod", "reve"]

    results = {"dataset": "adftd", "results": {}}

    for model in models:
        # Load original cache — source of 195-rec meta
        orig = np.load(REPO / f"results/features_cache/frozen_{model}_adftd_perwindow.npz")
        win_rec = orig["window_rec_idx"]
        rec_lbl = orig["rec_labels"]
        rec_pid = orig["rec_pids"]
        X_orig = orig["features"]

        results["results"][model] = {}
        for cond in conditions:
            print(f"[{model} | {cond}]")
            if cond == "original":
                X = X_orig
            else:
                ab = np.load(
                    REPO / f"results/features_cache/fooof_ablation/feat_{model}_adftd_{cond}.npz"
                )
                X = ab["features"]
                if X.shape[0] != X_orig.shape[0]:
                    print(f"  SKIP: window count mismatch {X.shape[0]} vs {X_orig.shape[0]}")
                    continue

            subj_bas, state_bas = [], []
            for s in SEEDS:
                subj_bas.append(subject_probe(X, win_rec, rec_lbl, rec_pid, s))
                state_bas.append(state_probe(X, win_rec, rec_lbl, rec_pid, s))
            subj = np.array(subj_bas)
            st = np.array(state_bas)
            entry = {
                "subject_probe_mean": float(subj.mean()),
                "subject_probe_std":  float(subj.std(ddof=1)),
                "subject_probe_bas":  subj_bas,
                "state_probe_mean":   float(st.mean()),
                "state_probe_std":    float(st.std(ddof=1)),
                "state_probe_bas":    state_bas,
            }
            print(f"  subject_probe: {entry['subject_probe_mean']:.4f} "
                  f"± {entry['subject_probe_std']:.4f}")
            print(f"  state_probe:   {entry['state_probe_mean']:.4f} "
                  f"± {entry['state_probe_std']:.4f}")
            results["results"][model][cond] = entry

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
