# `results/final/` — paper-surface snapshot layer

**Purpose**: every number cited in the paper is traceable to a JSON under this tree. Small files, heavy provenance. The raw per-fold / per-seed runs live in `results/studies/exp##_*/` (raw layer) — this directory is a **curated snapshot** built from them.

**Entry point**: `docs/paper_experiments_index.md` lists which paper section each snapshot feeds.

**Rebuild command**: `python scripts/analysis/rebuild_final_snapshots.py --all` regenerates this tree from the raw layer. Re-run any time the raw layer changes.

## Layout

```
results/final/
├── README.md                         # this file
├── {adftd,eegmat,sleepdep,stress,tdbrain}/
│   ├── ft/{labram,cbramod,reve}/seed{42,123,2024}/
│   │   ├── summary.json              # subject_bal_acc, per-subject breakdown
│   │   └── provenance.json           # raw_dir, commit, HP recipe
│   ├── lp/{labram,cbramod,reve}.json # 8-seed per-window LogReg + provenance
│   ├── classical/summary.json        # LogReg/SVM/RF/XGB × 3 seeds + provenance
│   ├── nonfm_deep/
│   │   ├── eegnet.json               # 3-seed aggregate + provenance
│   │   └── shallowconvnet.json
│   ├── perm_null/{model}_null.json   # 30-seed null aggregate + real-FT p-value
│   ├── fooof_ablation/probes.json    # FOOOF {aperiodic,periodic,both}_removed probe BAs
│   └── band_stop/probes.json         # per-band probe BA + cosine distance
└── cross_cell/
    ├── band_rsa.json                 # 3 FM × 4 cells × 4 bands RSA
    ├── variance_decomp.json          # 4-cell variance partition (Subject/Label/Residual)
    └── tab1_benchmark.json           # assembled Tab 1 source
```

## Provenance convention

Every JSON has a top-level `provenance` field:

```json
{
  "provenance": {
    "raw_dir": "results/studies/exp07_adftd_multiseed",
    "snapshot_date": "2026-04-23",
    "commit": "<short-sha>",
    "script": "scripts/analysis/rebuild_final_snapshots.py",
    "notes": "…"
  },
  ...  // actual data
}
```

## Rules

1. **Do not hand-edit snapshot JSONs.** Edit the raw run or the rebuild script, then re-run.
2. **Do not commit inconsistent snapshots.** If one cell × model × diagnostic updates, re-run the rebuild for that slice and commit together.
3. **Do not duplicate raw data here.** `summary.json` ≤ a few KB, `provenance.json` points back to the raw dir for per-fold details.
4. **Consumers** (`scripts/figures/build_*`): read from `results/final/`, not from `results/studies/exp##_*/` (except as a migration TODO).
