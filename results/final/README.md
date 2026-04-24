# `results/final/` — paper-surface snapshot layer

**Purpose**: every number cited in the paper is traceable to a JSON under this tree. Small files, heavy provenance. The raw per-fold / per-seed runs live in `results/studies/exp##_*/` (raw layer) — this directory is a **curated snapshot** built from them.

**Canonical access**: paper figure/table builders MUST read numbers through `from src import results` (see `src/results.py`), NOT by memorising paths. When a file moves, patch the accessor body; callers inherit the fix for free.

**Index**: `docs/paper_experiments_index.md` lists which paper section each snapshot feeds.

**Rebuild command**: `python scripts/analysis/rebuild_final_snapshots.py --all` regenerates this tree from the raw layer. Re-run any time the raw layer changes.

## Layout

```
results/final/
├── README.md                         # this file
├── source_tables/                    # aggregated JSONs consumed by figure builders
│   ├── variance_analysis_all.json        # pooled variance fractions (Subject/Label/Residual)
│   ├── variance_analysis_window_level.json
│   ├── sleepdep_variance_rsa.json
│   ├── f14_within_subject.json           # within-subject direction consistency (eegmat+stress)
│   ├── sleepdep_within_subject.json
│   ├── master_frozen_ft_table_v2.json    # pre-aggregated Frozen/FT numbers by {fm × dataset}
│   └── ft_rsa_stress_eegmat.json
├── {adftd,eegmat,sleepdep,stress,tdbrain}/
│   ├── ft/{labram,cbramod,reve}/seed{42,123,2024}/
│   │   ├── summary.json              # subject_bal_acc, per-subject breakdown
│   │   └── provenance.json           # raw_dir, commit, HP recipe
│   ├── lp/{labram,cbramod,reve}.json # 8-seed per-window LogReg + provenance
│   ├── classical/summary.json        # LogReg/SVM/RF/XGB × 3 seeds + provenance
│   ├── nonfm_deep/{eegnet,shallowconvnet}.json  # 3-seed aggregate + provenance
│   ├── perm_null/{model}_null.json   # 30-seed null aggregate + real-FT p-value
│   ├── fooof_ablation/probes.json    # FOOOF {aperiodic,periodic,both}_removed probe BAs
│   ├── band_stop/probes.json         # per-band probe BA + cosine distance
│   ├── variance_triangulation/*.json # cross-method variance decomposition
│   └── subject_probe_temporal_block/*.json
└── cross_cell/
    ├── band_rsa.json                 # 3 FM × 4 cells × 4 bands RSA
    ├── variance_decomp.json          # 4-cell variance partition
    └── tab1_benchmark.json           # assembled Tab 1 source
```

## Schema: what each file type must contain

### `ft/<fm>/seed<N>/summary.json`
```json
{ "subject_bal_acc": 0.7086, "subject_f1": ..., "recording_bal_acc": ...,
  "provenance": {...} }
```
Accessor: `results.ft_stats(dataset, fm)` — returns `{mean, std, n_seeds, source}` across 3 seeds.

### `lp/<fm>.json` (8-seed per-window LP)
```json
{ "extractor": "labram", "dataset": "stress", "cv": "stratified-kfold",
  "seeds": [42,123,2024,7,0,1,99,31337],
  "per_seed_ba": {"42": 0.5179, ...},
  "mean_8seed": 0.5119, "std_8seed_ddof1": 0.0103,
  "mean_3seed_42_123_2024": 0.5089, "std_3seed_42_123_2024_ddof1": 0.0135,
  "protocol": "...", "source_features": "..." }
```
Accessors: `results.lp_multiseed(dataset, fm)` (full), `results.lp_stats_3seed(dataset, fm)` (Table 1 format).
Producer: `python train_lp.py --extractor <fm> --dataset <dataset>`.

### `perm_null/<fm>_null.json` (30-seed paired-null FT)
Currently lives at `results/studies/exp27_paired_null/<dataset>/perm_s*/summary.json` (not yet promoted). Each summary has `subject_bal_acc` per seed.
Accessor: `results.perm_null_summaries(dataset)` — list of 30 summary dicts.

### `source_tables/<name>.json`
Pre-aggregated cross-dataset views (variance decomposition, RSA, direction consistency). Schema varies per file — consult the file itself.
Accessor: `results.source_table(name)`.

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
4. **Consumers** (`scripts/figures/build_*`, paper tables, notebooks): access numbers via `src.results.*`, not by reading paths directly. If `src.results` is missing an accessor you need, add one — don't route around the module.

## Adding a new accessor to `src/results.py`

1. Add the function shaped around **what the caller asks** ("give me X for (dataset, fm)"), not around the underlying file layout.
2. Inside the function, hide any current bifurcation (studies/ vs final/, per-cell exp dirs, etc.) so callers don't care.
3. Add a TODO in the docstring noting "when data lands at final/<canonical-path>, flip this body".
4. Smoke-test with an in-script assert that the accessor returns `== json.load(<path>)` for a known path.
