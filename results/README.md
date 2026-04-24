# results/ — layout guide

Everything the training and analysis scripts write goes here. The top
level has exactly three children:

```
results/
├── archive/         # Read-only historical runs. Do not run or cite from here.
├── features_cache/  # Cached feature extractions shared by downstream analysis.
└── studies/         # Self-contained investigations (the live science).
```

## `archive/`

Dated snapshots of runs that are no longer part of the live story but are
kept for traceability. Every subdirectory is `YYYY-MM-DD_<slug>/` and has
its own README explaining why it was archived.

| Path | What's inside |
|---|---|
| `2026-04-02_early_dev/` | First-week prototype runs (REVE LP/LoRA, trial-level sweeps) |
| `2026-04-03_hp_search/` | Early HP search pre-canonical recipe |
| `2026-04-03_killed_incomplete/` | Runs that crashed / were killed mid-training |
| `2026-04-05_classical_baselines_subjectdass/` | Classical-ML baselines; STALE because they used `--label subject-dass` (see 2026-04-09 cleanup) |
| `2026-04-05_fm_diagnosis/` | Early FM comparison PNGs + feature dumps, now superseded by `studies/` investigations |
| `2026-04-09_subject_dass_cleanup/` | 143 runs moved here when subject-dass was reclassified as trait memorization artifact. Contains canonical LaBraM 0.6559 landmark + trial-level 0.862 landmark. See the directory's own README for landmarks. |

**Rule**: never write into `archive/`. If a script needs data from
there, copy it out into `features_cache/` or a new `studies/*/` dir.

## `features_cache/`

Extracted features that multiple downstream scripts depend on. Split
into frozen-encoder features (pooled, fixed weights) and fine-tuned
encoder features (from FT runs with `--save-features`).

```
features_cache/
├── frozen_labram_stress_30ch.npz
├── frozen_labram_adftd_19ch.npz
├── frozen_cbramod_stress_30ch.npz
├── frozen_reve_adftd_19ch.npz
├── …                              # 15 frozen .npz total
├── ft_labram_stress/              # fold*_features.npz from FT runs
├── ft_labram_adftd/
├── ft_cbramod_stress/
└── …                              # 12 ft_* dirs (3 models × 4 datasets)
```

Naming convention: `frozen_<model>_<dataset>_<channels>.npz` for frozen
caches, `ft_<model>_<dataset>/` for FT feature directories.

Producers:
- `scripts/extract_frozen_eegmat.py` → `frozen_labram_eegmat_19ch.npz`
- `scripts/extract_frozen_tdbrain.py` → `frozen_labram_tdbrain_19ch.npz`
- (ADFTD / Stress frozen caches were produced by earlier notebooks; see
  `archive/2026-04-05_fm_diagnosis/` for their provenance.)
- `train_ft.py --save-features` → populates `ft_<model>_<dataset>/`

Consumers:
- `train_lp.py` (canonical per-window LP entry point)
- `scripts/analysis/run_variance_analysis.py`
- `scripts/figures/build_*.py` (figure builders — paths vary by figure)

## `studies/`

Self-contained research investigations. Each `exp##_<slug>/` is
one question, one evidence package, one README, one `analysis.json` that
aggregates all the numbers. Individual runs live in subdirectories.

### Template

```
studies/exp##_<slug>/
├── README.md                   # question + headline + how to re-run
├── analysis.json               # single source of truth for numbers
├── <condition_a>/              # sub-runs grouped by condition
│   └── <run_name>/summary.json
├── <condition_b>/
└── logs/                       # driver + launcher logs
```

### Current studies

| ID | Path | Category | Key artefact |
|---|---|---|---|
| 01 | `exp01_cross_dataset_signal/` | Signal characterization | `signal_strength_results.json` |
| 02 | `exp02_classical_dass/` | Baselines | `summary.json`, `rerun_70rec/` |
| 03 | `exp03_stress_erosion/` | Erosion analysis | `analysis.json` + 4 FT + 10 null + 8-seed frozen LP |
| 04 | `exp04_eegmat_feat_multiseed/` | FT features | 3-seed LaBraM on EEGMAT |
| 05 | `exp05_stress_feat_multiseed/` | FT features | 3-seed LaBraM on Stress |
| 06 | `exp06_fm_task_fitness/` | Representation fitness | `fitness_metrics_full.json` |
| 07 | `exp07_adftd_multiseed/` | Multi-model multi-seed | 3 models × 2 seeds on ADFTD |
| 08 | `exp08_tdbrain_multiseed/` | Multi-model multi-seed | 3 models × 2 seeds on TDBrain |
| 09 | `exp09_multimodel_matched/` | Matched subsampling | `matched_subsample_multimodel.json` |
| 10 | `exp10_stat_hardening/` | Statistical hardening | `stat_hardening.json` |
| 11 | `exp11_longitudinal_dss/` | Longitudinal (negative) | 3 models, centroid/1-NN/linear |

## Habits

1. **Every study gets its own exp## directory** — do not stuff new runs
   into existing study dirs from previous investigations, even if the
   question feels similar. Start a fresh `exp<next>_<slug>/`.
2. **Every study has a README + analysis.json** — README explains what
   and why, analysis.json holds aggregated numbers. Individual run
   `summary.json` files are the source of truth, `analysis.json` is the
   derived aggregation.
3. **If a number comes from code, the code goes into `scripts/`** — no
   inline shell aggregations whose provenance dies with the terminal.
4. **When a study is superseded, archive it** — move into
   `archive/` with a brief note in the archived README explaining why.
5. **Features go into `features_cache/`, not into study dirs** — studies
   reference cached features by path, they don't own them.
