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
├── frozen_labram_stress_19ch.npz
├── frozen_labram_adftd_19ch.npz
├── frozen_labram_tdbrain_19ch.npz
├── frozen_labram_eegmat_19ch.npz
└── ft_labram/
    ├── adftd_2026-04-06/        # fold*_features.npz + summary.json
    ├── tdbrain_2026-04-07/
    └── eegmat_2026-04-08/
```

Naming convention: `<mode>_<model>_<dataset>_<channels>.npz` for bare
caches, `<dataset>_<date>/` for FT feature runs (which carry their own
summary.json).

Producers:
- `scripts/extract_frozen_eegmat.py` → `frozen_labram_eegmat_19ch.npz`
- `scripts/extract_frozen_tdbrain.py` → `frozen_labram_tdbrain_19ch.npz`
- (ADFTD / Stress frozen caches were produced by earlier notebooks; see
  `archive/2026-04-05_fm_diagnosis/` for their provenance.)
- `train_ft.py --save-features` → populates `ft_labram/<dataset>_<date>/`

Consumers:
- `scripts/stress_frozen_lp_multiseed.py`
- `scripts/analyze_eegmat.py`
- `scripts/run_variance_analysis.py`
- `scripts/build_label_subspace_figure.py`

## `studies/`

Self-contained research investigations. Each `YYYY-MM-DD_<slug>/` is
one question, one evidence package, one README, one `analysis.json` that
aggregates all the numbers. Individual runs live in subdirectories.

### Template

```
studies/YYYY-MM-DD_<slug>/
├── README.md                   # question + headline + how to re-run
├── analysis.json               # single source of truth for numbers
├── <condition_a>/              # sub-runs grouped by condition
│   └── <run_name>/summary.json
├── <condition_b>/
└── logs/                       # driver + launcher logs
```

### Current studies

| Path | Question | Key artefact |
|---|---|---|
| `2026-04-09_cross_dataset_signal/` | How does resting-state stress signal strength compare across ADFTD / TDBRAIN / Stress / EEGMAT spectra + t-SNE? | `signal_strength_results.json`, `signal_strength_spectrum.png`, `tsne_cross_dataset.png` |
| `2026-04-09_subject_dass_snapshot.json` | Snapshot of frozen LP vs FT matrix + hp_sweep leaderboard at the moment we reclassified subject-dass as trait memorization. Historical — superseded by the 2026-04-10 erosion study. | (single JSON) |
| `2026-04-10_stress_erosion/` | Does fine-tuning LaBraM on UCSD Stress per-rec DASS beat a frozen linear probe, or actively erode the representation? | `analysis.json` + 4 real FT runs + 10 null FT runs + 1 drift check + 8-seed frozen LP |

## Habits

1. **Every study gets its own dated directory** — do not stuff new runs
   into existing study dirs from previous investigations, even if the
   question feels similar. Start a fresh `YYYY-MM-DD_<slug>/`.
2. **Every study has a README + analysis.json** — README explains what
   and why, analysis.json holds aggregated numbers. Individual run
   `summary.json` files are the source of truth, `analysis.json` is the
   derived aggregation.
3. **If a number comes from code, the code goes into `scripts/`** — no
   inline shell aggregations whose provenance dies with the terminal.
4. **When a study is superseded, archive it** — move into
   `archive/YYYY-MM-DD_<slug>/` with a brief note in the archived
   README explaining why.
5. **Features go into `features_cache/`, not into study dirs** — studies
   reference cached features by path, they don't own them.
