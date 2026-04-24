# Subject-dass cleanup — 2026-04-09

All 143 runs moved here used `--label subject-dass`, which applies an
OR-aggregation across each subject's recordings (any recording labelled
"increase" → all of that subject's recordings treated as "increase").

On 2026-04-09 we discovered that this aggregation produces a **trait
memorization artifact** rather than a fair state-classification task. Under
the honest per-recording DASS label (`--label dass`, matching Lin et al.
2025's protocol), a frozen LaBraM linear probe achieves BA 0.616 while FT
only reaches 0.536 — so fine-tuning under subject-dass was inflating
results by learning subject identity, not stress state.

See `results/lin_compare/snapshot_2026-04-09_analysis.json` for the full
Frozen-LP vs FT matrix that motivated this cleanup.

## What's here

| Path | Count | Notes |
|---|---|---|
| `hp_sweep/20260409_main/` | 126 runs | 7 models × 6 configs × 3 seeds. REVE re-run after UI crash. All `--label subject-dass`. |
| `ft_subject/labram/` | 6 runs | Historical LaBraM FT on Stress. **Includes canonical 0.6559** (see below). |
| `ft_subject/cbramod/` | 1 run | Historical CBraMod 0.488 baseline. |
| `ft_subject/eegnet/` | 2 runs | Historical EEGNet baselines. |
| `ft_subject/reve/` | 1 run | Historical REVE 0.553 baseline. |
| `ft_trial/labram/` | 1 run | Historical trial-level LaBraM 0.862 (paper comparison to Lin). |
| `ft_trial/cbramod/` | 1 run | Trial-level CBraMod 0.712. |
| `ft_trial/reve/` | 1 run | Trial-level REVE 0.770. |
| `standalone/` | 3 runs | Feb-09 morning DL baseline runs (deepconvnet, shallowconvnet, eegconformer). |
| `feat_extract/20260406_0419_ft_subjectdass_aug75_labram_feat/` | 1 run | **Stress FT feat extraction** used by `paper/figures/variance_analysis.json`. |

## Landmarks to remember

### Canonical LaBraM 0.6559 run
`ft_subject/labram/20260405_1426_ft_subjectdass_aug75_labram/`

- BA = 0.6559 subject-level, seed=42
- HP: lr=1e-5, encoder_lr_scale=0.1, epochs=50, patience=15,
  warmup_freeze_epochs=1, llrd=1.0
- Referenced as the historical "ceiling" number in
  `docs/progress.md`, `docs/related_work.md`, `paper/figures/`, and
  multiple memory files. Keep the path documented even though it is
  archived — we cannot delete it.
- **Important**: this number uses `subject-dass`. Under the per-recording
  `dass` label the same model produces 0.5357 (llrd=1.0) / 0.5536
  (llrd=0.65). See `results/lin_compare/` for the corrected runs.

### Trial-level LaBraM 0.862 run
`ft_trial/labram/20260404_1714_trial_ft_subjectdass_aug75_labram/`

- BA = 0.862 trial-level on 82-rec
- Used as the "same protocol as Lin" comparison number
- **Important**: this also uses `subject-dass`, so it is NOT actually the
  same label regime as Lin et al. 2025 (who use per-recording DASS).
  Archived here with the caveat that the direct Lin comparison should
  use per-recording labels instead. See `docs/drafts/trial_vs_subject_paragraph.md`.

### Stress FT feature extraction
`feat_extract/20260406_0419_ft_subjectdass_aug75_labram_feat/`

- Source of the Stress row in `paper/figures/variance_analysis.json`
- `scripts/run_variance_analysis.py` line 45 points to the old path
  (`results/feat_extract/20260406_0419_...`) — this path is now broken
- `variance_analysis.json`'s Stress pooled label fraction (7.23% → 7.24%)
  was computed with subject-dass labels and needs to be recomputed under
  per-recording dass for the revised paper narrative

## Things NOT archived (kept alive)

`results/lin_compare/` — today's per-recording dass runs (Stress FT
seed=42 with llrd=1.0 and llrd=0.65) + snapshot_2026-04-09_analysis.json

`results/feat_extract/20260406_0935_ft_dass_aug75_labram_adftd_feat/`
`results/feat_extract/20260407_1533_ft_dass_aug75_labram_tdbrain_feat/`
`results/feat_extract/20260408_1536_ft_dass_aug75_labram_eegmat_feat/`
— these use their datasets' native per-recording labels, not affected
by the aggregation issue

`results/cross_dataset/features_*_19ch.npz` — frozen LaBraM features, no
labels baked in (Stress cache is in CSV row order so labels can be
loaded independently)

`results/classical/` — classical ML baselines, no label mode issue

## Broken paths that need updating after this archive

1. `scripts/run_variance_analysis.py:45` — Stress `ft_dir` path
2. Any memory file or doc that references absolute paths under the moved
   directories (check `MEMORY.md`, `docs/progress.md`, `docs/related_work.md`)
