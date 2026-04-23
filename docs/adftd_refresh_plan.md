# ADFTD Refresh Plan — n_splits=3 → 1 (2026-04-23)

## Why

ADFTD was cached under `n_splits=3` (each subject → 3 pseudo-recordings, n_rec=195) while the canonical binary protocol for the 4-dataset 2×2 factorial uses `n_splits=1` (one record per subject, n_rec ≈ 65–82). All ADFTD frozen-feature analyses need regeneration on the split1 cache.

FT results on ADFTD from yesterday's per-FM HP unified run are already on split1 — those feed §4.1 Tab 1 FT column and §4.3 Fig 3 without further work.

## Scope (only ADFTD; other datasets unaffected)

| Paper location | Analysis | Feature needed | Re-run script |
|---|---|---|---|
| §4.1 Tab 1 | Per-window frozen LP | perwindow | `scripts/experiments/frozen_lp_perwindow_all.py --dataset adftd` |
| §4.2 Fig 2 | Variance decomposition | pooled | `scripts/analysis/run_variance_analysis.py` |
| §4.5 Fig 5 | FOOOF ablation | perwindow | `scripts/experiments/fooof_ablation_probes.py ...` |
| App B.3 Fig B.3 | Band RSA | pooled | `scripts/analysis/band_rsa_analysis.py` |
| §4.5 Fig 5 / App B.2 | Band-stop | (no frozen cache — reads dataset) | `scripts/analysis/band_stop_ablation.py` |

## Per-FM config

- labram: window=5s, norm=zscore
- cbramod: window=5s, norm=none
- reve: window=10s, norm=none
- all: binary (AD vs HC), n_splits=1

## Artifact locations

- Features (new canonical): `results/features_cache/frozen_{model}_adftd_{19ch,perwindow}.npz`
- Old split3 backup: `results/features_cache/archive_split3_20260423/` (see its README)
- Cache dirs (new, auto-created by ADFTDDataset on first load):
  - `data/cache_adftd_split1` (labram, zscore)
  - `data/cache_adftd_split1_nnone` (cbramod/reve, norm=none)
- Launch logs: `logs/adftd_refresh_20260423/{model}_{pooled,perwindow}.log`

## Step 1 — Frozen feature extraction (DONE 2026-04-23 ~11:30 GMT+8)

Confirmed: split1 loads **65 subjects** (AD=36, HC=29). Total windows / FM:

| FM | Pooled shape | Perwindow shape | Window | Windows/rec |
|---|---|---|---|---|
| LaBraM | (65, 200) | (10735, 200) | 5s | ~165 |
| CBraMod | (65, 200) | (10735, 200) | 5s | ~165 |
| REVE | (65, 512) | (5349, 512) | 10s | ~82 |

- [x] labram × ADFTD pooled (GPU 3)
- [x] labram × ADFTD perwindow (GPU 3)
- [x] cbramod × ADFTD pooled (GPU 4)
- [x] cbramod × ADFTD perwindow (GPU 4)
- [x] reve × ADFTD pooled (GPU 5)
- [x] reve × ADFTD perwindow (GPU 5)

Each FM runs pooled→perwindow sequentially on one GPU (chained with `&&`). Three FMs parallel across GPUs 3/4/5.

## Step 2 — Downstream analyses (after Step 1)

- [ ] Per-window frozen LP (3 FMs × 8 seeds) → `results/studies/perwindow_lp_all/adftd/`
- [ ] Variance decomposition → update `paper/figures/variance_analysis.json`
- [ ] FOOOF ablation → update `results/studies/fooof_ablation/adftd_probes.json`
- [ ] Band RSA → update ADFTD row in band RSA outputs
- [ ] Band-stop ablation (reads ADFTD dataset cache, n_splits=1) → update band-stop outputs

## Step 3 — Figure / table rebuild

- [ ] Fig 2 (§4.2) variance decomposition — re-render
- [ ] Fig 5 (§4.5) FOOOF panel — re-render
- [ ] Fig B.2 / B.3 — re-render
- [ ] Tab 1 (§4.1) — re-assemble LP + FT columns; note that FT already uses per-FM HP run

## Reference commits

- Cache change + extract_frozen_all.py patch: (this commit)
- Prior per-FM unified FT runs (2026-04-22/23): obs 3348, 3361, 3385

## Supersedes

- `docs/findings.md` F-C / F-D ADFTD rows under n_splits=3 — pending refresh
- `results/studies/perwindow_lp_all/SUMMARY.md` ADFTD row (0.653 labram etc., n_rec=195) — will be re-run under n_splits=1
