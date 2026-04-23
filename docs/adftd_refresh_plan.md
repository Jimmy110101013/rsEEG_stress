# ADFTD Refresh Plan — n_splits=3 → 1 (2026-04-23)

> **Single entry point for this refresh.** All artifacts live at their
> canonical paths (so downstream scripts keep working) but are indexed
> from this doc. Do not create parallel `results/adftd_refresh_*/`
> directories — keep everything at canonical locations + update the
> index below.

## Artifact index

| What | Canonical path | Updated? |
|---|---|---|
| New ADFTD frozen features (pooled) | `results/features_cache/frozen_{labram,cbramod,reve}_adftd_19ch.npz` | ✅ 2026-04-23 |
| New ADFTD frozen features (perwindow) | `results/features_cache/frozen_{labram,cbramod,reve}_adftd_perwindow.npz` | ✅ 2026-04-23 |
| Old split3 backup | `results/features_cache/archive_split3_20260423/` | archived |
| Per-window LP | `results/studies/perwindow_lp_all/adftd/{model}_multi_seed.json` | ✅ 2026-04-23 |
| Variance decomposition source | `paper/figures/_historical/source_tables/variance_analysis_all.json` (and `sleepdep_variance_rsa.json`) | ⏳ pending FT v2 features |
| FOOOF probes | `results/studies/fooof_ablation/adftd_probes.json` | ✅ 2026-04-24 — split1 + per-FM window (labram/cbramod=5s, reve=10s); `.bak_split3_20260423` retained |
| FOOOF reconstructed features | `results/features_cache/fooof_ablation/feat_{model}_adftd_{cond}.npz` + `adftd_norm_none_w{5,10}.npz` | ✅ 2026-04-24 |
| Band-stop | `results/studies/exp14_channel_importance/band_stop_ablation.json` (combined) | ✅ 2026-04-24 — ADFTD key updated to split1 + per-FM window; `.bak_pre_adftd_split1_20260423` retained |
| Band-RSA | `results/studies/exp14_channel_importance/band_rsa.json` | ✅ 2026-04-23 — ADFTD row added (split1, per-FM window) |
| New ADFTD dataset caches | `data/cache_adftd_split1/` (zscore), `data/cache_adftd_split1_nnone/` (norm=none) | auto-created |
| Execution logs | `logs/adftd_refresh_20260423/*.log` | live |
| Methodology guardrails | `docs/methodology_notes.md` §G-F11, G-F12 | ✅ 2026-04-23 |

**Rule of thumb**: if you need to find something related to this refresh, start here and follow the path. If you add a new artifact, append a row.

## Quick numeric results (2026-04-23)

### LP (per-window sklearn LogReg, 8 seeds, subject-level StratifiedGroupKFold(5))

| FM | BA 8-seed | Range | vs old split3 |
|---|---|---|---|
| LaBraM | 0.6334 ± 0.0327 | 0.5987–0.7026 | 0.653 → 0.633 (−2 pp, n_rec 195 → 65) |
| CBraMod | 0.5943 ± 0.0255 | 0.5398–0.6231 | 0.570 → 0.594 (+2.4 pp) |
| REVE | 0.6629 ± 0.0210 | 0.6403–0.6992 | 0.652 → 0.663 (+1 pp) |

Direction (REVE > LaBraM > CBraMod) unchanged; magnitudes shifted <3 pp; std wider as expected with 3× fewer records.


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

- [x] Per-window frozen LP (3 FMs × 8 seeds) → `results/studies/perwindow_lp_all/adftd/` (2026-04-23)
- [ ] Variance decomposition → update `paper/figures/variance_analysis.json` (waiting on FT v2 + script rewrite)
- [x] FOOOF ablation → update `results/studies/fooof_ablation/adftd_probes.json` (2026-04-24, per-FM window)
- [x] Band RSA → update ADFTD row in band RSA outputs (2026-04-23)
- [x] Band-stop ablation (reads ADFTD dataset cache, n_splits=1) → update band-stop outputs (2026-04-24, per-FM window)

## Step 3 — Figure / table rebuild

- [ ] Fig 2 (§4.2) variance decomposition — re-render (pending Step 2 variance)
- [x] Fig 5 (§4.5) panels re-rendered 2026-04-24 via canonical builder `notebooks/_build_figures_consolidated.py` → `paper/figures/fig5/fig5{a,b,c}_*.{pdf,png}`. **Protocol decision (2026-04-24)**: fig5b drops ADFTD (3 datasets shown: EEGMAT/SleepDep/Stress). Rationale: session-level subject-ID probe is undefined at n_splits=1 (1 rec/subject); unifying on session-level keeps the probe semantics consistent with the paper's "FM trait-memorization" narrative, rather than down-grading to window-level holdout. ADFTD still contributes via fig5a (PSD) + fig5c (band-stop). Caption must note this. Legacy builders `scripts/figures/build_fooof_ablation_figure.py` + `build_band_stop_all_bands_figure.py` removed.
- [x] Fig B.2 band-stop per-FM breakdown re-rendered 2026-04-24 (`paper/figures/appendix/figB2_band_stop_breakdown.{pdf,png}`); Fig B.3 band-RSA already done 2026-04-23
- [ ] Tab 1 (§4.1) — re-assemble LP + FT columns; note that FT already uses per-FM HP run

## Step 2 numeric deltas (2026-04-24)

### FOOOF state_probe BA (8-seed mean, ADFTD AD-vs-HC subject-level CV)

| FM | Condition | split3 (n=195, w=5 all FMs) | split1 (n=65, per-FM window) | Δ |
|---|---|---|---|---|
| LaBraM | original | 0.6539 ± 0.018 | 0.6334 ± 0.033 | −2.1 pp |
| LaBraM | aperiodic_removed | 0.5418 ± 0.040 | 0.5779 ± 0.060 | +3.6 pp |
| LaBraM | periodic_removed | 0.6553 ± 0.015 | 0.6312 ± 0.033 | −2.4 pp |
| LaBraM | both_removed | 0.5441 ± 0.040 | 0.5705 ± 0.057 | +2.6 pp |
| CBraMod | original | 0.5706 ± 0.035 | 0.5943 ± 0.026 | +2.4 pp |
| CBraMod | aperiodic_removed | 0.6917 ± 0.055 | **0.7236 ± 0.063** | +3.2 pp |
| CBraMod | periodic_removed | 0.5725 ± 0.034 | 0.5978 ± 0.022 | +2.5 pp |
| CBraMod | both_removed | 0.6921 ± 0.053 | **0.7219 ± 0.062** | +3.0 pp |
| REVE (w=10) | original | 0.6520 ± 0.027 | 0.6629 ± 0.021 | +1.1 pp |
| REVE (w=10) | aperiodic_removed | 0.6136 ± 0.055 | 0.6567 ± 0.044 | +4.3 pp |
| REVE (w=10) | periodic_removed | 0.6527 ± 0.025 | 0.6646 ± 0.025 | +1.2 pp |
| REVE (w=10) | both_removed | 0.6142 ± 0.055 | 0.6584 ± 0.046 | +4.4 pp |

**Qualitative directions preserved** across all 12 (FM × condition) cells; magnitudes shift ≤4.4 pp. The striking pattern is model-divergent:
- **LaBraM**: aperiodic-removed drops BA (0.63→0.58, periodic-preserving → keeps BA). Consistent with 1/f-aperiodic anchoring (Fig 4.5 taxonomy).
- **CBraMod**: aperiodic-removed *raises* BA (0.59→0.72). Aperiodic component acts as a **nuisance** for CBraMod — removing it unmasks the oscillatory signal. Substantive for §4.5 narrative.
- **REVE** (w=10s): aperiodic-removed drops BA (0.66→0.66, marginal). Mixed anchoring.

### Band-stop cosine distance (ADFTD)

labram/cbramod numbers **unchanged** (they were already at split1/w=5 via `cache_adftd_nnone_w5`; only the cache path was legacy-named). REVE shifted because window changed 5→10:

| FM | band | split1/w=5 (old) | split1/w=10 (new, REVE only) |
|---|---|---|---|
| REVE | delta | 0.0327 ± 0.082 | **0.0302 ± 0.071** |
| REVE | theta | 0.0125 ± 0.030 | 0.0129 ± 0.035 |
| REVE | alpha | 0.0051 ± 0.012 | 0.0038 ± 0.009 |
| REVE | beta | 0.0011 ± 0.002 | 0.0013 ± 0.003 |

REVE's ADFTD band-ranking (delta ≫ theta > alpha > beta) is preserved. Magnitudes marginally softened at w=10.

## Reference commits

- Cache change + extract_frozen_all.py patch: (this commit)
- Prior per-FM unified FT runs (2026-04-22/23): obs 3348, 3361, 3385

## Supersedes

- `docs/findings.md` F-C / F-D ADFTD rows under n_splits=3 — pending refresh
- `results/studies/perwindow_lp_all/SUMMARY.md` ADFTD row (0.653 labram etc., n_rec=195) — will be re-run under n_splits=1
