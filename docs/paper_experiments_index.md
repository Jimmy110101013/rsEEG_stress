# Paper Experiments Index — 4 datasets × all diagnostics

**Purpose**: single navigation doc for every paper-relevant experiment. Mirrors the IMRaD layout of `docs/paper_outline.md` so each §/Fig/Table maps to its source data, script, log, and status. If you can't find a result in ≤30 s, the problem is this doc, not the filesystem — fix it here.

**Scope**: main text + appendix only. Exploratory / historical / deprecated work lives in §Archive at the bottom.

**Last master refresh**: 2026-04-24 (ADFTD split1 — FOOOF + band-stop per-FM window complete)

---

## Dataset matrix (2×2 factorial cells)

| Cell | Dataset | n_rec / n_subj | Label | Regime | Alignment |
|---|---|---|---|---|---|
| Within × strong-aligned | EEGMAT | 72 / 36 | rest vs arithmetic | within-subject paired | strong (θ/α) |
| Within × weak-aligned | SleepDep | 72 / 36 | normal vs SD | within-subject paired | weak |
| Subject × strong-aligned | ADFTD | 65 / 65 (split1) | AD vs HC | subject-label trait | strong (1/f aperiodic) |
| Subject × weak-aligned | Stress-DASS | 70 / 17 | DASS binary | subject-label trait | weak |

Supplementary: TDBRAIN (Appendix A — replicates Subject × strong-aligned cell).

---

## §4.1 Benchmark landscape (Tab 1) — setup, no figure

**Table**: `table1_master_performance.tex` (pending build); source = `docs/master_results_table.md` + `paper/figures/_historical/source_tables/master_frozen_ft_table_v2.md`

**Ceiling table (Appendix C / Fig 6)**: `paper/tables/table_fig6_architecture_ceiling.tex` — ADFTD LP row refreshed 2026-04-23 (split1, 65/65).

**Source of truth for numbers**: `docs/master_results_table.md` (human-curated, ADFTD LP refreshed 2026-04-23 split1; FT values still pre-split1 — marked ⚠)

| Row | Content | Source |
|---|---|---|
| LP per cell × FM | per-window sklearn LogReg, 8 seeds | `results/studies/perwindow_lp_all/{cell}/{model}_multi_seed.json` |
| FT per cell × FM | 3 seeds, per-FM unified HP | `results/studies/exp_newdata/` + `results/studies/exp17_eegmat_cbramod_reve_ft/` + `results/studies/exp07_adftd_multiseed/` + `results/studies/exp03_stress_erosion/` |
| Classical ML baseline | LogReg/SVM/RF/XGB, 3 seeds | `results/studies/exp02_classical_dass/{cell}/` |
| Non-FM deep baseline | EEGNet / ShallowConvNet, 3 seeds | `results/studies/exp15_nonfm_baselines/{cell}/` |

**Status 2026-04-24**:
- LP: ✅ all 4 cells fresh (ADFTD split1 2026-04-23; Stress/EEGMAT/SleepDep unchanged since 2026-04-20)
- FT: ✅ all 4 cells × 3 FMs × 3 seeds complete at `results/final/{cell}/ft/{model}/seed{42,123,2024}/summary.json` under per-FM canonical HP (G-F09). Ceiling table.tex + master_results_table.md + paper_outline.md §4.1 entry statements all synced 2026-04-23.
- Classical: ✅ all 4 cells at split1. ADFTD snapshot at `results/final/adftd/classical/summary.json`.
- Non-FM deep: ✅ all 4 cells. **ADFTD refreshed 2026-04-24 on split1** (eegnet 0.761 ± 0.076, shallowconvnet 0.716 ± 0.050; snapshot at `results/final/adftd/nonfm_deep/summary.json`; split3 archive at `exp15_nonfm_baselines/adftd.bak_split3_20260424/`).

---

## §4.2 Variance decomposition — Fig 2 (window-level, unified 2026-04-24)

**Figure**: `paper/figures/fig2/fig2_representation_2x2.{pdf,png}` (canonical; rebuilt 2026-04-24 from `variance_analysis_window_level.json`)
**Build**: `notebooks/_build_figures_consolidated.py` → compiles `notebooks/figures_consolidated.ipynb`; FIG2 cell is executable standalone via `exec(ns['SETUP'] + '\n' + ns['FIG2'])`.
**Analysis script**: `scripts/analysis/run_variance_window_level.py` (uniform for all 4 cells × 3 FMs; frozen + FT side both from per-window features).
**Protocol**: window-level crossed SS decomposition `SS_total = SS_label + SS_subject + SS_residual_clipped` applied uniformly to all 4 cells, including single-session ADFTD. See `docs/variance_window_level_wording.md` for scope-condition wording intended for §3.6.1 and §4.2.

| Cell | Frozen source | FT source (seed 42) | Status |
|---|---|---|---|
| EEGMAT | `results/features_cache/frozen_{fm}_eegmat_perwindow.npz` | `results/final_winfeat/eegmat/{fm}/seed42/fold*_features.npz` | ✅ 2026-04-24 |
| SleepDep | `results/features_cache/frozen_{fm}_sleepdep_perwindow.npz` | `results/final_winfeat/sleepdep/{fm}/seed42/fold*_features.npz` | ✅ 2026-04-24 |
| ADFTD (split1, 65/65) | `results/features_cache/frozen_{fm}_adftd_perwindow.npz` | `results/final_winfeat/adftd/{fm}/seed42/fold*_features.npz` | ✅ 2026-04-24 |
| Stress (55/14 pure-label) | `results/features_cache/frozen_{fm}_stress_perwindow.npz` | `results/final_winfeat/stress/{fm}/seed42/fold*_features.npz` | ✅ 2026-04-24; auto-drops 3 mixed-label subjects (pid 3, 11, 17) |

**Combined output JSON**: `paper/figures/_historical/source_tables/variance_analysis_window_level.json`
**Shape per entry**: `{frozen: {label_frac, subject_frac, residual_frac, label_subject_ratio, SS_*, label_design}, ft: {same fields}, delta_label_frac, delta_subject_frac, ft_seed}`.

**Key numbers** (LaBraM Δlabel_frac, pp): ADFTD +6.25 >> EEGMAT +1.99 >> SleepDep +0.02 ≈ Stress −1.37 — cleanly tracks strong/weak task-substrate alignment.

**Supersedes**:
- `variance_analysis_all.json` — recording-level, split3 ADFTD, pre-canonical FT HP. Retained for historical reference only.
- `sleepdep_variance_rsa.json` — dataset-specific script. Window-level run unifies approach.
- Old `scripts/analysis/run_variance_analysis.py` — labram-only, recording-level, stale paths. Keep for recording-level sensitivity analyses if needed; not used for Fig 2.

**FT side is seed 42 only**. Full 3-seed window-level FT variance is a future extension — not required for Fig 2 main claim since Δlabel_frac ordering is monotonic under canonical HP.

---

## §4.3 Permutation null — Fig 3

**Figure**: `paper/figures/main/fig3_honest_evaluation_4panel.{pdf,png}` (rebuilt 2026-04-23 with renamed axis labels, commit f1acb4c)
**Build script**: `scripts/figures/build_fig3_perm_null_4panel.py`
**Protocol**: real FT BA vs 30-seed label-shuffle null; subject-level perm for ADFTD (obs 3379)

| Cell | Null chain | Real FT reference | Status |
|---|---|---|---|
| EEGMAT | `results/studies/exp27_paired_null/eegmat/` | `results/studies/exp17_eegmat_cbramod_reve_ft/` + `exp04_eegmat_feat_multiseed/` | ✅ 30 seeds complete, unified HP |
| SleepDep | `results/studies/exp27_paired_null/sleepdep/` | `results/studies/exp_newdata/` | ✅ 30 seeds complete (2026-04-22 obs 3381, SleepDep n=30 rebuild) |
| ADFTD | `results/studies/exp27_paired_null/adftd/` (split1, 2026-04-24; split3 archived under `adftd.bak_split3_20260424/`) | `results/final/adftd/ft/labram/seed{42,123,2024}/` | ✅ 30 seeds split1 subject-level perm. **p = 0.0323** (real FT 3-seed mean 0.7367 vs null mean 0.4972, 0/30 null ≥ real). Aggregated to `results/final/adftd/perm_null/labram_null.json` |
| Stress | `results/studies/exp03_stress_erosion/ft_null_{labram,cbramod,reve}/` | `results/studies/exp05_stress_feat_multiseed/` | ✅ 30 seeds |

**Log**: `results/studies/fooof_ablation/adftd_probes.json` (no — this is FOOOF; correct null logs are under each study dir's `logs/`)

---

## §4.4 Within-subject direction consistency — Fig 4

**Figure**: `paper/figures/main/fig4{a,b,c}_*.{pdf,png}` (pending build)
**Build script**: `scripts/figures/build_eegmat_within_subject_figure.py` + `scripts/figures/build_within_subject_supplementary.py`
**Scope**: EEGMAT + SleepDep **only** (Stress / ADFTD excluded by design — no within-subject label contrast)

| Cell | Source JSON | Script | Status |
|---|---|---|---|
| EEGMAT | `paper/figures/_historical/source_tables/f14_within_subject.json` | `scripts/analysis/sleepdep_within_subject.py` (misnamed — applies to both) | ✅ |
| SleepDep | `paper/figures/_historical/source_tables/sleepdep_within_subject.json` | `scripts/analysis/sleepdep_within_subject.py` | ✅ |
| ADFTD | — | — | N/A by design |
| Stress | — | — | N/A by design |

**Input features**: `results/features_cache/ft_{labram,cbramod,reve}_{eegmat,sleepdep}/` (FT extracted per-fold); **pending refresh** on new FT HP.

---

## §4.5 Causal anchor ablation — Fig 5

**Figure**: `paper/figures/fig5/fig5{a,b,c}_*.{pdf,png}` — canonical per-panel outputs; hand-composited master at `paper/figures/main/Fig5_FOOOF_band_stop.png`
**Build script**: `notebooks/_build_figures_consolidated.py` (compiles `notebooks/figures_consolidated.ipynb` → execute FIG5A/B/C cells). Legacy `scripts/figures/build_fooof_ablation_figure.py` + `build_band_stop_all_bands_figure.py` removed 2026-04-24.
**Protocol**: FOOOF fit → {aperiodic_removed, periodic_removed, both_removed} → re-extract FM features → probe BA; **and** band-stop Butterworth → re-extract → probe BA

### FOOOF ablation

| Cell | Probes JSON | Reconstructed features | Script | Status |
|---|---|---|---|---|
| EEGMAT | `results/studies/fooof_ablation/eegmat_probes.json` | `results/features_cache/fooof_ablation/feat_{fm}_eegmat_{cond}.npz` | `scripts/experiments/fooof_ablation_probes.py` | ✅ |
| SleepDep | `results/studies/fooof_ablation/sleepdep_probes.json` | same pattern | | ✅ |
| ADFTD | `results/studies/fooof_ablation/adftd_probes.json` | same pattern | | ✅ 2026-04-24 split1 + per-FM window (labram/cbramod=5s, reve=10s); `.bak_split3_20260423` retained. **Subject-ID probe = NaN** (1 rec/subject → session-level holdout undefined); fig5b drops ADFTD by design, caption must note. State probe values valid and used in the delta table in `adftd_refresh_plan.md`. |
| Stress | `results/studies/fooof_ablation/stress_probes.json` | same pattern | | ✅ |

**Orchestrators** (per-cell shell drivers): `scripts/experiments/_fooof_{adftd,sleepdep}_chain.sh`

**Resolved 2026-04-24**: `fooof_ablation.py` + `extract_fooof_ablated.py` now carry `MODEL_WINDOW = {labram:5, cbramod:5, reve:10}`. ADFTD chain runs FOOOF fit once per window → outputs `adftd_norm_none_w{5,10}.npz`, then extracts dispatch per-FM. Other cells unchanged (single-file naming `{cell}_norm_none.npz`).

### Band-stop ablation (§4.5 panel c + Appendix B.2)

| Cell | JSON | Status |
|---|---|---|
| EEGMAT | `results/studies/exp14_channel_importance/band_stop_ablation.json` (combined) | ✅ |
| SleepDep | `results/studies/exp14_channel_importance/band_stop_ablation.sleepdep_only.json` | ✅ |
| ADFTD | same combined file, adftd key | ✅ 2026-04-24 split1 + per-FM window (labram/cbramod=5s, reve=10s); `.bak_pre_adftd_split1_20260423` retained. labram/cbramod unchanged (already at split1); REVE magnitudes shifted under w=10 |
| Stress | same combined file, stress key | ✅ |

**Script**: `scripts/analysis/band_stop_ablation.py`

**Resolved 2026-04-24**: `MODEL_WINDOW` dict dispatches per-FM window; ADFTD uses `data/cache_adftd_split1{,_nnone}` with `n_splits=1`. `--out-suffix` arg added for parallel-safe writes (merge via per-dataset key).

---

## Appendix A — TDBRAIN variance replication (Fig A.1)

| Item | Path | Status |
|---|---|---|
| Source JSON | `paper/figures/_historical/source_tables/variance_analysis_all.json` (TDBRAIN key) | stale (labram-only) |
| Script | `scripts/analysis/run_variance_analysis.py` | stale |
| Features | `results/features_cache/frozen_{fm}_tdbrain_19ch.npz` | ✅ |

---

## Appendix B — 4-axis FM interpretability

### B.1 Channel ablation (Stress only, topomap)
| Item | Path |
|---|---|
| JSON | `paper/figures/_historical/source_tables/exp14_channel_importance.json` |
| Script | `scripts/analysis/compute_channel_importance.py` or `exp14_channel_importance/` scripts |
| Figure | `paper/figures/main/fig7a_topomap.pdf` (existing) — needs migration to `paper/figures/appendix/` |

### B.2 Band-stop cosine distance (3 FM × 4 cells × 4 bands)
Same raw JSON as §4.5 but different metric. See §4.5 table above + split into "cosine distance" vs "probe BA" metrics.

### B.3 Band-RSA (3 FM × 4 cells × 4 bands)
| Item | Path | Status |
|---|---|---|
| Combined JSON | `results/studies/exp14_channel_importance/band_rsa.json` | ✅ 2026-04-23 — Stress / EEGMAT / ADFTD (split1, per-FM window). SleepDep still missing. Backup `band_rsa.json.bak_pre_adftd_20260423` |
| Script | `scripts/analysis/band_rsa_analysis.py` | patched 2026-04-23 for split1 + ADFTD row |
| Source tables copy | `paper/figures/_historical/source_tables/exp14_band_rsa.json` | stale |

**Band-RSA observations (ADFTD split1, 2026-04-23)**:
- LaBraM: alpha r = −0.002 (p=0.94 ns) — flat; delta r=0.145, beta r=0.134, all_bands r=0.272
- CBraMod: all four bands significant (r 0.088–0.141), all_bands r=0.190
- REVE: all four bands marginal (r 0.051–0.101), all_bands r=0.179

LaBraM × ADFTD shows a **flat alpha RSA** — consistent with 1/f-aperiodic anchoring (Fig 4.5 taxonomy). Worth checking against FOOOF output when refreshed.

---

## Appendix C — Architecture ceiling (Fig 6)

| Item | Path | Status |
|---|---|---|
| Figure | `paper/figures/main/Fig6_ceiling.png` (canonical) | panel titles need terminology update |
| Build | `scripts/figures/build_paper_pdf.py` area | check |
| Source | `docs/master_results_table.md` + `results/studies/exp15_nonfm_baselines/` | ⏳ pending FT HP unification refresh |

---

## `results/final/` snapshot layer (paper-surface)

Curated snapshots; every paper-cited number traces here. Raw runs stay in `results/studies/exp##_*/`. Rebuild any slice with `scripts/analysis/rebuild_final_snapshots.py` (see `docs/results_final_plan.md`).

| Path | Content | Rebuilt by |
|---|---|---|
| `results/final/README.md` | Layer entry point + provenance convention | — |
| `results/final/{cell}/ft/{model}/seed{42,123,2024}/summary.json` | Per-seed FT results (subject_bal_acc etc.) | Written by `train_ft.py`; copied here on run |
| `results/final/{cell}/ft/{model}/seed{N}/provenance.json` | HP recipe + raw dir pointer | `rebuild_final_snapshots.py --section ft` |
| `results/final/{cell}/lp/{model}.json` | 8-seed per-window LP + provenance | `rebuild_final_snapshots.py --section lp` |
| `results/final/{cell}/classical/summary.json` | Classical ML 3-seed aggregate | ⏳ Phase 2 |
| `results/final/{cell}/nonfm_deep/{eegnet,shallowconvnet}.json` | From-scratch deep baselines | ⏳ Phase 2 |
| `results/final/{cell}/perm_null/{model}_null.json` | 30-seed null aggregate + p-value | ⏳ Phase 5 |
| `results/final/{cell}/fooof_ablation/probes.json` | FOOOF ablation probe BAs | ⏳ Phase 4 |
| `results/final/{cell}/band_stop/probes.json` | Per-band probe BA + cosine | ⏳ Phase 4 |
| `results/final/cross_cell/band_rsa.json` | 3 FM × 4 cells × 4 bands RSA | ⏳ Phase 6 |
| `results/final/cross_cell/variance_decomp.json` | 4-cell variance partition | ⏳ Phase 6 |
| `results/final/cross_cell/tab1_benchmark.json` | Assembled Tab 1 source | ⏳ Phase 6 |

Completed phases (2026-04-23): 0 (skeleton), 1 (LP snapshots — 12 files), 3 (FT provenance stamps — 36 files), 8 (SUPERSEDED.md stubs for exp07/exp12 ADFTD split3).

## Features and checkpoint catalog

### Frozen features (per-recording pooled + per-window)

| Pattern | Location |
|---|---|
| Pooled | `results/features_cache/frozen_{fm}_{cell}_{19ch or 30ch}.npz` |
| Per-window | `results/features_cache/frozen_{fm}_{cell}_perwindow.npz` |
| FOOOF-reconstructed | `results/features_cache/fooof_ablation/feat_{fm}_{cell}_{cond}.npz` |

Extraction scripts:
- `scripts/features/extract_frozen_all.py` (pooled; supports `--adftd-n-splits`, `--window-sec`)
- `scripts/features/extract_frozen_all_perwindow.py` (per-window; same flags)
- `scripts/features/extract_fooof_ablated.py` (FOOOF)

### FT checkpoints and features

| Pattern | Location |
|---|---|
| Stress FT features | `results/features_cache/ft_{fm}_stress/` + `_canonical/` (labram only) |
| EEGMAT FT features | `results/features_cache/ft_{fm}_eegmat/` |
| ADFTD FT features | `results/features_cache/ft_{fm}_adftd/` |
| SleepDep FT features | (pending extraction post-HP-unification) |
| TDBRAIN FT features | `results/features_cache/ft_{fm}_tdbrain/` |

### Dataset caches

| Dataset | Cache dir (zscore) | Cache dir (norm=none) |
|---|---|---|
| Stress | `data/cache/` | `data/cache_nnone/` |
| EEGMAT | `data/cache_eegmat/` | `data/cache_eegmat_nnone/` |
| SleepDep | `data/cache_sleepdep/` | `data/cache_sleepdep_nnone/` |
| ADFTD split1 binary | `data/cache_adftd_split1/` | `data/cache_adftd_split1_nnone/` |
| ADFTD split3 (legacy) | `data/cache_adftd_split3/` | `data/cache_adftd_split3_nnone/` |
| ADFTD 3-class split1 | `data/cache_adftd_3cls_split1_nzscore/` | `data/cache_adftd_3cls_split1_nnone/` |
| TDBRAIN | `data/cache_tdbrain/` | `data/cache_tdbrain_nnone/` |

---

## Active refresh tasks

| Task | Plan doc | Status |
|---|---|---|
| ADFTD split1 refresh | `docs/adftd_refresh_plan.md` | 🏃 Step 2 mostly done (FOOOF + band-stop + band-RSA + LP ✅; Variance decomposition still blocked on FT-v2 features + script rewrite) |
| Per-FM FT HP unification (4 cells) | (see memory obs 3348 / 3361 / 3385) | 🏃 runs completed 2026-04-23; feature extraction + table update pending |
| New variance script (4-cell × 3 FM + SleepDep not TDBRAIN) | — | ⏳ not yet written |

---

## Methodology guardrails (not experiment results — live in separate doc)

See `docs/methodology_notes.md` — G-F07 … G-F12 codify pipeline policies (per-FM norm, per-FM HP, dummy-tensor fix, LP deprecation, etc.).

---

## Archive (historical / deprecated)

- `paper/figures/_historical/` — pre-2026-04-18 figures (SDL narrative)
- `results/features_cache/archive_split3_20260423/` — ADFTD split3 frozen features (superseded by split1)
- `results/studies/exp_newdata/exp_newdata/` — HP-contaminated FT runs (pre per-FM unification)
- `train_lp.py` — deprecated pool-then-classify LP (G-F12)
- `results/studies/perwindow_lp_all/SUMMARY.md` 2026-04-20 section — ADFTD row uses split3; see today's refresh for split1
- `docs/historical/` — superseded narrative drafts

---

## How to add to this index

1. When you generate a new source JSON / figure / table, add its row here under the matching §.
2. If you supersede something, move the old row to §Archive with the supersession date.
3. Never create parallel `results/my_task_2026-xx-xx/` dirs. Put artifacts at canonical paths + add a row here.
4. For time-bounded refresh operations (e.g. the ADFTD split1 refresh), use a task-specific plan doc (e.g. `docs/adftd_refresh_plan.md`) as a companion — but the **permanent** answer to "where is X?" lives here.
