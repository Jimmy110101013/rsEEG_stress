# Paper Figures — Provenance & Status Ledger

**Created**: 2026-04-14 (consolidated from `results/studies/exp*/` after paper-review)
**Source of truth**: `docs/findings.md` (F01–F20) + `docs/paper_narrative_zh.md`

All PDFs below are **copies** of the originals in `results/studies/exp*/`. The
originals remain in place — edit this folder freely without worrying about
breaking study-level analyses.

Status legend:
- **READY** — can be used as-is in draft
- **REWORK** — exists but needs edits per paper-review findings
- **MISSING** — needs to be created before draft

---

## Main figures (8 planned; 6 have source assets)

| # | File | Source | Status | Review action required |
|---|---|---|---|---|
| 1 | `fig1_pipeline.pdf` | `scripts/build_pipeline_figure.py` | **READY** | 2-panel schematic: (a) data→norm→FM→heads, (b) subject-level vs trial-level CV |
| 2 | `fig2_cv_gap.pdf` | `scripts/build_cv_gap_figure.py` | **READY** (subject-dass) | Trial vs subject CV on same 3 FMs: mean gap 21.6 pp. Uses subject-dass (F01, deprecated per F07); per-rec dass rerun is TODO #4 |
| 3a | `fig3a_fitness_heatmap.pdf` | `exp06/fitness_heatmap.pdf` | **READY** | Claim RSA subject>label 12/12 |
| 3b | `fig3b_tsne_cross_dataset.png` | `exp01/tsne_cross_dataset.png` | **REWORK** | Needs composite with 3a into single Fig 3 (two-panel) |
| 4 | `fig4_classical_vs_fm.pdf` | `exp02/classical_vs_fm.pdf` | **VERIFY** | Confirm this is 70-rec rerun, not old 61-rec; if old, regenerate from `exp02/rerun_70rec/summary.json` |
| 5 | `fig5_cross_dataset_taxonomy.pdf` | `exp13/cross_dataset_model_comparison.pdf` | **REWORK** | Headline figure. Soften visual emphasis — effect sizes are 1–2pp. Add cluster-bootstrap CIs (compute from `f17_matched_n_multimodel.json`). Caption must state "directions differ" not "completely opposite" |
| 6 | `fig6_matched_n_curves.pdf` | `exp09/matched_subsample_curves.pdf` | **READY** | F04 N-invariance validation; LaBraM-only is a known limitation (cite in caption) |
| 7a | `fig7a_topomap.pdf` | `exp14/channel_importance_topomap.pdf` | **READY** | Spatial channel importance |
| 7b | `fig7b_band_rsa.pdf` | `exp14/band_rsa.pdf` | **READY** | Correlational spectral |
| 7c | `fig7c_band_stop.pdf` | `exp14/band_stop_ablation.pdf` | **REWORK** | paper-review C9: current figure only shows alpha-stop drop. Add control band-stop (e.g. gamma or random) to defuse "just removing energy" reviewer attack |
| 8a | `fig8a_stress_erosion.pdf` | `exp03/stress_erosion.pdf` | **REWORK** | Must clearly mark as "surface values — see §7 power floor analysis" in caption + visual treatment |
| 8b | `fig8b_stat_hardening.pdf` | `exp10/stat_hardening.pdf` | **READY** | Bootstrap CIs for §7 |
| 8c | `fig8c_non_fm_baselines.pdf` | `exp15_nonfm_baselines/sweep/` (JSON only) | **MISSING** | Create bar chart: ShallowConvNet/EEGNet vs 3 FMs, all 3-seed. Source data exists, no figure yet |

---

## Supplementary figures

| # | File | Source | Status |
|---|---|---|---|
| S1 | `s1_signal_strength_spectrum.png` | `exp01/signal_strength_spectrum.png` | **READY** |
| S2 | `s2_permutation_null_hist.pdf` | — | **MISSING** — create null histograms for LaBraM/CBraMod/REVE on Stress from `f05_f06_stress_erosion.json` |
| S3 | `s3_matched_n_ba_curve.pdf` | `exp09/matched_n_ba_curve.pdf` | **READY** |
| S4a | `s4a_longitudinal_feature_space.pdf` | `exp11/feature_space_analysis.pdf` | **READY** (F14 negative) |
| S4b | `s4b_within_subject.pdf` | `exp11/within_subject_supplementary.pdf` | **READY** |
| S5 | `s5_reve_window_sensitivity.pdf` | — | **MISSING** — plot from `f18_reve_window.json` + `f18_adftd_window.json` |
| S6 | `s6_eegmat_paired.pdf` | — | **MISSING** — EEGMAT F09 supporting (LaBraM only) |
| S7 | `s7_grl_sweep.pdf` | — | **MISSING** (or keep as text-only table for F12) |

---

## Source tables (JSON → paper tables)

These JSON files carry the numbers for in-text tables. Keep in sync with
`findings.md` revisions.

| File | Feeds | Findings |
|---|---|---|
| `f03_f16_classical_70rec.json` | Table: classical vs FM under per-rec dass | F03, F16 |
| `f05_f06_stress_erosion.json` | Table: Stress BA + permutation null | F05, F06 |
| `f13_fitness_metrics.json` | Text: RSA subject vs label values | F13 |
| `f15_bootstrap_ci.json` | Table: Cohen's d + bootstrap CI | F15 |
| `f17_matched_n_multimodel.json` | Table: pp Δ matrix + CI (needs compute) | F17 |
| `f17_adftd_multiseed.json` | ADFTD per-seed detail | F17 |
| `f14_within_subject.json` | Table: within-subject longitudinal | F14 |
| `f18_reve_window.json` | REVE 5s vs 10s | F18 |
| `f18_adftd_window.json` | ADFTD 5s vs 10s | F18 |
| `exp01_signal_strength.json` | S1 | — |
| `exp14_band_rsa.json` | Fig 7b numbers | — |
| `exp14_band_stop.json` | Fig 7c numbers | — |
| `exp14_channel_importance.json` | Fig 7a numbers | — |
| `variance_analysis_all.json` | §5 variance decomp raw | F04, F17 |

---

## Summary of remaining work (prioritized by paper-review severity)

### Critical (blocks submission; from review Phase 4 trust scorecard)
1. **Fig 5 bootstrap CIs** — compute cluster bootstrap on pp Δ per cell; add error bars. Without this, F17's +1pp claims have no visible uncertainty.
2. **Fig 7c control band** — run band-stop on gamma (or random band) as control; confirms alpha-specific effect is not just energy removal.
3. **Fig 8 composite framing** — unify 8a+8b+8c visually so the "surface effect → null-indistinguishable → architecture-agnostic" story is self-contained.
4. **Fig 2 CV gap honesty** — explicit acknowledgement that Wang 2025 number was not re-run under our codebase (or: actually run Wang's trial-level per-rec dass protocol ourselves and report the exact gap).

### High (figure quality)
5. **Fig 1 pipeline** — conceptual diagram; vectorize in draw.io or similar.
6. **Fig 3 composite** — merge 3a + 3b into single two-panel figure.
7. **Fig 8c non-FM bar** — one afternoon of matplotlib.
8. **S2 null histograms** — overlay real BA markers on null distributions.

### Medium
9. **S5 window sensitivity** — 4-bar plot (REVE 5s frozen / 5s FT / 10s frozen / 10s FT) + 2-bar ADFTD.
10. **S6 EEGMAT paired supporting** — single-panel plot showing paired design yields similar pattern.
