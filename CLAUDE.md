# CLAUDE.md — UCSD_stress Architecture Blueprint

## 0. Environment
Use the `stress` conda env for **all** work — training, feature extraction, and stats (scipy 1.17 / statsmodels 0.14 / torch 2.5.1+cu124 / timm 1.0.26). Prefer the absolute python path `/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python` over `conda run`. `stats_env` was removed 2026-04-08; `timm_eeg` has a broken scipy — do not use for new work.

## 1. Project Overview

Resting-state EEG stress classification with EEG Foundation Models. 18 subjects, 82 recordings raw (70 after `--max-duration 400` filter), binary classification (normal vs increase DASS).

### Headline finding (updated 2026-04-10 — active erosion reclassification)

Fine-tuning LaBraM produces **three modes** driven by label–biomarker correspondence strength, not by dataset size. We verify this with **two complementary metrics**: representation-level pooled label fraction and behavioral-level BA. The Stress dataset's original "no-op" classification was reclassified to **behavioral erosion** on 2026-04-10 after correcting two issues: (i) the stale `--label subject-dass` OR-aggregation, which was a trait-memorization artifact; (ii) single-seed comparisons on a 70-recording dataset where cuDNN non-determinism creates ±10 pp swings.

**Representation-level** — pooled label fraction $SS_{\text{label}}/SS_{\text{total}}$ over the 200-d encoder output (matched-subsample, 100 draws/rung, subject-level permutation null):

| Dataset | n_rec / n_subj | Frozen (full N) | FT (full N) | N-controlled Δ | Mode |
|---|---|---|---|---|---|
| ADFTD (AD) | 195 / 65 | 2.79% | 7.70% | **+5.22 to +5.68 pp across N ∈ [17, 65]** | **injection** |
| EEGMAT (task) | 72 / 36 | 5.35% | 5.82% | n/a (crossed design) | mild injection |
| TDBRAIN (MDD) | 734 / 359 | 2.97% | 1.47% | **−1.06 to −1.58 pp across N ∈ [35, 359]** | **erosion (silent — BA unchanged)** |
| ~~Stress (DASS)~~ | ~~70 / 17~~ | ~~7.24%~~ | ~~7.23%~~ | stale | reclassified — see behavioral row below |

The Stress row above (from `paper/figures/variance_analysis.json`) was computed under `--label subject-dass` and is **stale**. Needs to be regenerated under `--label dass` (per-recording) before it can be cited.

**Behavioral-level** — subject-level 5-fold BA, multi-seed robust (per-recording DASS labels, `results/studies/2026-04-10_stress_erosion/`):

| Dataset | Frozen LP BA | FT BA (real) | FT BA (shuffled-label null) | Δ Frozen − FT | Mode |
|---|---|---|---|---|---|
| Stress | **0.605 ± 0.030** (8 seed) | **0.443 ± 0.068** (3 seed) | 0.497 ± 0.081 (10 perm) | **+16.2 pp** | **behavioral erosion** |

Real FT is statistically indistinguishable from null (one-sided p(null ≥ real) = 0.70). Frozen LP's seed variance (±0.030) is less than half of FT's (±0.068), yet Frozen LP still wins by 16 pp. FT under weak cross-subject DASS labels actively degrades the pretrained encoder's linearly-separable stress signal.

Canonical sources:
- Representation: `paper/figures/variance_analysis.json` + `variance_analysis_matched.json` (ADFTD/TDBRAIN/EEGMAT only)
- Behavioral (Stress): `results/studies/2026-04-10_stress_erosion/analysis.json`
- Methodology: `docs/eta_squared_pipeline_explanation.md`, `docs/progress.md` §4.5–4.6

**Do not cite**: (i) legacy "34× subject/label ratio"; (ii) "9.1→1.6 per-fold Stress drop"; (iii) unqualified "ADFTD ×2.76 rewrite" (N-inflation artifact); (iv) canonical LaBraM subject-dass 0.656 or trial-level 0.862 as reproducible BA (both used subject-dass OR-aggregation + single seed + cuDNN flakiness); (v) 2026-04-08 variance_analysis.json Stress row (7.23% → 7.24%) until regenerated with per-rec DASS.

### Results (subject-level CV, per-recording DASS, 70-rec)

| Model | Subject BA | n seeds | Status |
|---|---|---|---|
| LaBraM **Frozen LP** (sklearn LogisticRegression on 200-d features) | **0.605 ± 0.030** | 8 | primary — beats FT |
| LaBraM FT (canonical recipe: lr=1e-5, elrs=0.1, llrd=1.0) | 0.443 ± 0.068 | 3 | erosion |
| LaBraM FT, label-permutation null | 0.497 ± 0.081 | 10 perm | FT ≈ null |

Stale reference numbers (historical, **NOT reproducible** — do not cite):
- LaBraM FT subject-dass canonical (2026-04-05 seed=42): 0.656 (lucky draw; HEAD re-run gives 0.45)
- LaBraM FT subject-dass 3-seed sweep (2026-04-09): 0.609 ± 0.044
- Classical RF 156-feat subject-dass: 0.666 (not yet re-validated under per-rec dass)
- Trial-level LaBraM FT subject-dass: 0.862 (NOT comparable to Lin 2025 — uses OR-aggregation not per-rec)
- CBraMod subject-dass FT 0.488 / REVE subject-dass FT 0.553

### Paper strategy (active erosion, not passive failure)

1. **Expose** two inflation layers in prior work: (a) trial-vs-subject split leaks (~30 pp gap), (b) subject-dass OR-aggregation = trait memorization, not stress classification.
2. **Reframe** the FM failure mode: fine-tuning on weak psychiatric labels *actively degrades* the pretrained representation. Stress Frozen LP 0.605 vs FT 0.443 (−16 pp, FT indistinguishable from null) is the headline.
3. **Cross-dataset refutation**: ADFTD (injection +8 pp behavioral), EEGMAT (mild injection), TDBRAIN (silent erosion at representation level) — rules out method / sample-size / small-data limitations. Taxonomy is driven by label–biomarker correspondence strength, not paradigm.
4. **Highest-leverage open experiment**: within-subject longitudinal reframing on UCSD Stress using DSS (continuous per-session) instead of DASS (trait class). 15/17 subjects have DSS variation; only 3/17 have DASS class transitions.

**Ruled out — do not re-propose without strong justification**: subject-adversarial GRL/DANN (~0.12 BA drop), LEAD-style subject CE loss (0.439 BA), REVE, CBraMod on Stress. Canonical 0.656 as a "LaBraM ceiling" — it was cuDNN noise.

---

## 2. Architecture

```
EEG (.set) → StressEEGDataset (epoch + cache) → FM Backbone → Global Pool → Classifier
```

- **FM extractors** (`baseline/`): unified `(B, embed_dim)`. LaBraM 200, CBraMod 200, REVE 512. Use `create_extractor(name)` — **never** pass `embed_dim`.
- **Input normalisation** (`--norm`) is **per-model**, not a global default. Getting this wrong silently destroys the run:
  - `labram` → `zscore` (matches paper FT recipe)
  - `cbramod` → `none` (extractor internally does `x / 100` in `cbramod_extractor.py:102`; passing zscored input double-scales to ~0)
  - `reve` → `none` (patch embedding is a `Linear(patch_size → embed_dim)` trained on µV-scale pretraining data; scale-sensitive)
  - `eegnet` / `shallowconvnet` / `deepconvnet` / `eegconformer` → `zscore` (Lawhern/Schirrmeister/Song conventions + early BatchNorm)
- **Training non-determinism**: `train_ft.py` seeds numpy + torch but does **not** set `torch.backends.cudnn.deterministic=True`. On the 70-rec / 14-positive Stress regime this creates ±10 pp BA swings across runs at identical HP+seed. Any Stress BA claim must be multi-seed (≥3); single-seed BAs are not reproducible. The archived 0.656 canonical was a lucky-tail draw, not a ceiling.
- **Dataset**: `data/comprehensive_labels.csv`, `--max-duration 400`, label mode **`dass`** (per-recording, matches Lin 2025 protocol), cache at `data/cache/*.pt` → `(M, 30, 1000)` at 200 Hz × 5 s windows. The old `subject-dass` OR-aggregation is deprecated (trait memorization artifact).
- **Evaluation**:
  - Subject-level CV — `StratifiedGroupKFold(5)` by patient_id — **primary metric**.
  - Trial-level CV — `StratifiedKFold(5)` on recordings — reference-paper comparison only (subject leakage).
  - Global prediction pooling; SMA-3 smoothed early stopping (patience 15); `--aug-overlap 0.75` on minority class.
  - Permutation null: `train_ft.py --permute-labels <seed>` shuffles recording-level labels before CV; use `scripts/run_perm_null.py` for a pool.

---

## 3. Key files

**Training + pipeline**
| Path | Purpose |
|---|---|
| `train_ft.py` | Subject-level CV fine-tuning (main evaluation). Flags: `--label dass` (per-rec, default), `--permute-labels <seed>` (null test), `--save-features` (dump test-fold features). |
| `train_trial.py` | Trial-level CV (reference comparison only — leaks subject identity). |
| `train_lp.py` | Linear probing baseline (frozen encoder + MLP head). |
| `pipeline/dataset.py` | `StressEEGDataset`, `WindowDataset`, `RecordingGroupSampler` |
| `pipeline/eegmat_dataset.py` | EEGMAT within-subject loader |
| `src/model.py` | `DecoupledStressModel` (extract_pooled + classify) |
| `src/variance_analysis.py` | Nested SS, mixed-effects, cluster bootstrap, PERMANOVA, label-subspace (top-level numpy-only; `mixed_effects_variance` lazy-imports statsmodels) |

**Analysis + figure scripts**
| Path | Purpose |
|---|---|
| `scripts/run_variance_analysis.py` | Regenerates `paper/figures/variance_analysis.json` (ADFTD/TDBRAIN/EEGMAT only; Stress row pending re-run) |
| `scripts/stress_frozen_lp_multiseed.py` | 8-seed Frozen LP on Stress per-rec DASS → `studies/2026-04-10_stress_erosion/frozen_lp/multi_seed.json` |
| `scripts/run_perm_null.py` | Pool-launches N permutation-null FT runs |
| `scripts/run_hp_sweep.py` + `summarize_sweep.py` | HP grid sweep + leaderboard aggregation |
| `scripts/build_cross_dataset_figure.py` | Cross-dataset bar figure (`paper/figures/cross_dataset_signal_strength.pdf`) |
| `scripts/build_matched_curve_figure.py` | Matched-subsample curves figure |
| `scripts/build_label_subspace_figure.py` | Label-subspace PCA figure (Stress row removed pending per-rec re-run) |
| `scripts/build_results_index.py` | Auto-generate `results/results_index.{csv,md}` (currently empty under new layout) |

**Results layout** (see `results/README.md` for full guide)
- `results/features_cache/` — cached frozen + FT features (LaBraM 19ch)
- `results/studies/YYYY-MM-DD_<slug>/` — self-contained investigations with README + analysis.json
- `results/archive/YYYY-MM-DD_<slug>/` — read-only historical runs

**Docs**
| Path | Purpose |
|---|---|
| `docs/eta_squared_pipeline_explanation.md` | Variance-decomposition methodology reference |
| `docs/related_work.md` | Living literature review |
| `docs/paper_strategy.md` | Paper framing & open experiments |
| `docs/progress.md` | Running log of findings + decisions |

---

## 4. Current priorities

1. **Align advisor on the erosion reframing** — new headline is "fine-tuning actively degrades representations under weak psychiatric labels", not "method-independent 0.66 ceiling". Draft summary: `studies/2026-04-10_stress_erosion/README.md` + `analysis.json`.
2. **Re-run Stress for variance_analysis.json** with `--label dass --save-features` (canonical recipe), then update `scripts/run_variance_analysis.py` to re-include Stress row and regenerate `paper/figures/variance_analysis.json`. Current representation-level Stress row is stale (computed under subject-dass).
3. **Medium priority** — `--label dass` extension of the FM comparison:
   - Classical RF on per-rec dass (fast)
   - CBraMod + REVE canonical recipe on per-rec dass (prove erosion isn't LaBraM-specific)
4. **HNC Dementia + MDD private dataset** integration (branch `feat/hnc-dementia-mdd`): encrypted HDF5+pkl, 30ch mV scale, 308+400 subjects, `hnc.function_hnc` decryption. Use HNC Dementia as an independent N-controlled replication of ADFTD's +5 pp injection mode, and HNC MDD as an N-controlled replication of TDBRAIN's erosion mode.
5. **Within-subject longitudinal reframing** on UCSD stress using DSS (continuous per-session) — highest-leverage open experiment; if subject BA ≥ 0.72 the paper title pivots to lead with this.
6. **Stat hardening (partial)**: matched-subsample resampling ✓, subject-level label-permutation null ✓, behavioral permutation null on Stress ✓. TODO: bootstrap CIs on BA, sign test on trial-vs-subject gap, Classical RF baseline under per-rec dass.

### Guard rails
- Per-fold ω² is mathematically degenerate on Stress (1 subject per positive class per fold) → `analyze_regime` emits `nested_identifiable: false`; downstream code must refuse to cite those numbers.
- Sparse-label-subspace hypothesis was tested and **refuted** (`label_subspace_analysis` in `src/variance_analysis.py`): FT does not concentrate label signal into fewer dims.
- `--label subject-dass` is **deprecated**. It OR-aggregates per-subject labels ("if any recording is increase, all of this subject's recordings become increase"), which turns the task into subject-identity broadcast rather than stress-state classification. All new experiments use `--label dass` (per-recording). Existing subject-dass runs are archived under `results/archive/2026-04-09_subject_dass_cleanup/`.
- Single-seed BAs on Stress are **not** reproducible claims — cuDNN non-determinism produces ±10 pp swings. Always report multi-seed mean ± std (n ≥ 3).
