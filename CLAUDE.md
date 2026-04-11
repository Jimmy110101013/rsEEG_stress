# CLAUDE.md — UCSD_stress Architecture Blueprint

## 0. Environment
Use the `stress` conda env for **all** work — training, feature extraction, and stats (scipy 1.17 / statsmodels 0.14 / torch 2.5.1+cu124 / timm 1.0.26). Prefer the absolute python path `/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python` over `conda run`. `stats_env` was removed 2026-04-08; `timm_eeg` has a broken scipy — do not use for new work.

## 1. Project Overview

Resting-state EEG stress classification with EEG Foundation Models. 18 subjects, 82 recordings raw (70 after `--max-duration 400` filter), binary classification (normal vs increase DASS).

### Headline finding (updated 2026-04-11 — multi-model HP sweep)

Fine-tuning on UCSD Stress (per-recording DASS labels) produces **model-dependent** outcomes: LaBraM shows **behavioral erosion** (FT degrades frozen representation), while CBraMod and REVE show **mild injection** (FT improves over weak frozen baseline). Cross-dataset taxonomy (ADFTD/TDBRAIN/EEGMAT) is currently **LaBraM-only** — other FMs not yet tested on those datasets.

**Cross-dataset representation-level** (LaBraM only) — pooled label fraction $SS_{\text{label}}/SS_{\text{total}}$ over the 200-d encoder output (matched-subsample, 100 draws/rung, subject-level permutation null):

| Dataset | n_rec / n_subj | Frozen (full N) | FT (full N) | N-controlled Δ | Mode |
|---|---|---|---|---|---|
| ADFTD (AD) | 195 / 65 | 2.79% | 7.70% | **+5.22 to +5.68 pp across N ∈ [17, 65]** | **injection** |
| EEGMAT (task) | 72 / 36 | 5.35% | 5.82% | n/a (crossed design) | mild injection |
| TDBRAIN (MDD) | 734 / 359 | 2.97% | 1.47% | **−1.06 to −1.58 pp across N ∈ [35, 359]** | **erosion (silent — BA unchanged)** |
| ~~Stress (DASS)~~ | ~~70 / 17~~ | ~~7.24%~~ | ~~7.23%~~ | stale | reclassified — see behavioral rows below |

Note: these cross-dataset modes are **LaBraM-only**. CBraMod/REVE have not been tested on ADFTD/TDBRAIN/EEGMAT. The Stress row was computed under `--label subject-dass` and is **stale** — needs regeneration under `--label dass`.

**Behavioral-level on Stress** — subject-level 5-fold BA, multi-seed, per-recording DASS:

**LaBraM erosion analysis** (`results/studies/2026-04-10_stress_erosion/`):

| Condition | BA | n seeds | Notes |
|---|---|---|---|
| LaBraM Frozen LP | **0.605 ± 0.030** | 8 | primary |
| LaBraM FT (canonical: lr=1e-5, elrs=0.1) | 0.443 ± 0.068 | 3 | erosion (−16.2 pp) |
| LaBraM FT, permutation null | 0.497 ± 0.081 | 10 perm | real FT ≈ null (p=0.70) |

**Multi-model HP sweep** (`results/hp_sweep/20260410_dass/`, 3 seeds × 6 HP configs per model):

| Model | Frozen LP (8-seed) | Best FT (3-seed) | Best FT config | Δ (FT − Frozen) |
|---|---|---|---|---|
| **LaBraM** | **0.605 ± 0.030** | 0.524 ± 0.008 | lr=1e-4, elrs=1.0 | **−8.1 pp (erosion)** |
| **CBraMod** | 0.452 ± 0.030 | **0.548 ± 0.026** | lr=1e-5, elrs=0.1 | **+8.3 pp (injection)** |
| **REVE** | 0.494 ± 0.017 | **0.577 ± 0.041** | lr=3e-5, elrs=0.1 | **+8.0 pp (injection)** |

Key insight: erosion on Stress is **LaBraM-specific**, not model-universal. LaBraM's frozen representation is already strong (0.605); FT degrades it. CBraMod/REVE have weaker frozen representations (0.45/0.49); FT can improve them. Whether this pattern holds on other datasets is untested.

**EEGMAT 3-seed study** (`results/studies/2026-04-10_eegmat_feat_multiseed/`):
LaBraM FT on EEGMAT: BA = 0.731 ± 0.017 (3 seeds), features saved for variance analysis.

Canonical sources:
- Representation: `paper/figures/variance_analysis.json` + `variance_analysis_matched.json` (ADFTD/TDBRAIN/EEGMAT, LaBraM only)
- Behavioral (Stress): `results/studies/2026-04-10_stress_erosion/analysis.json`
- HP sweep: `results/hp_sweep/20260410_dass/` (LaBraM + CBraMod + REVE, 54 runs)
- Methodology: `docs/eta_squared_pipeline_explanation.md`, `docs/progress.md` §4.5–4.7

**Do not cite**: (i) legacy "34× subject/label ratio"; (ii) "9.1→1.6 per-fold Stress drop"; (iii) unqualified "ADFTD ×2.76 rewrite" (N-inflation artifact); (iv) canonical LaBraM subject-dass 0.656 or trial-level 0.862 as reproducible BA (both used subject-dass OR-aggregation + single seed + cuDNN flakiness); (v) 2026-04-08 variance_analysis.json Stress row (7.23% → 7.24%) until regenerated with per-rec DASS; (vi) "erosion is model-universal" — refuted by HP sweep (CBraMod/REVE show injection on Stress).

### Results (subject-level CV, per-recording DASS, 70-rec)

| Model | Frozen LP BA (8-seed) | Best FT BA (3-seed) | Best FT config | Δ | Status |
|---|---|---|---|---|---|
| **LaBraM** | **0.605 ± 0.030** | 0.524 ± 0.008 | lr=1e-4, elrs=1.0 | −8.1 pp | **erosion** |
| **CBraMod** | 0.452 ± 0.030 | **0.548 ± 0.026** | lr=1e-5, elrs=0.1 | +8.3 pp | injection |
| **REVE** | 0.494 ± 0.017 | **0.577 ± 0.041** | lr=3e-5, elrs=0.1 | +8.0 pp | injection |

LaBraM canonical recipe (lr=1e-5, elrs=0.1): 0.443 ± 0.068 (3 seed) — erosion gap is 16.2 pp with this config, 8.1 pp with best HP.
LaBraM FT permutation null: 0.497 ± 0.081 (10 perm). Canonical FT ≈ null (p=0.70).

Stale reference numbers (historical, **NOT reproducible** — do not cite):
- LaBraM FT subject-dass canonical (2026-04-05 seed=42): 0.656 (lucky draw; HEAD re-run gives 0.45)
- LaBraM FT subject-dass 3-seed sweep (2026-04-09): 0.609 ± 0.044
- Classical RF 156-feat subject-dass: 0.666 (not yet re-validated under per-rec dass)
- Trial-level LaBraM FT subject-dass: 0.862 (NOT comparable to Lin 2025 — uses OR-aggregation not per-rec)
- CBraMod subject-dass FT 0.488 / REVE subject-dass FT 0.553 (subject-dass, not per-rec)

### Paper strategy (model-dependent erosion/injection)

1. **Expose** two inflation layers in prior work: (a) trial-vs-subject split leaks (~30 pp gap), (b) subject-dass OR-aggregation = trait memorization, not stress classification.
2. **Reframe** the FM finding: on Stress, LaBraM FT degrades a strong frozen representation (−8.1 pp with best HP, −16.2 pp with canonical HP), while CBraMod/REVE FT improves weak frozen representations (+8 pp each). Erosion requires the frozen representation to already be strong.
3. **Cross-dataset taxonomy** (LaBraM only): ADFTD (injection), EEGMAT (mild injection), TDBRAIN (silent erosion) — shows FT mode is driven by label–biomarker strength. **Not yet tested with CBraMod/REVE on these datasets.**
4. **Highest-leverage open experiment**: within-subject longitudinal reframing on UCSD Stress using DSS (continuous per-session) instead of DASS (trait class). 15/17 subjects have DSS variation; only 3/17 have DASS class transitions.

**Ruled out — do not re-propose without strong justification**: subject-adversarial GRL/DANN (~0.12 BA drop), LEAD-style subject CE loss (0.439 BA). Canonical 0.656 as a "LaBraM ceiling" — it was cuDNN noise.

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
- **Training determinism**: `train_ft.py` now sets `torch.backends.cudnn.deterministic=True` and `benchmark=False` (added 2026-04-10). However, on the 70-rec / 14-positive Stress regime, seed variance is still significant (±5–10 pp across seeds). Any Stress BA claim must be multi-seed (≥3); single-seed BAs are not reproducible. The archived 0.656 canonical was a lucky-tail draw, not a ceiling.
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

1. **Align advisor on updated findings** — erosion is LaBraM-specific on Stress (not model-universal). CBraMod/REVE show injection. Cross-dataset taxonomy is LaBraM-only and needs multi-model validation.
2. **Re-run Stress for variance_analysis.json** with `--label dass --save-features` (canonical recipe), then update `scripts/run_variance_analysis.py` to re-include Stress row and regenerate `paper/figures/variance_analysis.json`. Current representation-level Stress row is stale (computed under subject-dass).
3. **Cross-model validation on other datasets** — Run CBraMod + REVE Frozen LP + FT on ADFTD/TDBRAIN/EEGMAT to determine whether injection/erosion modes are model-universal or model-specific on those datasets too.
4. **Classical RF on per-rec dass** — re-validate classical baseline under honest labels.
5. **HNC Dementia + MDD private dataset** integration (branch `feat/hnc-dementia-mdd`): encrypted HDF5+pkl, 30ch mV scale, 308+400 subjects, `hnc.function_hnc` decryption.
6. **Within-subject longitudinal reframing** on UCSD stress using DSS (continuous per-session) — highest-leverage open experiment; if subject BA ≥ 0.72 the paper title pivots to lead with this.
7. **Stat hardening (partial)**: matched-subsample resampling ✓, subject-level label-permutation null ✓, behavioral permutation null on Stress ✓. TODO: bootstrap CIs on BA, sign test on trial-vs-subject gap.

### Guard rails
- Per-fold ω² is mathematically degenerate on Stress (1 subject per positive class per fold) → `analyze_regime` emits `nested_identifiable: false`; downstream code must refuse to cite those numbers.
- Sparse-label-subspace hypothesis was tested and **refuted** (`label_subspace_analysis` in `src/variance_analysis.py`): FT does not concentrate label signal into fewer dims.
- `--label subject-dass` is **deprecated**. It OR-aggregates per-subject labels ("if any recording is increase, all of this subject's recordings become increase"), which turns the task into subject-identity broadcast rather than stress-state classification. All new experiments use `--label dass` (per-recording). Existing subject-dass runs are archived under `results/archive/2026-04-09_subject_dass_cleanup/`.
- Single-seed BAs on Stress are **not** reproducible claims — seed variance produces ±5–10 pp swings even with `cudnn.deterministic=True`. Always report multi-seed mean ± std (n ≥ 3).
- Cross-dataset taxonomy (ADFTD injection, TDBRAIN erosion, EEGMAT mild injection) is **LaBraM-only**. Do not generalize to other FMs without running those experiments.
