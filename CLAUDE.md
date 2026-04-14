# CLAUDE.md — UCSD_stress Technical Reference

**Findings**: `docs/findings.md` (single source of truth for all scientific claims)
**Priorities**: `docs/TODO.md` (living task list)
**Paper strategy**: `docs/paper_strategy.md`
**Literature**: `docs/related_work.md`
**Methodology**: `docs/eta_squared_pipeline_explanation.md`
**History**: `docs/progress.md` (append-only log, not authoritative — findings.md supersedes)

---

## 0. Environment

Use the `stress` conda env for **all** work — training, feature extraction, and stats (scipy 1.17 / statsmodels 0.14 / torch 2.5.1+cu124 / timm 1.0.26). Prefer the absolute python path `/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python` over `conda run`. `stats_env` was removed 2026-04-08; `timm_eeg` has a broken scipy — do not use for new work.

---

## 1. Project Overview

Resting-state EEG stress classification with EEG Foundation Models. 17 subjects, 92 recordings total (`comprehensive_labels_stress.csv`), 82 with duration ≤ 400 s (Lin 2025 protocol), 70 with valid DASS scores (`comprehensive_labels.csv`). Our experiments use the 70-recording subset (`--label dass`); paper notes the difference from Lin's 82. `Binary_Stress` column was removed — all labelling goes through `--label dass` with `--threshold`.

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
- **Training determinism**: `train_ft.py` sets `torch.backends.cudnn.deterministic=True` and `benchmark=False` (added 2026-04-10). Seed variance on 70-rec / 14-positive regime is still ±5–10 pp. All Stress BA claims require multi-seed (≥3).
- **Dataset**: `data/comprehensive_labels.csv`, `--max-duration 400`, label mode **`dass`** (per-recording, matches Lin 2025 protocol), cache at `data/cache/*.pt` → `(M, 30, 1000)` at 200 Hz × 5 s windows.
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
| `train_ft.py` | Subject-level CV fine-tuning. Flags: `--label dass` (default), `--permute-labels <seed>`, `--save-features`. |
| `train_trial.py` | Trial-level CV (reference comparison only). |
| `train_lp.py` | Linear probing baseline (frozen encoder + MLP head). |
| `pipeline/dataset.py` | `StressEEGDataset`, `WindowDataset`, `RecordingGroupSampler` |
| `pipeline/eegmat_dataset.py` | EEGMAT within-subject loader |
| `src/model.py` | `DecoupledStressModel` (extract_pooled + classify) |
| `src/variance_analysis.py` | Nested SS, mixed-effects, cluster bootstrap, PERMANOVA, label-subspace |

**Analysis + figure scripts**
| Path | Purpose |
|---|---|
| `scripts/run_variance_analysis.py` | Regenerates `paper/figures/variance_analysis.json` |
| `scripts/stress_frozen_lp_multiseed.py` | Multi-seed Frozen LP |
| `scripts/run_perm_null.py` | Pool-launches permutation-null FT runs |
| `scripts/run_hp_sweep.py` + `summarize_sweep.py` | HP grid sweep + leaderboard |
| `scripts/build_cross_dataset_figure.py` | Cross-dataset bar figure |
| `scripts/build_matched_curve_figure.py` | Matched-subsample curves figure |
| `scripts/build_label_subspace_figure.py` | Label-subspace PCA figure |

**Results layout** (see `results/README.md` for full guide)
- `results/features_cache/` — cached frozen + FT features
- `results/studies/exp##_<slug>/` — self-contained investigations (exp01–exp11 current; new experiments increment from next available ID)
- `results/hp_sweep/` — hyperparameter sweep results
- `results/archive/` — read-only historical runs

---

## 4. Guard rails

- **`--label subject-dass` is deprecated.** OR-aggregation = trait memorization. Use `--label dass` (per-recording). See `docs/findings.md` F07.
- **Single-seed BAs on Stress are not reproducible.** ±5–10 pp swings even with `cudnn.deterministic=True`. Always multi-seed (≥3). See F08.
- **Per-fold ω² is degenerate on Stress** (1 subject per positive class per fold). `analyze_regime` emits `nested_identifiable: false`; refuse to cite per-fold numbers.
- **Sparse-label-subspace hypothesis was refuted.** FT does not concentrate label signal into fewer dims.
- **Cross-dataset taxonomy is LaBraM-only.** Do not generalize to other FMs without running those experiments. See F04.
- **FM input norm is per-model.** Never use a global `--norm` in multi-model sweeps. See §2 above.
- **GPU usage limit: max 3 GPUs.** Leave the remaining GPUs free for the user's own work. Chain sequential jobs on the same GPU with `&&` if needed.
