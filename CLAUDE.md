# CLAUDE.md — UCSD_stress Technical Reference

**Findings** (paper claims): `docs/findings.md` — 5 CLAIMs, `F-A`…`F-E`
**Methodology notes**: `docs/methodology_notes.md` — guardrails (G-F07/F08/F10/F12), internal notes, archived entries
**Priorities**: `docs/TODO.md` (living task list)
**Paper outline**: `docs/paper_outline.md` (IMRaD + figures/tables layout)
**Experiments index**: `docs/paper_experiments_index.md` (single navigation doc — §/Fig/Table → source data/script/log/status)
**Literature**: `docs/related_work.md`
**η² pipeline**: `docs/eta_squared_pipeline_explanation.md`
**Historical archive** (read-only): `docs/historical/` — prior drafts, reviews (R1–R4), pre-SDL strategy, `progress.md` legacy log, sdl_paper_draft v1/v2
**Reference papers / specs**: `docs/reference/` (external PDFs — CbraMod, REVE, EEG-FM-Bench, Komarov, Wang 2025, etc.); `specs/` (HHT integration, normalization ablation — reference-only, never auto-loaded)

> **Canonical results access**: paper figure/table builders and notebooks MUST read numbers through `from src import results` (see `src/results.py`), NOT by memorising filesystem paths. Schema spec + accessor contract: `results/final/README.md`.

> **ID migration (2026-04-15)**: old `F01`–`F21` were consolidated into 5 paper claims (`F-A`…`F-E`) plus guardrails/notes. Each claim's `Absorbs:` line in `findings.md` lists the original F## IDs, so prior references resolve via that mapping.

---

## 0. Environment

Use the `stress` conda env for **all** work — training, feature extraction, and stats (scipy 1.17 / statsmodels 0.14 / torch 2.5.1+cu124 / timm 1.0.26). Prefer the absolute python path `/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python` over `conda run`. `stats_env` was removed 2026-04-08; `timm_eeg` has a broken scipy — do not use for new work.

---

## 1. Project Overview

Resting-state EEG stress classification with EEG Foundation Models. Dataset from Komarov, Ko, Jung (2020, IEEE TNSRE 28(4):795) — Taiwan graduate students, longitudinal resting-state EEG with DASS-21 + DSS collected per recording. 17 subjects, 92 recordings total (`comprehensive_labels_stress.csv`); we use the 70 recordings that have **both DASS-21 and DSS labels** (`comprehensive_labels.csv`, `--label dass`). Wang et al. (2025, arXiv:2505.23042, same lab) reports 90.47% BA on this dataset under trial-level CV — we treat this as the inflated baseline. `Binary_Stress` column was removed — all labelling goes through `--label dass` with `--threshold`.

---

## 2. Architecture

```
EEG (.set) → StressEEGDataset (epoch + cache) → FM Backbone → Global Pool → Classifier
```

- **FM extractors** (`baseline/`): unified `(B, embed_dim)`. LaBraM 200, CBraMod 200, REVE 512. Use `create_extractor(name)` — **never** pass `embed_dim`.
- **Input normalisation** (`--norm`) is **per-model**, not a global default. Getting this wrong silently destroys the run:
  - `labram` → `none` (extractor internally does `x / 100` in `labram_extractor.py:142` — matches original LaBraM `engine_for_finetuning.py:87`; aligned 2026-04-26. **Pre-2026-04-26 results used `zscore`** — see git log around 2026-04-26 for the alignment commit; old runs are not directly comparable to post-fix runs.)
  - `cbramod` → `none` (extractor internally does `x / 100` in `cbramod_extractor.py:102`; passing zscored input double-scales to ~0)
  - `reve` → `none` (patch embedding is a `Linear(patch_size → embed_dim)` trained on µV-scale pretraining data; scale-sensitive)
  - `eegnet` / `shallowconvnet` / `deepconvnet` / `eegconformer` → `zscore` (Lawhern/Schirrmeister/Song conventions + early BatchNorm)
- **Training determinism**: `train_ft.py` sets `torch.backends.cudnn.deterministic=True` and `benchmark=False` (added 2026-04-10). Seed variance on 70-rec / 14-positive regime is still ±5–10 pp. All Stress BA claims require multi-seed (≥3).
- **Dataset**: `data/comprehensive_labels.csv` (70 rows — recordings with both DASS-21 and DSS labels), label mode **`dass`** (per-recording), cache at `data/cache/*.pt` → `(M, 30, 1000)` at 200 Hz × 5 s windows. `--max-duration` flag exists but is not used in the main pipeline; no 400 s filter is applied.
- **Evaluation**:
  - Subject-level CV — `StratifiedGroupKFold(5)` by patient_id — **primary metric**.
  - Trial-level CV — `StratifiedKFold(5)` on recordings — reference-paper comparison only (subject leakage).
  - Global prediction pooling; SMA-3 smoothed early stopping (patience 15); `--aug-overlap 0.75` on minority class.
  - Permutation null: `train_ft.py --permute-labels <seed>` shuffles recording-level labels before CV; use `scripts/experiments/run_perm_null.py` for a pool.

---

## 3. Key files

**Training + pipeline**
| Path | Purpose |
|---|---|
| `train_ft.py` | Subject-level CV fine-tuning. Flags: `--label dass` (default), `--permute-labels <seed>`, `--save-features`. |
| `train_trial.py` | Trial-level CV (reference comparison only). |
| `train_lp.py` | **Canonical LP entry point** (rewritten 2026-04-25, G-F10): per-window sklearn LogReg + train-fit percentile clip + recording-level mean-pool → BA. CLI: `python train_lp.py --extractor {labram,cbramod,reve} --dataset {stress,eegmat,adftd,tdbrain,sleepdep,meditation} [--cv {stratified-kfold,loso}]`. Library: `from train_lp import run_canonical_lp, eval_seed`. The pre-rewrite pool-then-classify body is preserved at git tag `lp-pool-then-classify-v1`. |
| `pipeline/dataset.py` | `StressEEGDataset`, `WindowDataset`, `RecordingGroupSampler` (with `_preprocess` fallback when cache missing) |
| `pipeline/eegmat_dataset.py` | EEGMAT within-subject loader |
| `pipeline/meditation_dataset.py` | OpenNeuro ds001787 loader (BioSemi 64ch → COMMON_19; expert/novice between-subject) |
| `pipeline/sam40_dataset.py` | SAM40 stress dataset loader |
| `src/model.py` | `DecoupledStressModel` (extract_pooled + classify) |
| `src/variance_analysis.py` | Nested SS, mixed-effects, cluster bootstrap, PERMANOVA, label-subspace |
| `src/wsci.py` | WSCI (Within-Subject Contrast Index) metric |
| `src/results.py` | **Canonical results accessor**. Paper builders read numbers through `results.source_table(...)`, `results.lp_multiseed(...)`, `results.lp_stats_3seed(...)`, `results.ft_stats(...)`, `results.perm_null_summaries(...)`, `results.labram_ft_ba_null_matched(...)`. Module grows on demand — add a function when a builder needs one, not before. Path changes live here, callers unaffected. |

**Scripts layout** (subdirectories under `scripts/`)
| Subdirectory | Contents |
|---|---|
| `scripts/figures/` | Active paper figure builders: `build_fig2_2x2.py`, `build_fig3_perm_null_4panel.py`, `build_master_performance_table.py`, `build_paper_pdf.py` (4 files). Fig 4/5/6 + appendix figures produced by `notebooks/_build_figures_consolidated.py`. 23 pre-pivot SDL-era builders retired 2026-04-25 (available via `git log`). |
| `scripts/hhsa/` | HHSA pipeline — cache, holospectra, analysis directions (9 scripts) |
| `scripts/experiments/` | Experiment launchers — `.py` orchestrators (`run_perm_null.py`, `run_hp_sweep.py`, `frozen_lp_perwindow_all.py` etc.) + 9 reusable `.sh` templates (`run_final_ft_*.sh`, `run_ft_winfeat_*.sh`, `run_variance_triangulation*.sh`, `snapshot_variance_triangulation.sh`). One-time exp chains retired 2026-04-25. |
| `scripts/features/` | Feature extraction — frozen + FT (8 scripts) |
| `scripts/analysis/` | Statistical analysis, PSD anchors, variance, WSCI, representation drift (~37 scripts) |

**Canonical LP / key analysis scripts**
| Path | Purpose |
|---|---|
| `train_lp.py` | Canonical LP (see above in Training row) |
| `scripts/experiments/run_perm_null.py` | Pool-launches permutation-null FT runs |
| `scripts/experiments/run_hp_sweep.py` | HP grid sweep |
| `scripts/analysis/run_variance_analysis.py` | Regenerates variance analysis JSON (default out: `paper/figures/variance_analysis.json` — override with `--out` to land under current figure layout) |
| `scripts/analysis/build_master_results_table.py` | Master results table JSON generator |
| `scripts/analysis/summarize_sweep.py` | HP sweep leaderboard |
| `scripts/analysis/representation_drift_lp_vs_ft.py` | LP vs FT representation drift |
| `scripts/analysis/sleepdep_within_subject.py`, `compute_sleepdep_dir_consistency.py` | SleepDep within-subject direction consistency |
| `scripts/figures/build_fig2_2x2.py` | 2×2 factorial panel (task-substrate × regime) |

**Results layout** (see `results/README.md`)
- `results/final/<dataset>/<experiment_type>/<fm>/seed*/` — canonical per-paper results (ft / lp / classical / band_stop / nonfm_deep / perm_null / variance_triangulation / subject_probe_temporal_block / fooof_ablation). Datasets: adftd / eegmat / sleepdep / stress / tdbrain / cross_cell.
- `results/final_winfeat/` — parallel window-feature snapshot layer (same shape as `final/` but keeping fold-level feature artifacts).
- `results/studies/exp##_<slug>/` — self-contained investigations. Current range: exp01–exp33 (exp22 skipped) + non-numbered (`exp_newdata`, `exp_30_sdl_vs_between`, `fooof_ablation`, `ft_vs_lp`, `perwindow_lp_all`, `representation_drift`, `sanity_lead_adftd`). New experiments increment from next available ID.
- `results/hp_sweep/` — hyperparameter sweep results (`20260410_dass`).
- `results/archive/` — read-only historical runs (pre-2026-04-10 work; dated subdirs).
- `results/features_cache/` — frozen + FT feature caches (`.npz` and `ft_*` dirs are `.gitignore`d; one canonical tracked checkpoint).
- `results/hhsa/` — HHSA analysis outputs (01_eyeball..08_am_coherence + cross_dataset_comparison). `cache/` and `holospectra/` are `.gitignore`d intermediates.

---

## 4. Guard rails

- **`--label subject-dass` is deprecated.** OR-aggregation = trait memorization. Use `--label dass` (per-recording). See `docs/methodology_notes.md` G-F07.
- **Single-seed BAs on Stress are not reproducible.** ±5–10 pp swings even with `cudnn.deterministic=True`. Always multi-seed (≥3). See G-F08.
- **Per-fold ω² is degenerate on Stress** (1 subject per positive class per fold). `analyze_regime` emits `nested_identifiable: false`; refuse to cite per-fold numbers.
- **Sparse-label-subspace hypothesis was refuted.** FT does not concentrate label signal into fewer dims.
- **Cross-dataset FT taxonomy is now 3-model × 3-dataset** (F-C in `findings.md`). The old "LaBraM-only" constraint was F04 (now archived as A-F04 in `methodology_notes.md`).
- **FM input norm is per-model.** Never use a global `--norm` in multi-model sweeps. See §2 above.
- **GPU usage limit: max 3 GPUs.** Leave the remaining GPUs free for the user's own work. Chain sequential jobs on the same GPU with `&&` if needed.
- **Index discipline — `docs/paper_experiments_index.md` is the single navigation doc.** Any commit that touches `results/studies/**`, `results/final/**`, `results/final_winfeat/**`, `results/features_cache/**`, `results/hp_sweep/**`, `paper/figures/**/*.{pdf,png}`, `paper/tables/**`, or `paper/figures/_historical/source_tables/**` **must** update `docs/paper_experiments_index.md` in the same commit — update the matching section's Status / source-path / last-updated column, or add a new row. Never create parallel `results/my_task_2026-xx-xx/` directories; keep artifacts at canonical paths and let the index be the map. Task-bounded refresh plans are companion docs, not replacements for the index; retire them into `docs/historical/` once the scope closes.
