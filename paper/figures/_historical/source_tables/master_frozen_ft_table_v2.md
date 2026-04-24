# Master table v2: Frozen LP (per-window) vs FT (canonical per-FM recipe)

**Date**: 2026-04-20. Replaces `master_frozen_ft_table.md` (v1).

## Protocol

- **Frozen LP**: per-window LogReg matched to FT's per-window training + prediction-pooling; percentile-clip + StandardScaler; `liblinear` solver; `StratifiedGroupKFold(5)` subject-disjoint; 8 seeds.
- **Fine-tuning**: per-FM canonical recipe — LaBraM `(lr=1e-5, encoder_lr_scale=0.1, zscore)`; CBraMod `(lr=1e-5, elrs=0.1, none)`; REVE `(lr=3e-5, elrs=0.1, none)`. Same recipe across all 4 datasets (no per-dataset HP tuning). 3 seeds.
- **Stress LaBraM FT source**: `exp03_stress_erosion/ft_real/` (canonical lr=1e-5), **NOT** `hp_sweep/20260410_dass/` best-HP (lr=1e-4). Consistency across datasets is the criterion.
- Balanced accuracy; sample std (ddof=1).

## Results table

| Model | Phase | Stress | ADFTD | TDBRAIN | EEGMAT |
|---|---|---|---|---|---|
| LaBraM | Frozen LP | 0.525 ± 0.040 (n=8) | 0.653 ± 0.018 (n=8) | 0.752 ± 0.011 (n=8) | 0.736 ± 0.025 (n=8) |
|  | Fine-tune | 0.443 ± 0.083 (n=3) | 0.709 ± 0.014 (n=3) | 0.665 ± 0.045 (n=3) | 0.731 ± 0.021 (n=3) |
| CBraMod | Frozen LP | 0.430 ± 0.033 (n=8) | 0.570 ± 0.030 (n=8) | 0.525 ± 0.022 (n=8) | 0.722 ± 0.022 (n=8) |
|  | Fine-tune | 0.548 ± 0.031 (n=3) | 0.537 ± 0.027 (n=3) | 0.488 ± 0.014 (n=3) | 0.620 ± 0.058 (n=3) |
| REVE | Frozen LP | 0.441 ± 0.022 (n=8) | 0.652 ± 0.027 (n=8) | 0.519 ± 0.022 (n=8) | 0.733 ± 0.021 (n=8) |
|  | Fine-tune | 0.577 ± 0.051 (n=3) | 0.658 ± 0.030 (n=3) | 0.488 ± 0.019 (n=3) | 0.727 ± 0.035 (n=3) |

## Δ (FT − Frozen LP), percentage points

| Model | Stress | ADFTD | TDBRAIN | EEGMAT |
|---|---|---|---|---|
| LaBraM | -8.1 | +5.6 | -8.7 | -0.5 |
| CBraMod | +11.8 | -3.3 | -3.7 | -10.2 |
| REVE | +13.7 | +0.6 | -3.1 | -0.6 |

## Verdict count (12 cells)

- **FT > LP (Δ > +0.01)**: 3 cells
- **FT ≈ LP (|Δ| ≤ 0.01)**: 3 cells
- **FT < LP (Δ < −0.01)**: 6 cells

## Per-cell verdicts

| Cell | Δ | Verdict |
|---|---:|---|
| LaBraM × Stress | -8.1 pp | ↓ FT degrades |
| LaBraM × ADFTD | +5.6 pp | ↑ FT rescues |
| LaBraM × TDBRAIN | -8.7 pp | ↓ FT degrades |
| LaBraM × EEGMAT | -0.5 pp | ≈ tied |
| CBraMod × Stress | +11.8 pp | ↑ FT rescues |
| CBraMod × ADFTD | -3.3 pp | ↓ FT degrades |
| CBraMod × TDBRAIN | -3.7 pp | ↓ FT degrades |
| CBraMod × EEGMAT | -10.2 pp | ↓ FT degrades |
| REVE × Stress | +13.7 pp | ↑ FT rescues |
| REVE × ADFTD | +0.6 pp | ≈ tied |
| REVE × TDBRAIN | -3.1 pp | ↓ FT degrades |
| REVE × EEGMAT | -0.6 pp | ≈ tied |

## Key differences vs v1

| Cell | v1 LP (feature-avg 8s) | v2 LP (per-window 8s) | v1 FT | v2 FT | Notes |
|---|---:|---:|---:|---:|---|
| LaBraM × Stress  | 0.605 ± 0.032 | **0.525 ± 0.040** | 0.524 ± 0.010 | **0.443 ± 0.083** | v1 LP was feat-avg (inflated), FT was hp_sweep best-HP (lr=1e-4) not canonical |
| CBraMod × Stress | 0.452 ± 0.032 | **0.430 ± 0.033** | 0.548 ± 0.031 | 0.548 ± 0.031 | LP per-window lower; FT identical |
| REVE × Stress    | 0.494 ± 0.018 | **0.441 ± 0.022** | 0.577 ± 0.051 | 0.577 ± 0.051 | LP per-window lower; FT identical |
| Others (9 cells) | old 3-seed feat-avg | **new 8-seed per-window** | unchanged | unchanged | LP more rigorous, FT unchanged |

## Dataset metadata

| Dataset | Task | N subjects | N recordings | Labels |
|---|---|---|---|---|
| Stress  | DASS stress state        | 17  | 70  | 14 pos / 56 neg (per-recording) |
| ADFTD   | AD vs HC                 | 65  | 195 | 108 pos / 87 neg |
| TDBRAIN | MDD vs HC                | 359 | 734 | 640 pos / 94 neg |
| EEGMAT  | rest vs arithmetic task  | 36  | 72  | 36 rest / 36 task (paired within-subject) |
