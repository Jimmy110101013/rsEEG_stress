# Per-window Frozen LP — Apples-to-Apples Results

**Date**: 2026-04-20
**Purpose**: Replace all draft LP numbers with apples-to-apples protocol matching FT's per-window training + prediction-pooling (was feature-averaged before).

## Protocol

**Previous (all prior results)**: mean-pool per-window FM features within each recording → single 200-d or 512-d vector per recording → LogReg on recording-level features with subject-level GroupKFold.

**New (this run)**: keep per-window features (no pooling) → LogReg trained on window-level samples (training-fold recordings) → predict probability per test-fold window → mean-pool probabilities per recording → threshold 0.5 → recording-level BA.

**Robustness additions** (required for REVE stability):
- Per-dim 1–99 percentile clip on training fold (applied to test fold with same bounds).
- `solver='liblinear'`, `tol=1e-3`, `max_iter=5000`.
- StandardScaler post-clip.

Why the new protocol matches FT: `train_ft.py` does per-window training with prediction pooling. Prior LP pool-then-train did not. The 8 pp drop on Stress LaBraM (0.605 → 0.525) reflects this asymmetry, not a computational error.

## Full results (8 seeds each)

| Dataset | Model | Mean ± Std (ddof=1) | Range | n_rec | pos/neg |
|---|---|---|---|---|---|
| stress | labram | **0.525 ± 0.040** | [0.473, 0.589] | 70 | 14/56 |
| stress | cbramod | 0.430 ± 0.033 | [0.366, 0.473] | 70 | 14/56 |
| stress | reve | 0.441 ± 0.022 | [0.411, 0.473] | 70 | 14/56 |
| eegmat | labram | **0.736 ± 0.025** | [0.708, 0.778] | 72 | 36/36 |
| eegmat | cbramod | 0.722 ± 0.022 | [0.694, 0.764] | 72 | 36/36 |
| eegmat | reve | 0.733 ± 0.021 | [0.708, 0.764] | 72 | 36/36 |
| adftd | labram | **0.653 ± 0.018** | [0.633, 0.692] | 195 | 108/87 |
| adftd | cbramod | 0.570 ± 0.030 | [0.534, 0.614] | 195 | 108/87 |
| adftd | reve | 0.652 ± 0.027 | [0.614, 0.695] | 195 | 108/87 |
| tdbrain | labram | **0.752 ± 0.011** | [0.738, 0.765] | 734 | 640/94 |
| tdbrain | cbramod | 0.525 ± 0.022 | [0.497, 0.558] | 734 | 640/94 |
| tdbrain | reve | 0.518 ± 0.023 | [0.474, 0.546] | 734 | 640/94 |

## Key deltas vs prior draft numbers

| Claim in draft | Prior (feature-avg LP) | New (per-window LP) | Δ |
|---|---|---|---|
| §5.2 Stress LaBraM frozen LP | 0.605 ± 0.030 | **0.525 ± 0.040** | −8.0 pp |
| §5.2 "FM > XGBoost balanced by 5 pp" | holds (0.605 vs 0.553) | **reverses** (0.525 < 0.553) | narrative flip |
| §6.2 EEGMAT LaBraM frozen LP | (implicit ~0.50) | **0.736 ± 0.025** | substantial upward revision |

**Narrative implication for §5.2**: The original claim "frozen FMs contribute 5 pp over class-balanced XGBoost" is reversed by the apples-to-apples protocol. Under matched per-window training, LaBraM frozen LP (0.525) is comparable to XGBoost balanced (0.553); CBraMod (0.430) and REVE (0.441) fall below chance-corrected and below XGBoost. The new claim strengthens SDL: frozen FM features do not contain more linearly separable DASS-state signal than hand-crafted band-power, on this 70-recording cohort.

**Narrative implication for §6.2**: The EEGMAT "23 pp FT rescue" framing must be re-checked — if frozen LP is already 0.736, then FT's 0.731 is NOT a rescue but essentially equal to LP. This changes the §6 paired-comparison narrative substantially. See open question below.

## Open question for §6 paired comparison

Under the new apples-to-apples LP, EEGMAT LaBraM LP = 0.736 and EEGMAT LaBraM FT = 0.731 (from exp04 with per-FM HP). These are statistically indistinguishable. The original framing "EEGMAT is the rescue case where FT reaches 0.73 above LP baseline" no longer holds.

**Options**:
1. Reframe §6.2 as "EEGMAT LP is already strong; FT does not add significantly — contrast strength affects frozen representation quality, not FT gain".
2. Report the LP itself as the rescue evidence: "EEGMAT has strong state signal already extractable by LP" vs "Stress has no signal extractable by LP or FT".
3. Compute paired FT gain (ΔBA = FT − LP per seed) as the primary metric instead of absolute FT BA.

Recommendation: Option 2 — the paper becomes about LINEAR SEPARABILITY of state signal at frozen level, which is cleaner and doesn't depend on FT hyperparameter choice. Consistent with SDL thesis: subject-dominance limits apply to the representation, not to a specific training recipe.

## Files

- Results per dataset: `results/studies/perwindow_lp_all/{dataset}/{model}_multi_seed.json`
- Per-window features: `results/features_cache/frozen_{model}_{dataset}_perwindow.npz`
- Extraction script: `scripts/features/extract_frozen_all_perwindow.py`
- LP script: `scripts/experiments/frozen_lp_perwindow_all.py`
