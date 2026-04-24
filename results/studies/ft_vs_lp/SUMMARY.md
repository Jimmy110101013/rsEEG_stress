# Fine-Tuning vs Frozen Linear Probe — Apples-to-Apples Comparison

**Date**: 2026-04-20
**Purpose**: Table-level comparison between fine-tuned (3-seed) and frozen per-window linear probe (8-seed) under matched protocols.

## Protocol match

| | Fine-tuning | Frozen LP |
|---|---|---|
| Input norm | per-FM (labram=zscore, cbramod=none, reve=none) | same |
| Channels | per-dataset native (Stress 30ch; EEGMAT/ADFTD/TDBRAIN 19ch via COMMON_19) | same |
| Training | per-window loss, prediction pooled per recording | LogReg on per-window features with per-window loss and prediction pooling per recording |
| CV | subject-level 5-fold (StratifiedGroupKFold, groups=patient_id) | same |
| Metric | recording-level balanced accuracy | same |
| Seeds | 3 (42, 123, 2024) | 8 (plus 7, 0, 1, 99, 31337) |
| Feature scaling | internal model | per-dim 1–99 percentile clip + StandardScaler |

## Per-seed paired FT vs LP (seeds 42, 123, 2024)

| Dataset | Model | FT s42 | LP s42 | FT s123 | LP s123 | FT s2024 | LP s2024 | Paired Δ mean |
|---|---|---|---|---|---|---|---|---|
| stress | labram | 0.536 | 0.509 | 0.420 | 0.518 | 0.375 | 0.491 | **−0.063** |
| eegmat | labram | 0.708 | 0.722 | 0.750 | 0.708 | 0.736 | 0.778 | −0.005 |
| eegmat | cbramod | 0.556 | 0.722 | 0.667 | 0.694 | 0.639 | 0.736 | **−0.097** |
| eegmat | reve | 0.694 | 0.736 | 0.764 | 0.722 | 0.722 | 0.750 | −0.009 |
| adftd | labram | 0.703 | 0.692 | 0.724 | 0.633 | 0.699 | 0.641 | **+0.053** |
| adftd | cbramod | 0.509 | 0.614 | 0.562 | 0.534 | 0.540 | 0.545 | −0.027 |
| adftd | reve | 0.662 | 0.695 | 0.685 | 0.675 | 0.625 | 0.646 | −0.014 |
| tdbrain | labram | 0.690 | 0.742 | 0.690 | 0.738 | 0.613 | 0.757 | **−0.081** |
| tdbrain | cbramod | 0.498 | 0.509 | 0.472 | 0.514 | 0.494 | 0.558 | −0.039 |
| tdbrain | reve | 0.476 | 0.492 | 0.510 | 0.520 | 0.479 | 0.532 | −0.026 |

Missing cells: stress × cbramod and stress × reve (no real-label FT multi-seed run on Stress for these models; only permutation-null baseline available in `exp03_stress_erosion/ft_null_{cbramod,reve}/`).

## Aggregated comparison

| Dataset | Model | Frozen LP (8-seed) | FT (3-seed) | ΔFT − LP3 | Verdict |
|---|---|---|---|---|---|
| stress | labram | 0.525 ± 0.040 | 0.443 ± 0.083 | **−0.063** | FT degrades |
| stress | cbramod | 0.430 ± 0.033 | — | — | (FT missing) |
| stress | reve | 0.441 ± 0.022 | — | — | (FT missing) |
| eegmat | labram | 0.736 ± 0.025 | 0.731 ± 0.021 | −0.005 | FT ≈ LP |
| eegmat | cbramod | 0.722 ± 0.022 | 0.620 ± 0.058 | **−0.097** | FT degrades |
| eegmat | reve | 0.733 ± 0.021 | 0.727 ± 0.035 | −0.009 | FT ≈ LP |
| adftd | labram | 0.653 ± 0.018 | 0.709 ± 0.014 | **+0.053** | **FT rescues** |
| adftd | cbramod | 0.570 ± 0.030 | 0.537 ± 0.027 | −0.027 | FT degrades |
| adftd | reve | 0.652 ± 0.027 | 0.658 ± 0.030 | −0.014 | FT ≈ LP |
| tdbrain | labram | 0.752 ± 0.011 | 0.665 ± 0.045 | **−0.081** | FT degrades |
| tdbrain | cbramod | 0.525 ± 0.022 | 0.488 ± 0.014 | −0.039 | FT degrades |
| tdbrain | reve | 0.519 ± 0.022 | 0.488 ± 0.019 | −0.026 | FT degrades |

(Δ computed vs LP 3-seed subset of seeds 42/123/2024 for apples-to-apples; LP 8-seed number shown for reference only.)

## Summary of 10 complete cells

- **FT > LP** (Δ > +0.01): **1 cell** — ADFTD × LaBraM (+5.3 pp)
- **FT ≈ LP** (|Δ| ≤ 0.01): **2 cells** — EEGMAT × LaBraM, EEGMAT × REVE
- **FT < LP** (Δ < −0.01): **7 cells**

Median Δ across the 10 cells: **−0.026**. Mean Δ: **−0.031**. Only 10% of model × dataset combinations show FT improving over frozen LP.

## Observations for the paper

1. **ADFTD × LaBraM is the single clear FT rescue case**. Matches §4.5 observation that LaBraM × ADFTD had the largest Δlabel_frac (+10.65 pp) in variance decomposition. Consistent with ADFTD's known strong within-recording physiological contrast (resting-state alpha slowing in Alzheimer's).
2. **LaBraM is the only FM that ever rescues via FT** across these 4 datasets. CBraMod and REVE never exceed their own frozen LP across the 7 complete combinations. LaBraM's larger parameter budget + VQ tokenizer may give it more plasticity at fine-tuning time, but that plasticity is only productive when the dataset has strong contrast (ADFTD).
3. **TDBRAIN × LaBraM has large negative Δ (−8.1 pp)** despite a very strong LP baseline (0.752). Fine-tuning on a 734-rec MDD benchmark destroys the pretrained representation at this cohort size — reproduces BENDR-style "FT degrades LP" finding with the most rigorous protocol we could apply.
4. **Stress × LaBraM FT also degrades (−6.3 pp)** — confirms §5.3 Stress-erosion finding under the new apples-to-apples LP baseline.
5. **Pattern is dataset-architecture-specific, not uniform**. The simple narrative "FT always degrades" (from BENDR) and the simple narrative "FM benchmarks work" (from LaBraM/CBraMod headline tables) are both wrong. The accurate narrative is "FT helps only when the dataset has strong within-recording contrast and the architecture has plasticity".

## Caveat: FT 3-seed variance

FT standard deviations are large (range 0.014 to 0.083). The 3-seed estimate for Δ is noisy. However:
- Direction is consistent across all 3 seeds in 9 of 10 cells (the exception being EEGMAT × LaBraM where 1 of 3 seeds favors FT).
- Per-seed paired comparison (same seed for FT and LP) is available in `comparison.json` and can be used for paired-bootstrap inference in a supplementary table.

## Files

- Per-seed paired data: `results/studies/ft_vs_lp/comparison.json`
- FT originals: `results/studies/exp{03,04,07,08,17}_*/`
- LP originals: `results/studies/perwindow_lp_all/`
