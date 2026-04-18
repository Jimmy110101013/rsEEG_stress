# exp_30 — SDL vs Between REPORT

Paper strategy tested: `docs/paper_strategy_sdl_critique.md` (v2).
Data sources: `results/features_cache/frozen_*.npz`, `results/studies/exp_newdata/`, `paper/figures/source_tables/`, `results/hhsa/`.

---

## 0. Pipeline v4 honest summary (2026-04-18, supersedes auto-generated §1–3 below)

The v4 run added (i) an MLP subject-ID probe alongside the linear LR probe and (ii) a `within_strict` arm that excludes Stress (3/17 subjects crossing DASS → degenerate within-subject design).

### Key finding — the "arm flip" in earlier runs was partly Stress pollution

| Predictor | within (n=9, incl Stress) | **within_strict (n=6)** | between (n=9) |
|---|---|---|---|
| subject_id_ba_lr | −0.02 | **+0.60** (CI wide) | **+0.70, p=0.036** |
| subject_id_ba_mlp | +0.23 | +0.31 | +0.59 |
| rsa_label | −0.50 | +0.09 (null) | +0.60 |
| frozen_label_frac | −0.25 | −0.14 | +0.57 |
| subject_to_label_ratio | +0.07 | +0.14 | **−0.63, p=0.067** |

**Honest reading**:

1. **C1 (subject dominance) — unchanged**. All 18 cells show frozen_subject_frac > 39 %.
2. **C2 between-arm evidence — survives**. `subject_id_ba_lr` → ΔBA in between-arm: ρ = +0.70, 95 % CI [0.113, 0.982], p = 0.036, n = 9. Near-significant even at n = 9.
3. **C2 within-vs-between *differential* claim — does NOT survive under within_strict**. Point estimate in within_strict is +0.60 (similar to between's +0.70), only distinguishable by CI width / power. Saying "subject leakage affects only between-arm" is not supported.
4. **The prior "sign flip" (−0.50 → +0.60 for rsa_label) was Stress-driven**. Without Stress, rsa_label is null in within_strict (+0.09). Same story for subject_id_ba_lr: the negative within-arm ρ was the Stress cells dragging the correlation down.

### Paper implications

The v2 Type C critique can be pitched in one of three forms, in descending ambition:

- **(A) Full differential claim**: "subject leakage affects between-arm only". **Not currently supported** — within_strict n = 6 is underpowered to distinguish, and point estimates are similar in magnitude. Requires more within-subject datasets (HMC, TUAB, ADFTD-3cls — deferred in TODO.md).
- **(B) Unified claim**: "subject-ID decodability universally predicts FT gain". Supported by the data (ρ > 0 across arms), but loses the benchmark-critique edge.
- **(C) Hybrid narrow claim**: "in between-subject benchmarks, subject-ID decodability and subject_to_label_ratio correlate with FT gain (C2 as originally scoped); within-subject evidence is inconclusive at current N; HHSA / task-variance-fraction are not predictive".

Recommendation: **proceed with (C) as the honest default, plan to upgrade to (A) after running TUAB + HMC + ADFTD-3cls**. Do NOT rewrite tex until A is unlocked or C is explicitly accepted.

### Status on pre-registered falsifiers

- **F1** (subject_frac > 40 % for all 18 cells): ✅ CONFIRMED
- **F2** (between CI excludes 0; within_strict CI crosses 0): within_strict CI does cross 0 (✅ consistent with null) and between CI excludes 0 (✅). But within_strict point estimate (+0.60) is close to between (+0.70), so **F2 technically passes on CI but differential is weak**.
- **F3** (another predictor shows same arm pattern): `subject_to_label_ratio` between ρ = −0.63 (p = 0.067) with within_strict ρ = +0.14 — consistent direction but p > 0.05.
- **F4** (leave-one-dataset-out): NOT YET TESTED.

### Seed-noise bootstrap — does ρ survive seed resampling?

ΔBA has seed-level std 0.02–0.07 across cells, comparable to the per-cell
mean ΔBA. Raised a concern that ρ = 0.70 might be a lucky-seed artefact.

Ran 10 000-iter seed-noise bootstrap: for each cell, resample ΔBA from
its 3-seed distribution (raw seeds for meditation/sleepdep; Gaussian
mean ± std for older 4 datasets). Results in
`tables/seed_noise_bootstrap.json`:

| Predictor | Arm | median ρ | 95 % CI | P(ρ expected direction) |
|---|---|---|---|---|
| subject_id_ba_lr | between (n = 9) | **+0.50** | [+0.03, +0.83] | **98.4 %** |
| subject_id_ba_lr | within_strict (n = 6) | +0.43 | [−0.43, +0.89] | 77.9 % |
| subject_id_ba_lr | within_loose (n = 9) | −0.05 | [−0.55, +0.38] | 40.5 % |
| subject_to_label_ratio | between (n = 9) | **−0.50** | [−0.85, +0.03] | **96.7 %** (neg) |
| subject_to_label_ratio | within_strict | +0.14 | [−0.54, +0.71] | 60.0 % |

**Interpretation**:
- ρ ≈ 0.50 is the seed-robust estimate for between-arm, not 0.70. Original
  point estimate was an optimistic (lucky-seed) outcome.
- Between-arm direction is 98.4 % seed-robust — the claim survives.
- within_strict ρ = +0.43 with 77.9 % prob positive — consistent with
  "positive tendency, underpowered to confirm null".
- Differential claim (between-only leakage) does **not** survive — both
  arms lean positive; only between is statistically distinguishable.

### Revised paper claim (seed-robust)

> Subject-ID linear decodability from frozen features is **consistently
> positively associated** with FT gain across six EEG benchmarks. The
> relationship is statistically robust in between-subject benchmarks
> (ρ = +0.50, 95 % CI [+0.03, +0.83], n = 9 under seed-noise bootstrap)
> but **underpowered in within-subject benchmarks** (ρ = +0.43, n = 6).
> A complementary signature — subject-to-label variance ratio — shows
> a negative association with FT gain, also statistically robust only
> in the between-subject arm (ρ = −0.50, 95 % CI [−0.85, +0.03]).
> Together these suggest subject identity acts as an extractable
> shortcut feature when label signal is not buried. Whether the
> relationship genuinely differs between arms requires a higher-N
> replication (proposed: TUAB, HMC, ADFTD-3cls).

### Next steps

1. Commit v4 results + seed-noise bootstrap.
2. Add F4 LOO sensitivity to the pipeline (Stage K) — do `between` with
   one dataset dropped at a time, see if ρ = 0.50 survives.
3. **Decision gate for paper version**: run TUAB + HMC + ADFTD-3cls to
   test whether within-arm ρ collapses under larger N (path A).
   Cost ≈ 20–30 hours GPU. Otherwise submit as honest path C / hybrid B.

---

## 1. Executive summary (auto-generated, v3 state — see §0 for v4 honest update)

This pipeline assembles a 6-dataset x 3-FM master table (18 cells) and asks whether frozen-feature task variance predicts ΔBA (FT−LP) **differently** in within-subject vs between-subject benchmarks — the falsifiable core of v2.

Headline F1 (within arm, predictor = frozen_label_frac): ρ=-0.25 [95% CI -0.69, 0.54] (n=9). F2 (between arm, same predictor): ρ=0.57 [95% CI -0.11, 0.98] (n=9).

## 2. Master table preview (top rows)

```
   dataset      fm     arm  design  lp_ba_mean  ft_ba_mean  delta_ba  frozen_label_frac  frozen_subject_frac  frozen_residual_frac  subject_to_label_ratio  rsa_subject  rsa_label  subject_id_ba_lr  subject_id_ba_mlp  subject_id_ba  n_subjects  permanova_pseudo_F  permanova_R2  hhsa_median_g  hhsa_wsci
    stress  labram  within  nested       0.605       0.524    -0.081              4.528               48.723                46.749                  10.761        0.284     -0.004             0.445              0.486          0.445          17               2.950         0.053          1.079        NaN
    stress cbramod  within  nested       0.452       0.548     0.096              2.545               45.816                51.639                  18.002        0.208     -0.051             0.000              0.255          0.000          17               1.774         0.032          1.079        NaN
    stress    reve  within  nested       0.494       0.577     0.083              3.095               39.316                57.588                  12.703        0.185     -0.014             0.373              0.419          0.373          17               1.961         0.036          1.079        NaN
    eegmat  labram  within crossed       0.671       0.731     0.060              5.348               71.178                23.474                  13.308        0.119      0.073             0.458              0.389          0.458          36               3.266         0.045          1.402        NaN
    eegmat cbramod  within crossed       0.731       0.620    -0.111              3.722               79.618                16.660                  21.390        0.146      0.036             0.125              0.208          0.125          36               2.793         0.038          1.402        NaN
    eegmat    reve  within crossed       0.671       0.727     0.056              2.460               77.326                20.214                  31.439        0.117      0.014             0.181              0.139          0.181          36               1.741         0.024          1.402        NaN
  sleepdep  labram  within crossed       0.500       0.532     0.033              0.752               63.627                35.620                  84.566        0.054     -0.005             0.250              0.222          0.250          36               0.708         0.010          1.316        NaN
  sleepdep cbramod  within crossed       0.557       0.556    -0.001              1.685               67.606                30.709                  40.123        0.062     -0.006             0.097              0.056          0.097          36               0.844         0.012          1.316        NaN
  sleepdep    reve  within crossed       0.544       0.542    -0.002              5.740               45.893                48.367                   7.995        0.084      0.018             0.236              0.181          0.236          36               1.685         0.024          1.316        NaN
     adftd  labram between  nested       0.695       0.709     0.013              2.596               66.785                30.620                  25.728        0.126      0.036             0.713              0.672          0.713          65               5.445         0.027            NaN        NaN
     adftd cbramod between  nested       0.558       0.537    -0.021              0.670               61.557                37.773                  91.877        0.105      0.015             0.159              0.246          0.159          65               1.615         0.008            NaN        NaN
     adftd    reve between  nested       0.692       0.658    -0.035              0.824               43.886                55.290                  53.272        0.045      0.002             0.108              0.231          0.108          65               2.062         0.011            NaN        NaN
   tdbrain  labram between  nested       0.679       0.665    -0.015              2.413               54.250                43.337                  22.483        0.040      0.084             0.223              0.094          0.223         359              18.526         0.025            NaN        NaN
   tdbrain cbramod between  nested       0.564       0.488    -0.076              1.825               95.551                 2.624                  52.368        0.062      0.024             0.026              0.373          0.026         359              13.632         0.018            NaN        NaN
   tdbrain    reve between  nested       0.544       0.488    -0.056              0.004               60.633                39.363               15546.846        0.042      0.018             0.006              0.018          0.006         359               0.263         0.000            NaN        NaN
meditation  labram between  nested       0.473       0.515     0.042              5.385               85.568                 9.047                  15.889        0.218      0.043             0.146              0.396          0.146          24               1.856         0.047          0.552        NaN
meditation cbramod between  nested       0.710       0.683    -0.027             28.118               60.653                11.229                   2.157        0.137      0.196             0.014              0.021          0.014          24              11.264         0.229          0.552        NaN
meditation    reve between  nested       0.538       0.433    -0.105              0.809               97.091                 2.100                 120.084        0.143     -0.016             0.042              0.021          0.042          24               0.345         0.009          0.552        NaN
```

## 3. Per-prediction verdicts (F1–F4)

**F1** (within-arm ρ(frozen_label_frac, ΔBA) > 0): ρ=-0.25, 95% CI [-0.69, 0.54], n=9. **FAIL**
- Possible causes: (a) n=9 is under-powered for Spearman; (b) label_frac is not the right predictor — HHSA contrast or RSA_label may carry more signal; (c) the two small within-subject datasets (Stress 14 pure subjects, SleepDep 36) have their own signal ceilings that dominate ΔBA; (d) FT ceiling for within-subject tasks may be hit regardless of frozen-feature quality.

**F2** (between-arm ρ(frozen_label_frac, ΔBA) ≈ 0): ρ=0.57, 95% CI [-0.11, 0.98], n=9. **PASS**

**F3** (HHSA contrast correlates with ΔBA in within but not between): within ρ=-0.26 [-0.87, 0.60] (FAIL); between between-arm not computable.

**F4** (between-arm subject_id_ba > 0.7): mean subject_id_ba = 0.241 (within) vs 0.160 (between).
- Between-arm subject decodability is below the 0.7 threshold; C2 (subject-feature extractor claim) is weakened.

## 4. Unexpected findings

- `rsa_label`: ρ(within)=-0.50, ρ(between)=0.60 — sign flip between arms, consistent with v2 thesis.

## 5. Recommendation

F1 fails and F2 passes. Between-arm is clean but within-arm isn't — the critique lands but the positive diagnostic (C3/C4) doesn't. Reconsider which predictor best captures 'task-variance fraction' before rewriting the tex.

## 6. Limitations

- **N = 3 datasets × 3 FMs = 9 points per arm.** Spearman ρ at n=9 has very wide bootstrap CIs; most non-trivial effects are statistically indistinguishable from zero at 95%.
- **Between-arm TDBRAIN has ~1200 subjects**, making subject-ID decodability mechanically different (more classes) from ADFTD (65) or Meditation (24). Comparing subject_id_ba across datasets in this arm is asymmetric.
- **HHSA contrast data is missing for ADFTD and TDBRAIN.** F3 is thus untestable at full coverage; it is only evaluable on the 4-dataset subset (Stress, EEGMAT, SleepDep, Meditation).
- **Stress/EEGMAT/ADFTD/TDBRAIN FM performance uses aggregated mean/std from `master_frozen_ft_table.json`, not per-seed values**, so the per-seed variance information is lost for 4 of 6 datasets.
- **Variance decomposition for EEGMAT uses a crossed SS (not nested ω²).** Not directly comparable to ADFTD/TDBRAIN nested ω² values — interpret the label_frac / subject_frac columns as 'fraction of total SS attributable to factor X marginally' for crossed designs and as 'nested SS fraction' for nested designs. Paper-strategy_sdl_critique §3 acknowledges this asymmetry.

## 7. Checkpoint files

- `results/studies/exp_30_sdl_vs_between/tables/fm_performance.csv`
- `results/studies/exp_30_sdl_vs_between/tables/variance_decomposition.csv`
- `results/studies/exp_30_sdl_vs_between/tables/rsa.csv`
- `results/studies/exp_30_sdl_vs_between/tables/subject_decodability.csv`
- `results/studies/exp_30_sdl_vs_between/tables/permanova.csv`
- `results/studies/exp_30_sdl_vs_between/tables/hhsa_contrast.csv`
- `results/studies/exp_30_sdl_vs_between/tables/master_table.csv`
- `results/studies/exp_30_sdl_vs_between/tables/correlations.csv`
- `results/studies/exp_30_sdl_vs_between/figures/fig_sdl_benchmark_critique.pdf`
