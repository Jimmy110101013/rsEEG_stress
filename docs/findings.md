# Findings

Single source of truth for confirmed scientific findings.
Each finding has a stable ID (F##), status, evidence, and date metadata.
**Read this file at session start to know what is current.**

---

### F01: Subject leakage inflates EEG classification by 20+ pp
**Status**: confirmed
**Evidence**: All 3 FMs show ~21 pp BA drop from trial-level to subject-level CV (LaBraM 0.862→0.656, REVE 0.770→0.553, CBraMod 0.712→0.488 under subject-dass; gap persists under per-rec dass).
**First observed**: 2026-04-03
**Last updated**: 2026-04-10
**Notes**: Wang et al. (arXiv:2505.23042) report 90% BA on UCSD Stress using trial-level splits — our result explains their inflated number.

---

### F02: Frozen FM representations encode subject identity, not diagnostic signal
**Status**: confirmed
**Evidence**: LaBraM pooled label fraction is 2.8–7.2% across 4 datasets (`paper/figures/variance_analysis.json`). Cosine similarity within-subject >> between-subject for all 3 FMs. Subject identity dominates 71% of EEGMAT representation variance (mixed-effects ICC).
**First observed**: 2026-04-05
**Last updated**: 2026-04-12
**Notes**: fitness_metrics_full.json — RSA subject r > RSA label r in 12/12 model×dataset combinations.

---

### F03: Classical features collapse under per-rec dass; FM frozen features do not
**Status**: confirmed (updated 2026-04-13 — re-validated with correct 70-rec sample)
**Evidence**: Under per-rec dass with 70 recordings (matching FM pipeline): RF BA=0.44, LogReg BA=0.49, XGBoost BA=0.44, SVM BA=0.43 — all at or below chance. All kappas negative. Source: `results/studies/exp02_classical_dass/rerun_70rec/summary.json`.
**First observed**: 2026-04-04 (subject-dass parity); **revised** 2026-04-13
**Last updated**: 2026-04-13
**Key insight**: The previous 0.666 RF was entirely due to subject-dass OR-aggregation. Under honest per-rec labels, classical band-power features cannot discriminate stress state at all, while LaBraM frozen LP still achieves 0.605. This reverses the F03 narrative: **FMs outperform classical features under honest labels**.
**Bug found**: Original classical dass run used `--max-duration 400` (61 recs) while FMs used 70 recs. Re-run with 70 recs confirmed the collapse is real, not a sample mismatch artifact.

---

### F04: Fine-tuning produces three N-invariant modes driven by label biology (LaBraM only)
**Status**: partially superseded by F17 — LaBraM ADFTD injection direction confirmed but magnitude inflated (10s window artifact); REVE shows opposite pattern on same datasets
**Evidence**: Matched subsampling 100 draws/rung, subject-level permutation null. Source: `paper/figures/variance_analysis_matched.json`, `scripts/run_variance_analysis.py`.
**First observed**: 2026-04-08
**Last updated**: 2026-04-10

**Cross-dataset taxonomy (representation-level pooled label fraction, LaBraM):**

| Mode | Dataset | n_rec/n_subj | Frozen | FT | N-controlled Δ | Permutation null |
|---|---|---|---|---|---|---|
| **Injection** | ADFTD (AD vs HC) | 195/65 | 2.79% | 7.70% | +5.22 to +5.68 pp across N∈[17,65] | 4–8× above null |
| **Mild injection** | EEGMAT (rest vs task) | 72/36 | 5.35% | 5.82% | n/a (crossed design) | — |
| **Silent erosion** | TDBRAIN (MDD vs HC) | 734/359 | 2.97% | 1.47% | −1.06 to −1.58 pp across N∈[35,359] | Opposite side of zero from null |
| **Stale** | Stress (DASS) | 70/17 | 7.23% | 7.24% | n/a | **Computed under subject-dass — needs re-run** |

**Key insight**: At matched N=17 with ~7% frozen baselines, ADFTD gains +5 pp and Stress gains 0 pp → modes are label-biology, not sample size.

**Supersedes**: "dataset-size-dependent taxonomy" (2026-04-08 afternoon), "fine-tuning reduces subject dominance 5-6×" (naive η²), "ADFTD ×2.76 rewrite" (N-inflation artifact — honest multiplier at N=17 is 1.87×).

---

### F05: Erosion on Stress is LaBraM-specific, not model-universal
**Status**: confirmed
**Evidence**: Multi-model HP sweep, 54 runs (3 LRs × 2 encoder LR scales × 3 seeds × 3 models). Source: `results/hp_sweep/20260410_dass/`.
**First observed**: 2026-04-11
**Last updated**: 2026-04-11

**Behavioral-level (subject-level 5-fold BA, per-recording DASS):**

| Model | Frozen LP (8-seed) | Best FT (3-seed) | Best FT config | Δ | Mode |
|---|---|---|---|---|---|
| **LaBraM** | **0.605 ± 0.030** | 0.524 ± 0.008 | lr=1e-4, elrs=1.0 | **−8.1 pp** | erosion |
| **CBraMod** | 0.452 ± 0.030 | **0.548 ± 0.026** | lr=1e-5, elrs=0.1 | **+8.3 pp** | injection |
| **REVE** | 0.494 ± 0.017 | **0.577 ± 0.041** | lr=3e-5, elrs=0.1 | **+8.0 pp** | injection |

LaBraM canonical recipe (lr=1e-5, elrs=0.1): 0.443 ± 0.068 (−16.2 pp gap).

**Key insight**: Erosion requires the frozen representation to already be strong. LaBraM frozen 0.605 >> CBraMod 0.452 / REVE 0.494.

**Supersedes**: "erosion is model-universal" hypothesis.

---

### F06: LaBraM FT on Stress is indistinguishable from permutation null
**Status**: confirmed (canonical recipe only; best-HP not yet null-tested)
**Evidence**: Real FT 0.443 ± 0.068 (3 seeds) vs null FT 0.497 ± 0.081 (10 perm), one-sided p=0.70. Two null seeds (s4=0.607, s8=0.643) beat all real-label seeds. Source: `results/studies/exp03_stress_erosion/analysis.json`.
**First observed**: 2026-04-10
**Last updated**: 2026-04-10

---

### F07: subject-dass OR-aggregation is a trait-memorization artifact
**Status**: confirmed
**Evidence**: Only 3/17 subjects have within-subject DASS class transitions. Subject-dass turns per-recording stress classification into subject-identity broadcast. Canonical 0.656 LaBraM FT under subject-dass is a non-reproducible single-seed draw (HEAD re-run gives 0.45). Source: `docs/progress.md` §4.6.
**First observed**: 2026-04-09
**Last updated**: 2026-04-10
**Action**: `--label subject-dass` deprecated. All new work uses `--label dass` (per-recording).

---

### F08: cuDNN non-determinism produces 10–20 pp single-seed swings on Stress
**Status**: confirmed
**Evidence**: Exact same canonical recipe (subject-dass, seed=42) at HEAD gives 0.4505 vs original 0.6559 — 20.5 pp swing. Root cause: 70-rec / 14-positive regime amplifies microscopic init differences. Mitigated (not eliminated) by `cudnn.deterministic=True` (added 2026-04-10).
**First observed**: 2026-04-10
**Last updated**: 2026-04-10
**Action**: All Stress BA claims require multi-seed (≥3). Single-seed BAs are not reproducible.

---

### F09: EEGMAT within-subject labels do NOT rescue FT rewriting
**Status**: confirmed
**Evidence**: EEGMAT pooled label fraction 5.35%→5.82% (within bootstrap noise) despite paired rest/task per subject. Mixed-effects ICC for subject increases 0.56→0.63 under FT. BA=0.736 achieved via projection, not representation rewrite.
**First observed**: 2026-04-08
**Last updated**: 2026-04-10
**Notes**: Falsifies "label-structure alone explains FM ceiling" hypothesis. EEGMAT LaBraM FT 3-seed: 0.731 ± 0.017 (`results/studies/exp04_eegmat_feat_multiseed/`).

---

### F10: Alpha lateralization is the dominant stress feature in classical ML
**Status**: confirmed
**Evidence**: RF feature importance — 13/20 top features are alpha band. Right-hemisphere alpha (T4, C4, CP4, P4) + frontal (Fp2, F8) most discriminative. Source: `docs/fm_comparison_summary.md` §6.
**First observed**: 2026-04-04
**Last updated**: 2026-04-04

---

### F11: Three independent FM benchmarks confirm our findings
**Status**: confirmed
**Evidence**: Brain4FMs (2026), EEG-FM-Bench (Xiong 2025), AdaBrain-Bench (2025) all report FM failure on cross-subject affective/cognitive tasks. Our contribution is the mechanistic explanation (variance decomposition) they lack. Source: `docs/related_work.md` §2.
**First observed**: 2026-04-07
**Last updated**: 2026-04-08

---

### F12: Subject-adversarial and LEAD-style losses hurt, not help
**Status**: confirmed (ruled out)
**Evidence**: GRL λ=0.1 → 0.537 BA (−12 pp), λ=1.0 → 0.558 BA (−10 pp), LEAD loss → 0.439 BA. Full λ sweep [0.01–1.0] all worse than baseline. Source: `docs/progress.md` §6.
**First observed**: 2026-04-05
**Last updated**: 2026-04-05

---

### F13: FM-task fitness metrics show subject identity dominates across all models
**Status**: confirmed
**Evidence**: RSA subject r > RSA label r in 12/12 frozen model×dataset combinations. Silhouette, Fisher, kNN, LogME, H-score all computed. Source: `results/studies/exp06_fm_task_fitness/fitness_metrics_full.json`.
**First observed**: 2026-04-12
**Last updated**: 2026-04-12

---

### F17: FT mode is a model × dataset interaction, not label-biology alone
**Status**: confirmed (3-seed validated, 2026-04-13)
**Evidence**: 3 models × 2 datasets × 3 seeds = 18 FT runs + variance decomposition. All frozen-FT pairs matched at 5s windows. Sources: `results/studies/exp07_adftd_multiseed/`, `results/studies/exp08_tdbrain_multiseed/`, `results/features_cache/ft_*/`.
**First observed**: 2026-04-13
**Last updated**: 2026-04-13

**3-seed pooled label fraction Δ (FT − Frozen, pp):**

| Model | ADFTD Δ (mean ± std) | Stable? | TDBRAIN Δ (mean ± std) | Stable? |
|---|---|---|---|---|
| **LaBraM** | **+1.03 ± 0.61** | ✅ direction stable | **−1.56 ± 0.23** | ✅ very stable |
| **CBraMod** | **+0.83 ± 2.73** | ❌ unstable | **−0.02 ± 0.03** | ✅ stable (≈0) |
| **REVE** | **−1.53 ± 0.23** | ✅ very stable | **+0.44 ± 0.26** | ✅ direction stable |

Per-seed detail:

| Run | s42 | s123 | s2024 |
|---|---|---|---|
| LaBraM×ADFTD | +0.17 (BA .703) | +1.44 (BA .724) | +1.47 (BA .699) |
| LaBraM×TDBRAIN | −1.24 (BA .690) | −1.64 (BA .690) | −1.79 (BA .613) |
| CBraMod×ADFTD | −0.88 (BA .509) | +4.68 (BA .562) | −1.31 (BA .540) |
| CBraMod×TDBRAIN | −0.06 (BA .498) | −0.02 (BA .472) | +0.02 (BA .494) |
| REVE×ADFTD | −1.22 (BA .662) | −1.76 (BA .685) | −1.61 (BA .626) |
| REVE×TDBRAIN | +0.33 (BA .476) | +0.18 (BA .510) | +0.80 (BA .479) |

**Key findings:**

1. **LaBraM and REVE show opposite patterns on both datasets.** LaBraM: ADFTD injection, TDBRAIN erosion. REVE: ADFTD erosion, TDBRAIN mild injection. Same labels, opposite FT effects → **model architecture determines FT direction, not label biology alone.**

2. **CBraMod is FT-insensitive on TDBRAIN** (Δ ≈ 0, std 0.03pp) **but wildly unstable on ADFTD** (std 2.73pp with one seed hitting +4.68pp). Its criss-cross attention may interact differently with ADFTD's spectral structure.

3. **Seed sensitivity is model-dependent.** LaBraM and REVE have small std (0.23–0.61pp); CBraMod×ADFTD has std 2.73pp. Variance decomposition robustness depends on the FM architecture.

4. **F04's +5pp ADFTD injection is not reproducible at 5s windows.** The 3-seed LaBraM mean is +1.03pp. The original +5pp came from a 10s window run (different experimental condition). The injection direction is confirmed but the magnitude was inflated by 5×.

**Supersedes**: F04 cross-dataset taxonomy as the primary multi-model result. F04 remains valid as LaBraM-only, 10s-window historical evidence.

**Window note**: All results use 5s windows (per LaBraM/CBraMod extractor config `"per stress paper"`). F04 canonical used 10s (dataset loader default, no paper basis). The 5s setting is the principled choice.

---

### F14: Within-subject longitudinal DSS classification fails for all 3 FMs
**Status**: confirmed (negative result)
**Evidence**: LOO within-subject evaluation using personal median DSS as threshold. All methods (centroid, 1-NN, logistic regression) at or below chance across all 3 FMs. Source: `results/studies/exp11_longitudinal_dss/`.
**First observed**: 2026-04-13
**Last updated**: 2026-04-13

| Model | Centroid BA | 1-NN BA | Linear BA | n evaluable recordings |
|---|---|---|---|---|
| LaBraM | 0.296 | 0.296 | 0.000 | 54 |
| CBraMod | 0.241 | 0.167 | 0.000 | 54 |
| REVE | 0.333 | 0.426 | 0.000 | 54 |

**Key insight**: Frozen FM representations do not capture within-subject stress variation. The "within-subject longitudinal reframing" hypothesis (use DSS trajectory instead of cross-subject DASS) does **not** rescue classification. All kappas are negative. This rules out the highest-leverage open experiment from the paper strategy.

---

### F15: Statistical hardening — bootstrap CIs and effect sizes for erosion/injection
**Status**: confirmed
**Evidence**: Bootstrap (10k resamples) on per-seed BA values. Source: `results/studies/exp10_stat_hardening/stat_hardening.json`.
**First observed**: 2026-04-13
**Last updated**: 2026-04-13

**Bootstrap 95% CIs:**

| Model | Frozen LP [95% CI] | Best FT [95% CI] | Cohen's d |
|---|---|---|---|
| LaBraM | 0.605 [0.585, 0.626] | 0.524 [0.518, 0.536] | **+4.30** (erosion) |
| CBraMod | 0.452 [0.430, 0.472] | 0.548 [0.518, 0.580] | **−3.28** (injection) |
| REVE | 0.494 [0.482, 0.506] | 0.577 [0.536, 0.634] | **−2.65** (injection) |

**CIs do not overlap** for LaBraM (frozen > FT) and CBraMod (FT > frozen). REVE CIs have minimal overlap. All Cohen's d > 2.5 — very large effects.

**Paired permutation test** (3 matched seeds): all models show p=0.25 — expected with only 3 paired observations (minimum possible p with n=3 is 0.125). Direction is consistent.

**Sign test on trial-vs-subject gap**: 3/3 FMs show trial > subject (mean gap +21.6 pp), p=0.125 (exact binomial). Wilcoxon W=6.0, p=0.125.

---

### F16: Classical RF subject-dass 0.666 was entirely an OR-aggregation artifact
**Status**: confirmed
**Evidence**: Re-run with 70-rec per-rec dass (matching FM pipeline): RF BA=0.44 (was 0.666 under subject-dass). All 5 classical methods below chance. Source: `results/studies/exp02_classical_dass/rerun_70rec/summary.json`.
**First observed**: 2026-04-13
**Last updated**: 2026-04-13
**Notes**: Strengthens the trait-memorization narrative (F07). Under honest per-rec labels, only FM frozen features retain signal (LaBraM 0.605). Classical band-power features that "matched" FMs were actually reading subject identity, not stress state.

---

### F18: REVE injection magnitude is window-dependent
**Status**: confirmed (2026-04-14)
**Evidence**: REVE frozen LP and FT measured at both 5s and 10s windows on Stress. Source: `results/studies/exp16_reve_window_match/`.

| Window | Frozen LP (3-seed) | FT (3-seed) | Δ(FT−Frozen) |
|---|---|---|---|
| **5s-matched** | 0.4970 ± 0.0111 | 0.5446 ± 0.0635 | **+4.8 pp** |
| **10s-matched** | 0.4494 ± 0.0111 | 0.5774 ± 0.0414 | **+12.8 pp** |
| Original (MISMATCH) | 0.494 (5s) | 0.577 (10s) | +8.3 pp |

**Key insight**: F05 reported REVE injection of +8.3pp, but the frozen baseline (5s) and FT (10s) used different windows. Window-matched comparisons show injection IS real but magnitude depends on window. 5s-matched gives the cleanest comparison; 10s gap is inflated because 10s FT has 2× fewer training windows.

**Supersedes obs 1702 claim "5s/10s frozen LP identical"** — frozen LP differs by ~4.8pp between 5s and 10s, and features are NOT bit-identical (max abs diff 22.5).

**Action**: F05 REVE row should be split into 5s-matched (+4.8pp) and 10s-matched (+12.8pp). Cite 5s-matched as the principled comparison.

---

### F19: CBraMod/REVE permutation null borderline on Stress
**Status**: confirmed (2026-04-14)
**Evidence**: 10 permutation-null runs each for CBraMod and REVE at best HP config. Source: `results/studies/exp03_stress_erosion/ft_null_{cbramod,reve}/`.

| Model | Real FT (3-seed) | Null FT (10-perm) | Δ | p (one-sided) |
|---|---|---|---|---|
| CBraMod | 0.5476 ± 0.0256 | 0.4839 ± 0.0472 | +6.4pp | **0.100** |
| REVE | 0.5774 ± 0.0414 | 0.4857 ± 0.0656 | +9.2pp | **0.100** |

**Key insight**: With only 10 perms, minimum non-zero p-value is 0.1 — both CBraMod and REVE injection are at this floor (1/10 null seeds beat real mean). Weak but not absent evidence. LaBraM null already confirmed indistinguishable (F06, p=0.70).

**Action**: Consider extending to 20-perm for tighter bound if paper reviewers push. Current evidence is consistent with weak real injection or noise.

---

### F20: ShallowConvNet from scratch matches FM ceiling on Stress
**Status**: confirmed (2026-04-14)
**Evidence**: Multi-seed (3 seeds) non-FM baselines with best LR per model. Source: `results/studies/exp15_nonfm_baselines/sweep/`.

| Model | Best LR | 3-seed BA | Notes |
|---|---|---|---|
| ShallowConvNet | 1e-4 | **0.557 ± 0.026** | Matches FM range |
| EEGNet | 5e-4 | 0.518 ± 0.079 | Below FM range, high variance |
| LaBraM FT (F05) | — | 0.524 ± 0.008 | |
| CBraMod FT (F05) | — | 0.548 ± 0.026 | |
| REVE FT (F05) | — | 0.577 ± 0.041 | |

**Key insight**: ShallowConvNet trained from scratch on 70 recordings reaches 0.557, within the FM FT range (0.524-0.577). FM pretraining does NOT provide task-specific advantage on Stress. Subject-dominated Stress is a **task property**, not an FM property.

**Caveats**:
- EEGNet shows high variance (0.41-0.60 across seeds), consistent with cuDNN non-determinism (F08) on 70-rec regime.
- Only single LR sweep done at seed=42; other LRs might give better mean.

**Implication**: The paper's FM-vs-classical story should be extended — Stress is a task where even 2017-vintage CNNs reach FM performance. This strengthens the "task-property, not model-property" narrative.

---

## Deprecated numbers — do NOT cite

| Number | Why wrong | Replacement |
|---|---|---|
| LaBraM FT subject-dass 0.656 | Single-seed, cuDNN noise, OR-aggregation | Multi-seed per-rec dass: 0.443 ± 0.068 (canonical) or 0.524 ± 0.008 (best HP) |
| Classical RF subject-dass 0.666 | OR-aggregation artifact (F16) | Per-rec dass: RF=0.44, all classical below chance |
| Trial-level LaBraM 0.862 | Subject leakage, OR-aggregation | Not comparable to per-rec subject-level CV |
| "34× subject/label η² ratio" | Naive one-way η² with nesting confound | Pooled label fraction (2.8–7.2%) |
| "ADFTD ×2.76 rewrite" | N-inflation artifact | +5 pp additive (N-invariant) |
| "9.1→1.6 per-fold Stress drop" | Per-fold degenerate (1 subject per positive class) | Do not cite per-fold Stress |
| "Erosion is model-universal" | Refuted by HP sweep | LaBraM-specific; CBraMod/REVE show injection |
| Stress variance row (7.23%→7.24%) | Computed under subject-dass | Needs re-run with `--label dass --save-features` |
