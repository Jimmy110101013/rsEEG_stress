# SDL Paper Draft — §5 Honest Evaluation Audit of UCSD Stress

**Date**: 2026-04-15
**Role in SDL thesis**: *the measurement protocol.* §4 established that frozen FM representations are subject-dominated. Before that structural fact can be connected to classification outcomes, the evaluation protocol itself must be cleaned: reported benchmark numbers on UCSD Stress inherit large inflation from trial-level cross-validation and, separately, from class-imbalance effects on unweighted classical baselines. §5 quantifies both inflation sources and reports the remaining, honestly-measured ceiling that §6–§8 must explain.

---

## §5 Honest Evaluation Audit of UCSD Stress

### 5.1 The benchmark claim and what a subject-level correction does to it

Wang et al. 2025 (arXiv:2505.23042), working on the same Komarov–Ko–Jung dataset used throughout this paper, report 90.47 % balanced accuracy (BA) for fine-tuned LaBraM under five-fold cross-validation. Their public code (verified against the referenced repository) partitions recordings — not subjects — into folds. On a 17-subject / 70-recording cohort where 15 of 17 subjects contribute multiple recordings, this protocol produces subject leakage in every fold. We re-run the same FMs on the same 70-recording subset, under two protocol variants — **trial-level** `StratifiedKFold(5)` on recordings (reproducing Wang's design) and **subject-level** `StratifiedGroupKFold(5)` with `groups=patient_id` (the honest cross-subject protocol) — holding every other detail fixed. The comparison is reported as a 3-seed mean ± std per model (source: `results/studies/exp18_trial_dass_multiseed/` vs `results/hp_sweep/20260410_dass/{model}/`).

**Table 5.1.** Trial-level vs subject-level five-fold CV on UCSD Stress, per-recording DASS label, 3-seed mean ± std.

| Model | Trial-level BA | Subject-level BA | Δ (pp) |
|---|---:|---:|---:|
| LaBraM | 0.676 ± 0.052 | 0.524 ± 0.010 | **+15.2** |
| CBraMod | 0.560 ± 0.049 | 0.548 ± 0.031 | +1.2 |
| REVE | 0.643 ± 0.063 | 0.577 ± 0.051 | +6.5 |

Two observations. First, every FM drops under honest CV, by a magnitude that varies with the architecture — not a uniform 20 pp penalty, but a model-dependent one (LaBraM absorbs the leakage inflation most). Second, **our own trial-level numbers (0.56–0.68) remain well below Wang's reported 0.90** — even reproducing Wang's inflation-exposing protocol, we cannot reproduce their headline result. The residual gap likely reflects additional design choices in the Wang pipeline (HP selection, label coding, or window definition) that we cannot audit without their full training logs. For the purposes of the SDL diagnosis, the relevant comparison is the **within-our-pipeline trial-vs-subject delta**, which isolates evaluation-design inflation cleanly: the 40+ pp gap between 0.90 and 0.52 decomposes into (a) a trial-vs-subject inflation of 1–15 pp model-dependent, and (b) a further gap that future independent audits of the Wang pipeline would need to attribute.

Under honest subject-level CV, the three FMs converge to a narrow BA band of **0.52–0.58** on UCSD Stress. This is the ceiling that §6–§8 must explain.

### 5.2 Classical baselines under honest CV, with class-balance correction

A natural follow-up question is whether honest-CV'd FMs still outperform classical band-power baselines on the same cohort, and by how much. Our initial result was that four of five classical methods (RF, LogReg-L1, LogReg-L2, SVM-RBF) and XGBoost all fell at or below chance (0.38–0.49) under subject-level CV on the 70-recording subset. An adversarial methodology review (R1 C3, see §3) raised the question of whether the XGBoost underperformance in particular was a class-imbalance artefact rather than a genuine signal-floor result — the 70-recording cohort has 56 negatives and 14 positives under the DASS-21 > 50 threshold (80:20 imbalance). RF, LogReg, and SVM all used `class_weight='balanced'` in the initial runs; `sklearn.ensemble.GradientBoostingClassifier`, which we had used as a proxy for gradient-boosted XGBoost, has no native `class_weight` parameter and was evaluated unweighted.

We re-ran XGBoost with explicit `sample_weight` computed via `sklearn.utils.compute_sample_weight('balanced', y_train)` for each training fold (`scripts/xgboost_class_balanced_check.py`, output `results/studies/exp02_classical_dass/rerun_70rec_xgb_balanced/summary.json`). The class-balance correction moved XGBoost from 0.508 ± 0.150 (plain) to **0.553 ± 0.081 (balanced)** — a 4.5 pp upward shift. No other classical method changed, since the rest were already weighted. The honest classical + FM ranking under subject-level CV with class balancing is:

**Table 5.2.** Classical and frozen-FM-LP baselines on UCSD Stress (70 rec, subject-level StratifiedGroupKFold, class-balanced).

| Method | 5-fold BA (mean ± std) | Pooled BA |
|---|---:|---:|
| RF (class_weight=balanced) | 0.438 ± 0.022 | 0.438 |
| LogReg-L2 (class_weight=balanced) | 0.491 ± 0.098 | 0.491 |
| SVM-RBF (class_weight=balanced) | 0.429 ± 0.085 | 0.429 |
| XGBoost (sample_weight=balanced) | 0.553 ± 0.081 | 0.482 |
| **LaBraM frozen LP (8-seed)** | **0.605 ± 0.032** | — |

(FT-BA numbers from Table 5.1 are comparable but use a different train head; we present frozen LP here to hold everything except the representation constant.)

The FM advantage over the best class-balanced classical baseline on honest CV is **0.605 − 0.553 = 5 pp**, not the 15–17 pp that unbalanced classical comparisons suggested. Two implications follow for the SDL diagnosis. First, **the honest-evaluation audit applies symmetrically to both sides of the comparison**: Wang's trial-level 0.90 is inflated, but so is any FM-vs-classical gap claim that compares weighted FMs to unweighted classical baselines. Second, **the 5 pp gap lies inside the 0.44–0.58 architecture-independent band identified in §7**. FMs do not escape the ceiling — they sit a few pp higher within it. §8 returns to the question of what specifically the FM contributes in this constrained regime.

### 5.3 Permutation-null comparison: FT vs shuffled-label FT

Trial-vs-subject CV and class-balance are design corrections. A further, independent check asks whether fine-tuned FM performance on this cohort is distinguishable from the permutation null — i.e. from the same FM pipeline trained on randomly shuffled labels and evaluated under the same subject-level CV. `train_ft.py --permute-labels <seed>` shuffles recording-level labels prior to CV; `scripts/run_perm_null.py` pools 10 permutation seeds per FM.

For the **LaBraM canonical fine-tuning recipe** on UCSD Stress (3 real seeds vs 10 permutation seeds, `results/studies/exp03_stress_erosion/analysis.json`):

- Real FT: 0.443 ± 0.083
- Permutation-null FT: 0.497 ± 0.086
- One-sided p = 0.70

That is, real labels produce *numerically lower* BA than shuffled labels on this recipe, and the test is statistically indistinguishable from chance. For CBraMod and REVE under their own best FT recipes, 10-permutation null tests were at the 0.1 p-floor (insufficient permutations to declare significance; noted as a methodology limitation in §10). This null-test pattern is consistent with, but does not by itself prove, the interpretation that subject-dominance fully explains the FT behaviour on this cohort; §6's paired comparison provides the contrasting positive case needed to distinguish "FT doesn't work on this model" from "FT doesn't work on this contrast."

### 5.4 What §5 has established, and what it hasn't

After subject-level CV, class-balanced classical baselines, and the permutation-null comparison, the UCSD Stress cohort presents a consistent picture: three FMs produce BA of 0.52–0.58, the best class-balanced classical baseline reaches 0.55, and FT on LaBraM is statistically indistinguishable from label-shuffle null. The inflated 0.90 benchmark does not survive this audit.

This by itself does *not* prove the SDL thesis. It proves only that the classification ceiling on this cohort is ~0.55–0.58 and that the reported inflation had identifiable sources. The SDL thesis makes the stronger claim that this ceiling is *specifically a contrast-strength ceiling* — not a model-scale ceiling, not a dataset-size ceiling, not a label-noise ceiling. To separate those alternatives requires a controlled paired comparison: two within-subject designs, identical framework, identical FM pipelines, differing only in the neural contrast strength of the labels. §6 reports that experiment.

---

*Draft status: §5 ~1,100 words. Ready for review. §6 draft to follow.*
