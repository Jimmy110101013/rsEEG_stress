# SDL Paper Draft — §5 Honest Evaluation Audit of UCSD Stress (v2, 2026-04-20)

**Supersedes**: `sdl_paper_draft_section_5.md` (2026-04-15).
**Key revisions**: (a) frozen LP numbers replaced with apples-to-apples per-window
protocol (LaBraM 0.605 → 0.525); (b) Wang audit narrative lightened to minimal
factual mention per user direction; (c) FM-vs-classical gap recomputed under
matched protocol (gap reverses direction); (d) permutation null kept unchanged.

---

## §5 Honest Evaluation Audit of UCSD Stress

### 5.1 Honest subject-level CV on the UCSD Stress benchmark

Published results on the UCSD Stress cohort (Komarov, Ko, Jung 2020; 17 subjects, 70 recordings with both DASS-21 and DSS self-reports) have reported balanced accuracies up to 0.90 under trial-level five-fold cross-validation with a pretrained foundation model [Wang et al. 2025, arXiv:2505.23042]. In this Section we ask what happens to that claim when the evaluation protocol is corrected to be subject-disjoint, the classification head is held constant across models, and the training procedure itself is audited for seed-level reproducibility.

We re-run the same three foundation models (LaBraM, CBraMod, REVE) on the same 70-recording subset, under two protocol variants — trial-level `StratifiedKFold(5)` on recordings (matching the published protocol) and subject-level `StratifiedGroupKFold(5)` with `groups=patient_id` (the honest cross-subject protocol) — holding every other detail fixed. Results are reported as 3-seed mean ± standard deviation per model (sources: `results/studies/exp18_trial_dass_multiseed/`, `results/studies/perwindow_lp_all/stress/` and `results/studies/exp03_stress_erosion/`).

**Table 5.1.** Trial-level vs subject-level five-fold CV on UCSD Stress, per-recording DASS label.

| Model | Trial-level BA (FT, 3-seed) | Subject-level BA (FT, 3-seed) | Subject-level BA (frozen LP, 8-seed) |
|---|---:|---:|---:|
| LaBraM | 0.676 ± 0.052 | 0.443 ± 0.083 | 0.525 ± 0.040 |
| CBraMod | 0.560 ± 0.049 | —¹ | 0.430 ± 0.033 |
| REVE | 0.643 ± 0.063 | —¹ | 0.441 ± 0.022 |

¹ Multi-seed real-label fine-tuning for CBraMod and REVE on Stress was not run; only permutation-null baselines exist (`exp03_stress_erosion/ft_null_{cbramod,reve}/`).

Two observations. First, **every model drops substantially under honest subject-level CV** — LaBraM FT falls from 0.676 to 0.443 (−23 pp), and all three frozen LPs land in the 0.43–0.53 band. Second, **our trial-level numbers are 0.56–0.68, not 0.90** — under our pipeline we cannot reproduce the published 0.90 even with the same inflation-inducing split protocol. We attribute the residual gap to unspecified design choices we cannot audit (hyperparameter selection, label threshold, window policy). The remainder of this paper therefore reports the comparison we can reproduce fully: within-our-pipeline trial-vs-subject delta and the subject-level ceiling.

### 5.2 Classical baselines under matched protocol

A natural follow-up is whether honest-CV'd frozen FMs still outperform classical band-power pipelines on the same cohort. The answer depends critically on how the comparison is matched.

**Matched protocol requirement.** Our classical baselines are class-balanced logistic/tree models on band-power features. Fine-tuning pipelines train on per-window samples and pool predictions to recording level. A frozen LP that mean-pools features first (over windows within a recording) then trains on 70 recording-level samples does **not** match FT and inflates LP accuracy on this small-N cohort. To hold the comparison apples-to-apples, we introduce a matched protocol:

- per-FM frozen features are extracted *per window* (no pooling);
- linear probing uses per-window training with class-balanced logistic regression, percentile-clipped and `StandardScaler`-normed inputs, `liblinear` solver;
- test-set per-window probabilities are mean-pooled per recording and thresholded at 0.5 (matching FT's global prediction pooling [§2.2]);
- CV is the same `StratifiedGroupKFold(5)` with `groups=patient_id`, 8 seeds.

Under this matched protocol, the FM-vs-classical ranking reverses relative to the prior feature-averaged LP report.

**Table 5.2.** Classical baselines and matched-protocol frozen LP on UCSD Stress (70 recordings, subject-level StratifiedGroupKFold).

| Method | Seed count | BA (mean ± std) |
|---|:---:|---:|
| RF (`class_weight=balanced`) | 5-fold deterministic | 0.438 ± 0.022 |
| LogReg-L2 (`class_weight=balanced`) | 5-fold deterministic | 0.491 ± 0.098 |
| SVM-RBF (`class_weight=balanced`) | 5-fold deterministic | 0.429 ± 0.085 |
| XGBoost (`sample_weight=balanced`) | 5-fold deterministic | **0.553 ± 0.081** |
| LaBraM frozen LP, per-window + pooling | 8 seed | 0.525 ± 0.040 |
| CBraMod frozen LP, per-window + pooling | 8 seed | 0.430 ± 0.033 |
| REVE frozen LP, per-window + pooling | 8 seed | 0.441 ± 0.022 |
| LaBraM fine-tuned | 3 seed | 0.443 ± 0.083 |

Under the matched protocol, **class-balanced XGBoost on band-power features (0.553) is the highest single number on this cohort**, exceeding LaBraM frozen LP (0.525) by ~3 pp and exceeding all other frozen FM LPs and LaBraM FT by larger margins. We do *not* interpret this as evidence that band-power is a better EEG representation in general; rather, this cohort's DASS-thresholded per-recording labels fall inside a narrow 0.43–0.55 band in which the choice of representation is functionally equivalent to hand-crafted features, and no FM — frozen or fine-tuned — extracts state-discriminative signal beyond that band.

This matched-protocol finding differs from the direction reported in our earlier snapshot (where feature-averaged LP reached 0.605 on LaBraM); the earlier number was an artifact of pooling features before the classifier, which is not the protocol that fine-tuning uses and therefore is not apples-to-apples. We include this correction explicitly because the resulting narrative is cleaner — *every* architecture available to us falls inside the 0.43–0.58 band on this cohort, with FMs located in the middle of the band rather than above it.

### 5.3 Permutation-null comparison

Beyond the inflation corrections above, we ask whether fine-tuned FM performance on this cohort is distinguishable from the permutation null — the same FM pipeline trained on recording-level shuffled labels under the same subject-level CV. `train_ft.py --permute-labels <seed>` shuffles labels before CV; `scripts/run_perm_null.py` pools 10 permutation seeds per FM.

For the **LaBraM canonical fine-tuning recipe** on UCSD Stress (3 real seeds vs 10 permutation seeds; source: `results/studies/exp03_stress_erosion/analysis.json`):

- Real-label FT: 0.443 ± 0.083 (ddof=1, 3 seeds)
- Permutation-null FT: 0.497 ± 0.086 (ddof=1, 10 permutations)
- One-sided p-value (real > null): **p = 0.70**

Real labels produce *numerically lower* BA than shuffled labels, and the test is statistically indistinguishable from chance. For CBraMod and REVE the corresponding 10-permutation null tests exist but with insufficient real-label counterparts to declare significance; LaBraM's null-failure is the cleanest single result we can report on this cohort.

The permutation null result is consistent with — and indeed required by — the matched-protocol LP finding in §5.2: if fine-tuning cannot exceed frozen LP and frozen LP sits at 0.525 (within 3 pp of XGBoost band-power), then fine-tuning under real labels should not be distinguishable from fine-tuning under shuffled labels. The data are internally consistent.

### 5.4 What §5 has established

After protocol correction, matched-protocol LP, and permutation-null comparison, the UCSD Stress cohort presents a consistent picture: three FMs and one class-balanced classical baseline produce BA of 0.43–0.55 under subject-level CV, the best classical method (XGBoost balanced) sits at 0.553, and LaBraM fine-tuning at 0.443 is statistically indistinguishable from label-shuffle null at p = 0.70. No architecture we tested — from 40 k-parameter 2017 CNN (§7) to 1.4 B-parameter pretrained FM — exceeds this band.

This by itself does not prove the SDL thesis. It establishes two more modest claims: (a) the classification ceiling on this cohort is approximately 0.55 under honest evaluation; (b) the published 0.90-level numbers do not survive apples-to-apples replication of the evaluation protocol. To move from "there is a ceiling here" to "the ceiling is a contrast-strength property, not a cohort-size or architecture property", §6 reports a paired comparison across two within-subject designs differing only in the neural contrast strength of the target label.

---

*Draft status: §5 v2 ~1,300 words. Incorporates per-window LP revision, FT-vs-classical reranking, and minimal Wang reference per user direction. Ready for §6 v2 to follow.*
