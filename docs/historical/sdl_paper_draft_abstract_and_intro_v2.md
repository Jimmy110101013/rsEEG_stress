# SDL Paper Draft — Abstract + §1 Introduction (v2, 2026-04-20)

**Framing**: Path A — clinical-engineering audience (~60% clinical / ~40% engineering). Target venue: *Journal of Neural Engineering*, *Brain Informatics*, *IEEE TNSRE*, or *Frontiers in Neuroinformatics*. Tone: practical, decision-oriented, concrete. Avoids pure-ML framing; leads with the clinical practitioner's question.

**Supersedes**: the original thesis-format abstract and §1 in `sdl_paper_draft_sections_1-3.md` (2026-04-15).

---

## Abstract

Electroencephalography foundation models (EEG-FMs) have reported strong performance on diagnostic benchmarks, rekindling interest in applying them to small clinical cohorts. We ask a practical question facing any clinical researcher with a 20–500-subject resting-state EEG dataset: *can I fine-tune an EEG-FM on my data and expect a usable model?* Across three publicly available pretrained FMs (LaBraM, CBraMod, REVE) and four clinical cohorts (UCSD Stress, ADFTD Alzheimer's, TDBRAIN MDD, EEGMAT mental arithmetic), evaluated under matched per-window linear probing and per-FM canonical fine-tuning with subject-disjoint cross-validation, we find that fine-tuning exceeds a frozen linear probe in only 1 of 12 cells (LaBraM × ADFTD, +5.6 pp); 2 of 12 show statistically suspect apparent rescues on a small imbalanced cohort (Stress, likely attributable to label–subject collinearity); and 6 of 12 cells show fine-tuning *degrading* performance below frozen linear probing. A published benchmark on UCSD Stress reporting 0.90 balanced accuracy drops to 0.43–0.55 under honest subject-level evaluation. Using controlled aperiodic-vs-periodic ablation based on FOOOF spectral parameterisation, we demonstrate causally that it is the 1/f aperiodic background — not periodic oscillations — that carries subject identity in frozen FM embeddings: removing the aperiodic component collapses subject-identity decodability (up to −26 percentage points, REVE on EEGMAT), whereas removing periodic peaks leaves it unchanged (≤ 0.7 pp). We distil these findings into a three-step pre-flight checklist for clinical EEG-FM deployment: (1) verify the target label has a published within-recording neural anchor; (2) measure frozen-LP balanced accuracy under subject-disjoint cross-validation; (3) check subject-identity decodability against state-label decodability to flag shortcut risk. Clinical researchers with cohorts lacking a published neural anchor should expect classification ceilings near 0.55 BA regardless of FM choice or fine-tuning effort; classical band-power baselines are within 5 pp of anything our three FMs achieved on that label class.

---

## §1 Introduction

Resting-state electroencephalography (rsEEG) is being increasingly proposed as a low-cost, accessible neurophysiological biomarker for psychiatric and neurological conditions, from Alzheimer's disease to depression to chronic stress. The clinical appeal is obvious: a 5–10 minute eyes-closed recording on a 19-channel cap costs a fraction of MRI and can be deployed in primary care or remote settings. The scientific question — whether rsEEG contains enough individual-level signal to distinguish clinical states across subjects — is less settled.

Recent work on large EEG foundation models (FMs) has argued the answer is yes. LaBraM [Jiang et al. 2024], CBraMod [Wang et al. 2024], REVE [Zhong et al. 2025], and related architectures train on thousands of hours of aggregated public EEG data and report strong performance on downstream clinical benchmarks. In one representative example, Wang et al. 2025 (arXiv:2505.23042) report 90.47 % balanced accuracy for fine-tuned LaBraM on a 17-subject rsEEG stress cohort [Komarov, Ko, Jung 2020]. Taken at face value, this result would imply that modern EEG FMs approximate a useful clinical classifier for an affective state on a cohort of fewer than 100 recordings — a regime in which traditional signal processing has historically struggled.

A clinical researcher reading such a result faces a concrete decision: should I buy a GPU, fine-tune this FM on my 50-subject Parkinson's cohort, and expect the reported performance levels to transfer? This paper answers that question empirically, and the answer is: **not without verification that is rarely reported in the benchmark literature**.

### 1.1 What we find

Across 12 (FM × dataset) cells — three pretrained EEG FMs evaluated on four publicly available clinical rsEEG cohorts (UCSD Stress 17 subjects, ADFTD 65 subjects, TDBRAIN 359 subjects, EEGMAT 36 subjects) — under subject-disjoint cross-validation and matched per-window evaluation protocols for frozen linear probing versus fine-tuning, we find four consistent patterns:

1. **Fine-tuning rarely exceeds a frozen linear probe.** Only 1 of 12 cells shows fine-tuning providing a clinically meaningful rescue (+5.6 pp on LaBraM × ADFTD). Three cells are tied within 1 pp. Six cells show fine-tuning *degrading* classification accuracy below the frozen baseline.

2. **Single-dataset published benchmarks can collapse by 20–40 pp under honest evaluation.** On the UCSD Stress cohort [Komarov 2020], the previously reported 0.90 BA falls to 0.43–0.53 across three FMs when the evaluation protocol is replaced by subject-disjoint cross-validation, and to null-indistinguishable under permutation-label control (LaBraM FT p = 0.70). No architecture we tested — from a 3-thousand-parameter 2018 compact CNN to a 1.4-billion-parameter 2025 foundation model — exceeds this honest ceiling.

3. **Apparent "FT rescues" on small imbalanced cohorts are mechanistically suspect.** Two of the three positive ΔFT values in our grid occur on the 17-subject Stress cohort with 14 positives concentrated in a subset of subjects. This structural collinearity between subject identity and recording label offers FT a shortcut: predict label by matching subject-fingerprint. Verifying whether these rescues are real signal acquisition or shortcut exploitation requires a permutation-null control that is rarely reported.

4. **Subject identity in frozen FM embeddings is causally carried by aperiodic 1/f structure.** Using a FOOOF-based ablation that surgically removes the aperiodic background from EEG input and re-extracts FM features, we show that subject-identity decodability from the frozen embedding drops by up to 26 percentage points when aperiodic is removed but remains essentially unchanged when periodic oscillatory peaks are removed. The 1/f background — a physiological trait property shaped by cortical E/I balance and volume conduction [Donoghue et al. 2020 Nat Neurosci; Gao et al. 2017] — is the substrate of subject dominance in modern EEG FM representations. Periodic oscillations are not.

### 1.2 What this means for clinical deployment

These four patterns have a single joint implication for clinical EEG-FM use: **whether a given (FM × cohort × label) combination will produce a useful clinical classifier is determined mostly by properties of the target label and cohort, not by the FM or by the fine-tuning effort**. We operationalise this in a three-step pre-flight checklist (§9):

- **Check 1** (literature search, < 1 hour): Does the target label have a published within-recording neural anchor — a peer-reviewed epoch-level EEG effect tied to the label category?

- **Check 2** (frozen LP, < 2 hours compute): Under subject-disjoint CV, does the frozen FM embedding linearly separate the target label at BA > 0.55?

- **Check 3** (subject-identity probe, minutes): Does the FM embedding also linearly separate subject identity at rates that dominate state-label separability? If so, any fine-tuning gain is shortcut-suspect until a permutation-null control confirms otherwise.

On the four cohorts studied, only EEGMAT (rest vs arithmetic, alpha-ERD anchor) and ADFTD (Alzheimer's, alpha-slowing anchor) pass all three checks. TDBRAIN and Stress fail Check 1, and reported FM benchmarks on these cohorts should be interpreted with suspicion.

### 1.3 Scope and contribution

This paper does not propose a new architecture, a new pretraining objective, or a new dataset. We propose three things:

1. **A diagnostic protocol** for auditing published EEG-FM benchmarks on small clinical cohorts (§5–§8). Applied to the UCSD Stress benchmark, the diagnostic accounts for a 20–40 pp accuracy inflation that previously lacked an explanation.

2. **A causal demonstration** — via FOOOF aperiodic/periodic ablation (§6.5) — that subject dominance in EEG FM embeddings is carried by the 1/f background and not by periodic oscillations. This is, to our knowledge, the first causal evidence on FM embeddings for a claim previously supported only by correlational EEG-biometric studies [Demuru & Fraschini 2020; Lanzone et al. 2023].

3. **A deployment checklist** for clinical researchers and reviewers (§9) that turns §5–§8's findings into three concrete pre-flight checks. The checklist is cheap to run (< 1 day of compute on a consumer GPU for typical cohorts) and can be reported as a standard supplementary diagnostic alongside any EEG-FM clinical benchmark.

### 1.4 Organisation

§2 situates this work relative to the EEG-FM benchmark literature, subject-leakage warnings (BENDR, AdaBrain-Bench, Brookshire 2024), and aperiodic-as-fingerprint work (Demuru 2020, Donoghue 2020). §3 describes datasets, foundation models, and protocols. §4 establishes the frozen subject-atlas structure of the three FMs across four datasets. §5 presents the honest re-evaluation of the UCSD Stress benchmark. §6 reports the paired EEGMAT–Stress comparison, cross-architecture band consensus, and FOOOF-based causal ablation. §7 establishes architecture-independence of the Stress ceiling across seven models spanning six orders of magnitude in parameter count. §8 presents the cross-dataset FT-versus-LP grid. §9 presents the clinical pre-flight checklist. §10 lists limitations, open questions, and future work.

---

*Draft status: Abstract ~280 words, §1 ~1,400 words. Written for clinical-engineering audience. Lead question is practitioner-facing. Technical depth preserved for reviewers.*
