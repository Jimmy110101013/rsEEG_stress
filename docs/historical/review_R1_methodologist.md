# R1 — Adversarial Methodologist Review

**Reviewer**: R1 (Methodology)
**Target venue**: IEEE TNSRE
**Manuscript**: "Beyond Accuracy: What EEG Foundation Models Encode, and Why Fine-Tuning Direction Depends on Model × Dataset Interactions"
**Date**: 2026-04-15

I reviewed `docs/findings.md`, `docs/methodology_notes.md`, `docs/paper_strategy.md`, and `CLAUDE.md`. The paper claims three contributions — a representation diagnosis, a model × dataset FT taxonomy, and a "power-floor" cautionary tale — built on 70 recordings / 17 subjects / 14 positives (Stress), 195/65 (ADFTD), 734/359 (TDBRAIN), and 72/36 (EEGMAT), with 3 seeds almost everywhere. My review is not sympathetic because the authors asked it not to be.

---

## Top 3 fatal concerns (rejection grounds at TNSRE)

### F1. The headline FT-taxonomy claim (F-C) is statistically indistinguishable from seed noise on two of three models.

F-C is the paper's central empirical contribution — "FT direction is a model × dataset interaction, not label biology" — and §6 is explicitly the "headline section." The authors' own numbers say the claim does not survive scrutiny:

- **ADFTD, CBraMod**: Δ = **+0.83 ± 3.35 pp**. The std is **four times** the mean. By the authors' own annotation (`σ > |μ|`) this cell carries no signal. The authors propose to label this in the figure caption; that does not save the claim, it documents its absence.
- **TDBRAIN, CBraMod**: Δ = −0.02 ± 0.04 pp — an effect smaller than a rounding error. Calling this "flat" is generous; it is zero.
- **TDBRAIN, REVE**: Δ = +0.44 ± 0.32 pp. With n = 3 seeds, the sample-std 95% CI (t_{2, .975} ≈ 4.30) is approximately [−0.36, +1.24] — crosses zero.
- **ADFTD, REVE**: Δ = −1.53 ± 0.28 pp — 95% CI ≈ [−2.23, −0.83], excludes zero.
- **ADFTD, LaBraM**: +1.03 ± 0.74 pp — 95% CI ≈ [−0.81, +2.87], **crosses zero**.
- **TDBRAIN, LaBraM**: −1.56 ± 0.28 pp — CI excludes zero.

So the "opposite direction on identical labels" claim (F-C.1), as the paper states it, reduces to: LaBraM and REVE disagree on TDBRAIN direction, and on ADFTD only REVE's sign is significant under a naïve t-interval. Of **six** cells in the F-C.1 table, three have CIs that cross zero, one is effectively zero, one is σ > |μ|. The single pair that actually produces a statistically separable "opposite direction" on identical labels is **ADFTD REVE (−1.53) vs TDBRAIN LaBraM (−1.56)** — which is not "opposite", it is the **same** direction. The honest clean opposite-sign pair is TDBRAIN LaBraM negative / REVE positive, but REVE +0.44 crosses zero.

N-F15 reports "all models p=0.25" on paired permutation of the 3 matched seeds, and the sign test is 3/3 with exact p=0.125. These are not significant at any conventional threshold and would be caught by any TNSRE statistical reviewer in thirty seconds. Cohen's d > 2.5 is arithmetically real but operationally meaningless when n=3 and the denominator is a 3-sample std.

**This is the paper. If F-C.1 does not survive, neither does the title.**

### F2. The "honest ceiling" (F-D, 0.52–0.58) is not statistically distinguishable from a 2017 CNN or from chance.

F-D.2 reports ShallowConvNet 0.557 ± 0.031, with LaBraM FT at 0.524 ± 0.010, CBraMod FT at 0.548 ± 0.031, REVE FT at 0.577 ± 0.051. All five means sit inside a ≈ 5 pp band. With n = 3 seeds each, pairwise Welch t or paired sign tests on matched seeds will not reject equality. F-D.1 makes this explicit: LaBraM canonical real FT 0.443 ± 0.083 vs null FT 0.497 ± 0.086, **p = 0.70**. Two null seeds beat all real seeds. This is not "evidence of a ceiling"; this is the authors showing that their best-studied model cannot be separated from label permutation on their primary task.

N-F19 admits the other two models hit the 10-perm p-floor at 0.10, so CBraMod and REVE "injection" claims on Stress rest on **one bit** of evidence each (is any null seed above the real mean? barely). Reporting this as "direction consistent but noisy" is not a defensible scientific statement; it is wishful thinking. A TNSRE reviewer will correctly observe that the authors have not refuted the null for any model on Stress.

If the paper's central task (Stress) cannot reject the null, and the other two datasets carry the load via cells where half the CIs cross zero, the paper's claim of a taxonomy is unsupported.

### F3. The "contrast-strength" thesis is post-hoc and unfalsifiable as currently written.

Paper_strategy §2 and §6.4 introduce "within-subject contrast strength" as the operative factor distinguishing EEGMAT success (0.731 BA) from Stress longitudinal failure (0.30–0.43 BA). No operational definition of "contrast strength" is given. There is no independent measurement of the construct (effect size of rest vs arithmetic task per subject in EEGMAT vs rest-A vs rest-B DSS change per subject in Stress); the construct is inferred from outcome. Two datasets × one outcome per dataset cannot identify a continuous mechanism — this is a **narrative fit to two data points**.

A falsifiable version would require: (a) a pre-specified metric of contrast strength computed from raw EEG independently of FM outputs (e.g., paired-sample Hedges g on classical band-power between conditions); (b) a third and fourth dataset spanning a range of that metric; (c) a monotone relationship between the metric and Δ(FT − frozen). The paper has none of these. As stated, the thesis is unfalsifiable: any dataset where FMs fail is labeled "weak contrast", any where they succeed "strong contrast". This is the textbook structure of an ad-hoc hypothesis.

---

## Top 5 concerns authors can probably address

### C1. Multiple-testing burden is unaccounted for.

The paper reports effects across 3 FMs × 4 datasets × 2 phases (frozen/FT) × multiple metrics (pooled label fraction, BA, RSA, silhouette, Fisher, kNN, LogME, H-score) × 2 CV protocols. F-A "12/12 frozen model × dataset combinations" is the only place a multiplicity argument is even implicitly made. For F-C the authors highlight one opposite-sign pair out of six; if the family of tests is {6 direction tests}, Bonferroni-corrected α = 0.0083. The naïve t-intervals I computed above do not clear that bar for any cell except LaBraM TDBRAIN and REVE ADFTD. Under BH-FDR at q=0.10 across the 6 cells, the authors might retain 2 cells — exactly the two that point in the **same** direction. The paper must pre-register a test family, report corrected p-values, and state which cells survive.

### C2. StratifiedGroupKFold(5) on 17 subjects / 14 positives is not a well-posed CV design, and bootstrap CIs are misapplied.

With 17 subjects, 5 folds holds out 3–4 subjects per fold. With only 14 positive recordings spread across some subset of the 17 subjects (and the paper's own note that "1 subject per positive class per fold" made per-fold ω² degenerate), each fold's BA is computed on ≈ 2–3 positive recordings. The BA estimator's variance at that scale is enormous — a single misclassified recording shifts per-fold BA by 10–20 pp. The authors should demonstrate:

- **Seed × split stability**: refit the pipeline with 10 different GroupKFold random_state values and report the between-split std separately from the between-seed std. The reader currently cannot tell how much of the ± 0.010 / ± 0.031 / ± 0.051 sample std is model variance vs split variance.
- **Bootstrap over what?** N-F15 claims 10k-resample bootstrap CIs on per-seed BA, but with n=3 seeds the bootstrap is resampling 3 numbers. This is not a 10k-effective-resample CI; it has three distinct support points. The reported LaBraM frozen CI [0.585, 0.626] (width 0.041) is narrower than the sample std (0.032) which is impossible for an honest bootstrap of 3 samples unless the authors are bootstrapping recordings within fold, which conflates two sources of variance. Clarify exactly what is resampled.
- **Exact binomial vs permutation**: with 3 matched seeds the exact binomial min p is 0.125, which is correctly reported. The paper should stop citing this as support for anything; it is the paper's own admission that the n is too small to detect anything.

### C3. The F-B Axis-2 claim (classical collapse under honest labels) is narrower than implied.

F-B Axis 2 table shows RF 0.44, LogReg 0.49, XGB 0.44, SVM 0.43 — all at or below chance. On 70 recordings with 14 positives, **chance BA** for an imbalanced classifier is not 0.50 — a majority-class predictor gives BA = 0.50 by definition, but a classifier with any bias toward the majority class (which 56/14 imbalance strongly encourages) trivially produces BA < 0.50. These numbers may simply reflect that classical classifiers are regressing toward "always predict negative". The paper needs to demonstrate that classical BA < 0.50 is a real signal-absence claim, not an imbalance artifact — ideally via class-balanced loss weights, threshold tuning on OOF probabilities, or comparison to a stratified-random baseline. Otherwise the "FMs beat classical" framing rests on an artifact of how the baselines were run.

Additionally: if Wang et al. 2025 used per-recording labels (not subject-dass), then the Wang→our gap is **purely CV-design** (trial-level vs subject-level), and Axis 2 is not about Wang at all — it is about the authors' own prior internal RF result (0.666 subject-dass vs 0.44 per-rec). That is a fine internal correction, but Wang comparisons should be confined to Axis 1, and §1's hook must say so. Conflating the two axes in the Introduction would be a genuine misrepresentation and would be caught.

### C4. Power calculation for "3 seeds" is absent.

The paper relies on 3-seed means throughout. Nowhere is there a power analysis justifying that n = 3 is adequate to detect the claimed effects. Given §2 reports 70-rec / 14-positive runs with sample stds of 0.03–0.08, a 5 pp effect (Δ ≈ 0.05) requires ≈ (1.96·0.05/0.05)² ≈ 4 seeds under a normal-approximation two-sample z — and paired tests give you back maybe a factor of 2, so ≥ 6 seeds at that std. The paper claims **sub-1 pp** representation deltas (F-C.1) from **3 seeds** at stds of 0.28–3.35 pp. That is mathematically underpowered: for Δ = 1 pp, σ = 0.7, n = 3, the observed-power is roughly 20%. The authors should either run 10+ seeds on the cells they wish to claim as significant, or downgrade the F-C.1 language from "direction" to "observation, underpowered".

### C5. N-F21 LaBraM bug-fix is being hidden, and "do NOT cite" is the wrong call.

`methodology_notes.md` N-F21 documents that LaBraM had an architectural mismatch (extra LayerNorm on mean-pooled patch tokens, pretrained `norm.*` weights stacked under a fresh `fc_norm`) that was fixed on 2026-04-15. The authors decided **not to re-run canonical experiments** and **not to disclose the bug in the paper**. The stated rationale is that the Wang-protocol window-level mean moved < 1 pp; however the same table shows recording-level BA moved −5.4 pp on the 12-seed mean, with per-seed swings of ±25 pp. The authors' own numbers say the bug is material at the seed-level granularity that the paper's 3-seed claims operate at.

Concealing a known architectural bug on the primary FM, while publishing 3-seed claims that are within the bug's per-seed noise envelope, is a research-integrity problem. Reviewers and editors at TNSRE will catch this via the supplementary code release and the paper will be rejected outright. The correct action is (a) re-run the 3-seed canonical experiments with the fixed architecture, (b) report both numbers in a supplementary table, (c) make a factual statement that the paper's conclusions are robust to the fix. Anything else invites retraction.

---

## Specific experiments / analyses to pre-empt reviewers

1. **Extend Stress permutation null to 200 perms at best-HP for all three FMs.** The 10-perm floor is the single biggest ammunition for a hostile reviewer. 200 perms × 3 models × 3 seeds is ~1800 FT runs — expensive but tractable on 3 GPUs. Alternative: 1 seed × 200 perms, and report p on the best-seed real value. Cite p-value with exact Monte-Carlo CI (Clopper-Pearson).
2. **10-seed (not 3-seed) re-run of the F-C.1 taxonomy on LaBraM ADFTD, LaBraM TDBRAIN, REVE ADFTD, REVE TDBRAIN** (the 4 cells the paper wants to claim). Report 95% t-interval **and** Bonferroni-corrected p from a paired permutation test (FT − frozen per seed, permute signs).
3. **Pre-register a contrast-strength metric** (e.g., per-subject paired Hedges g on classical band-power between conditions, averaged over subjects) and compute it for all 4 datasets. Show a scatter of contrast-strength vs Δ(FT − frozen). If the scatter is not monotone, drop the thesis. Two-point thesis is not publishable as mechanism.
4. **Seed × split separation**. For each F-C.1 cell, run 3 seeds × 5 split-seeds = 15 evaluations and decompose variance into (seed | split) + split. Report both stds; the paper currently shows only one.
5. **Re-run LaBraM canonical experiments with the N-F21 fix.** Put both old and new numbers in a supplementary table; acknowledge in Methods. Without this, reject is likely on reproducibility grounds alone.
6. **Power calculation section in Methods**: minimum detectable effect size at n = 3 seeds, n = 17 subjects, n = 14 positives. State explicitly which claims are underpowered and should be read as exploratory.
7. **Correct the "chance" baseline in F-B Axis 2.** Report stratified-dummy BA, and classical baselines with class-balanced loss + threshold tuning. If RF with balanced weights still gives 0.44, the claim is real; if it recovers to 0.55, Axis 2 collapses.
8. **Effective multiple-comparison correction across F-A, F-C, F-D.** State the test family up front; report BH-FDR-corrected q-values for every numerical claim in the paper.

---

## Minor issues (not blocking but will be flagged)

- F-C.4 / N-F18: the paper reports REVE Stress Δ = +8.3 pp in §6 but admits the principled 5s-matched number is +4.8 pp with a footnote. A reviewer will ask why the main number is not the principled one. Swap them. The footnote can be the 10s-native alternative.
- "Erosion is model-universal" was in earlier drafts, is now "refuted". This is fine but the paper should explicitly acknowledge that a prior version of the authors' own framing was wrong, and explain why. TNSRE reviewers sometimes receive earlier drafts; consistency matters.
- F-D.3 longitudinal DSS failure (BA 0.17–0.43) belongs in the appendix as the authors plan, but the appendix table has linear BA = 0.000 in all cells — that is not "failure", that is "the model refused to train". Investigate before publishing; a reader will ask.
- F-E alpha lateralization is correctly flagged as context-only. Good.
- The "FMs beat classical on Stress" claim (LaBraM frozen LP 0.605 vs RF 0.44) rests on classical possibly being broken (C3 above). If C3 is addressed and classical recovers, the entire §5 narrative weakens.

---

## Verdict

**Major revision, leaning reject.** The paper's central empirical claim (F-C FT taxonomy) is underpowered at the n = 3 seed scale used; the "power-floor" cautionary tale (F-D) honestly admits the null cannot be rejected on the primary dataset; and an undisclosed architectural bug in the primary FM (N-F21) must be openly reported and its canonical numbers re-run before any TNSRE submission. Authors can salvage this with 10+ seeds on 4 specified cells, 200-perm null on Stress, a pre-registered contrast-strength metric across ≥4 datasets, and full disclosure of the LaBraM fix — but as submitted, a methodology-literate reviewer will vote reject.
