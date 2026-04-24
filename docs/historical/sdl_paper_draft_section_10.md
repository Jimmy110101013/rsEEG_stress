# SDL Paper Draft — §10 Limitations

**Date**: 2026-04-15
**Role in SDL thesis**: *the empirical-boundary statement.* §9 stated what SDL claims and does not claim. §10 specifies the sample-size, psychometric, and design boundaries that any reader of §4–§9 should weight against the findings, and pre-empts the reasonable methodological objections a reviewer is likely to raise.

---

## §10 Limitations

### 10.1 Cohort size and single-seed fragility

The primary UCSD Stress cohort used throughout §5–§7 consists of 70 recordings from 17 subjects, with a 14:56 positive:negative class split. At this scale, single-seed balanced accuracies are known to swing by 5–10 pp on the same pipeline with no code changes, a fragility consistent with independent large-scale small-N EEG stability literature. All headline numbers in this paper are reported as 3-seed mean ± sample std; seed counts are held constant across the trial-level / subject-level / class-balance / architecture / HP arms of the analysis. A reader interpreting a given 3-seed point estimate should weight it against the ±5 pp (typical) to ±10 pp (worst observed) seed envelope, and the relative claims — not the absolute BA digits — carry the weight of the argument.

A consequence for the §5.3 permutation-null analysis is that the comparison budget is 10 permutation seeds per FM (2 GPU-weeks at our scale), below the 100+ permutations that would drive the p-floor well below 0.01. For CBraMod and REVE this means the null test bottoms out at p ≈ 0.1 and we cannot distinguish "true null" from "small real effect swamped by seed noise." We flag these two cells as underpowered relative to LaBraM's p = 0.70 real-vs-null comparison. The SDL diagnostic does not hinge on the REVE/CBraMod null cells — the controlled EEGMAT vs Stress paired comparison in §6 carries the thesis's weight — but the permutation-null analysis on a larger permutation pool would sharpen §5's audit claim and is a natural next replication target.

### 10.2 DASS-21 and DSS psychometric scope

DASS-21 is validated as a past-week state/trait screener and DSS as a daily-stress scale aggregated at the subject level; neither was designed or psychometrically validated as a per-recording single-session classification label. The per-recording DASS-thresholded classification used by Wang et al. 2025, and reproduced by us in §5 for the purpose of independent audit, extends these instruments beyond their validation envelope. A reader interpreting the §5–§6 Stress results should hold this scope in mind: the failure of FMs to classify per-recording DASS labels is a joint statement about FM capability and about the instrument–temporal-resolution mismatch, and we cannot separate these two factors with the current data.

This limitation, importantly, *does not weaken* the SDL thesis: it strengthens it. The two-regime picture in §6 and §8 relies on a sharp anchored-vs-unanchored contrast between EEGMAT (neural anchor established) and UCSD Stress (neural anchor absent), and the psychometric-scope mismatch is one of the reasons the Stress anchor is absent. A reader who rejects the Stress arm on psychometric grounds has independently produced the SDL-bounded-regime classification that our diagnostic assigns to this cohort.

### 10.3 Single anchor on the rescue arm

Our controlled paired comparison in §6 anchors one rescue dataset (EEGMAT, Klimesch-type alpha desynchronisation) against one bounded dataset (UCSD Stress longitudinal DSS, absent literature anchor). A natural follow-up would use multiple anchored contrasts — at minimum EEGMAT, ADFTD (AD alpha slowing), and perhaps a motor-imagery beta-ERD task — and multiple bounded contrasts, to map a contrast-strength gradient rather than a binary. The §8.3 observation that ADFTD produces a similar classical-to-frozen-to-FT cascade to EEGMAT is a preliminary step in this direction, but ADFTD's contrast-strength anchor (AD alpha slowing) is cross-subject rather than within-subject, and is therefore not directly comparable to the §6 paired design. A multi-dataset paired replication with a range of within-subject anchor strengths would strengthen the quantitative claim that the anchor-strength-to-rescue-magnitude relationship is monotonic.

### 10.4 Architecture panel breadth and per-model best-HP selection

The §7 architecture panel includes seven from-scratch and pretrained models spanning five orders of magnitude in parameter count but deliberately omits two classes of architecture that would strengthen the claim in different directions: (i) graph-based EEG models (GNNs over sensor adjacency), and (ii) state-space and Mamba-style models. Our panel is representative of the convolutional and transformer families currently dominant in EEG-FM literature, but a reader who suspects that an inductive bias specific to these omitted families might escape the ceiling has a legitimate concern that §7's panel does not directly address.

Separately, the three FM fine-tuning results cited throughout §5–§8 use per-model best-HP selection over a small 3 × 2 × 3 grid (learning rate × encoder-LR scale × seed) independently per model. This is an argmax-over-HP selection and not a factorial cross-model comparison. The per-model headline numbers we report (LaBraM 0.524, CBraMod 0.548, REVE 0.577) should be read as "each FM at its own best tested HP," not as a controlled ranking between architectures. §7's architecture-independent-ceiling claim does not rely on cross-FM ranking; it relies only on the fact that every tested architecture, at its own best HP, falls inside the same 5 pp window.

### 10.5 LaBraM `self.norm` architectural disclosure

During preparation of this manuscript we discovered and corrected an implementation drift in our LaBraM codebase: the `self.norm` LayerNorm was not set to Identity under mean-pooled readout, which caused pretrained norm weights intended for the CLS-token pretraining head to be reloaded into a second post-backbone LayerNorm. The corrected implementation matches the official LaBraM reference. A 3-seed subject-level spot-check at our canonical best-HP configuration on UCSD Stress confirms that the Δ between pre-fix and post-fix is within the cuDNN-determinism envelope reported for this cohort (pre-fix 0.524 ± 0.010 vs post-fix 0.538 ± 0.058, Δ = +1.4 pp — within the single-model seed std on both sides). All main-text LaBraM Stress FT claims use the pre-fix multi-seed run for internal consistency across §5–§7 tables; the post-fix spot-check is reported alongside as evidence that the architectural drift is immaterial to the paper's conclusions. The full exp23 vs exp24 Wang-protocol diagnostic (12 seeds each, additional analysis of window-level vs recording-level divergences) is reported in supplementary for transparency. We replicated the audit on CBraMod and REVE and confirm no analogous drift.

### 10.6 Reproducibility and what the reader can independently verify

All code, trained checkpoints, variance-decomposition outputs, permutation-null logs, per-seed training curves, the N-F21 pre/post diagnostic, the class-balanced XGBoost audit, and the contrast-strength anchoring protocol specification will be released at the project repository upon publication. We specifically invite independent groups to (i) reproduce the trial-vs-subject CV decomposition on their own Stress pipelines to confirm the evaluation-inflation effect is not specific to our code; (ii) replicate the paired EEGMAT vs Stress longitudinal comparison under their own FM weights and HP budgets; and (iii) apply the pre-benchmark contrast-anchoring protocol to their own clinical cohorts and report whether the SDL prediction holds.

---

*Draft status: §10 ~1,200 words. Draft complete for §1–§10.*
