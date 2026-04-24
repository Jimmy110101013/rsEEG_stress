# SDL Paper Draft — §9 Discussion

**Date**: 2026-04-15
**Role in SDL thesis**: *the synthesis.* Consolidates §4–§8 into the two-regime SDL diagnostic, positions the diagnostic against the surrounding EEG-FM benchmark-audit literature, and articulates the pre-benchmark contrast-anchoring protocol that follows from it.

---

## §9 Discussion

### 9.1 The SDL diagnostic, stated compactly

Combining the four experimental chapters yields a compact diagnostic. Frozen EEG FM representations are subject atlases (§4): across every model × dataset pair we examined, subject identity accounts for 10×–50× more representation variance than task labels, and representational similarity analyses agree with variance decompositions in every cell. Against that structural baseline, classification outcomes on a benchmark are governed not by architectural choice or model scale, but by whether the downstream label has an externally-anchored within-subject neural contrast. When such an anchor exists (EEGMAT mental-arithmetic alpha desynchronisation; Alzheimer's-disease resting alpha slowing), fine-tuning can project the label axis onto the subject-dominated representation and deliver measurable rescue above classical baselines (§6, §8). When it does not (UCSD Stress per-recording DASS or longitudinal DSS, under the psychometric scope in which these labels were originally validated), no architecture within five orders of magnitude of parameter scale exceeds a shared ceiling (§5, §7), and fine-tuning on a pretrained FM does not reliably distinguish itself from label-shuffle null.

This picture has two regimes of practical consequence. In the **anchored-contrast regime**, pretrained EEG FMs deliver real value — a frozen-representation bump above classical features *plus* a fine-tuning rescue step that reorganises the representation around the label axis. In the **contrast-bounded regime**, pretrained EEG FMs deliver a small frozen-representation bump *only*, and the fine-tuning step does not produce further gains on average and in some (model, dataset) combinations actively erodes the frozen representation.

We call this diagnostic the **Subject-Dominance Limit (SDL)**: the classification performance achievable from any architecture within the tested range is bounded above by the quality of the within-subject neural contrast available to the label, not by the representational capacity of the model.

### 9.2 Relationship to concurrent EEG-FM benchmark audits

The SDL diagnostic complements rather than displaces the recent wave of large-scale EEG-FM benchmark audits. EEG-Bench (Scherer et al., NeurIPS 2025) reports across 14 datasets × 11 clinical tasks that "while foundation models achieve strong performance in certain settings, simpler models often remain competitive, particularly under clinical distribution shifts." Kauffmann et al. (NeurIPS 2025) note that linear probing of EEG FMs is "relatively worse" than full fine-tuning and that general time-series FMs sometimes outperform EEG-specific ones. EEG-FM-Bench (Xiong et al., 2025) and AdaBrain-Bench (2025) converge on similar aggregate patterns. Each of these audits diagnoses *that* FMs sometimes fail; none of them diagnoses *which property of the downstream task* predicts the failure. SDL supplies that mechanistic layer: the property is the presence or absence of an external within-subject neural anchor, and the diagnostic can be applied *before* running a benchmark rather than inferred after it.

SDL is also compatible with recent non-EEG observations about fine-tuning dynamics. Chen et al.'s 757-model supervised-fine-tuning study on language models concluded that "the inductive biases of the base model outweigh the specific SFT corpus in determining the final representation" — a LLM-scale instance of the same projection-not-rewrite phenomenon we observed on EEG FMs. The cross-modality recurrence of projection-not-rewrite suggests that what we are reporting on EEG is a special case of a broader fine-tuning regime that holds whenever the pretraining signal dominates the SFT signal.

### 9.3 The pre-benchmark contrast-anchoring protocol

The operational consequence of SDL is a pre-registration style diagnostic that any research group deploying EEG FMs on a new clinical cohort can apply before investing in benchmark work. The protocol has three steps. (i) For the target label, conduct a systematic search for published within-subject EEG correlates at the temporal resolution of the intended classifier (single-recording, trial-level, or trajectory). (ii) If such a literature exists, extract the expected effect size and spectral locus and pre-register them as the contrast-strength anchor. (iii) If no such literature exists, declare the dataset a *contrast-bounded benchmark candidate*, pre-commit to honest subject-level CV plus class-balanced baselines plus permutation-null comparison, and report the resulting FM numbers as bounded by the subject-dominance ceiling rather than as an absolute FM capability claim.

This protocol recasts FM-on-small-clinical-cohort work as a two-stage enterprise: first anchor the task, then benchmark the architecture. It inverts the common current practice in which benchmarks are run first and the absence of a within-subject neural anchor is discovered, if at all, only when headline numbers fail to replicate across groups.

### 9.4 What SDL does not claim

We want to be explicit about three statements the SDL diagnostic does *not* make.

First, SDL does not claim that EEG FMs are without value. The anchored-contrast regime (§8.3) is where EEG FMs deliver meaningful rescue, and this regime includes well-established clinical targets (dementia, epilepsy, certain cognitive-load paradigms) whose neural anchors are mature. Where the anchor exists, FM pretraining appears to be a productive investment.

Second, SDL does not claim that the contrast-bounded ceiling is a permanent limit. A future finding that establishes a robust per-recording within-subject EEG correlate of DASS or DSS — or an intervention that induces such a correlate — would change the contrast-bounded classification of UCSD Stress to an anchored-contrast one, and the SDL diagnostic would correctly predict that FM fine-tuning would begin to rescue on the newly-anchored task. SDL is a statement about the current state of the anchoring literature, not a metaphysical limit on EEG FMs.

Third, SDL does not claim that the two-regime picture is exhaustive. Our controlled paired comparison (§6) spans one strong anchor (alpha-ERD) and one absent anchor (DSS). Intermediate-anchor regimes — tasks with weak or preliminary literature support — are outside our tested range, and whether they produce intermediate FM gains, noisy FM behaviour, or bimodal outcomes is a question we leave to future work. The pre-benchmark diagnostic we propose in §9.3 is deliberately binary (anchor / no-anchor); refining it into a graded anchor-strength score is a productive next step.

### 9.5 Implications for pretraining-corpus and loss design

A secondary implication worth flagging is that SDL suggests pretraining objectives specifically designed to *reduce* subject-dominance in the frozen representation — subject-adversarial pretraining, subject-invariant contrastive losses, cross-subject covariance alignment — would, if successful, shift the contrast-bounded regime upward even in the absence of an external label anchor. Multi-dataset Joint Pre-training of Emotional EEG (NeurIPS 2025, SUSTech) is an early entry in this direction. SDL predicts that such objectives should show the largest gains specifically on contrast-bounded benchmarks (where our current FMs are stuck) and smaller gains on contrast-anchored benchmarks (where current FMs already reach rescue). This is a concrete, falsifiable prediction that future pretraining work can test against our SDL diagnostic, and is a productive generative direction for the EEG-FM research programme beyond architecture iteration.

§10 summarises the specific empirical boundaries of the present study.

---

*Draft status: §9 ~1,150 words. Ready for review.*
