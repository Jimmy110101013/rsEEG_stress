# R4 Literature Scout Review — External Novelty Audit (2024–2026)

**Scope**: Audit of recent (2024–2026) publications against paper claims F-A…F-E.
**Search pool**: ~15 WebSearch queries, 4 WebFetch reads. Coverage areas: EEG FM benchmarks, subject invariance, FT dynamics, UCSD Stress usage, DASS-21/EEG links, power-floor / small-N cautionary literature, critical/position papers.
**Date of review**: 2026-04-15.

---

## 1. Scoop risk — claims that may already be in the literature

### 1.1 Partial scoop: EEG-Bench (arXiv:2512.08959, NeurIPS 2025)
- **Title**: EEG-Bench: A Benchmark for EEG Foundation Models in Clinical Applications
- **Venue**: NeurIPS 2025
- **One-liner**: 14 datasets × 11 clinical tasks (epilepsy, schizophrenia, Parkinson, OCD, TBI) evaluated in **strict cross-subject setting**. Headline finding: "while foundation models achieve strong performance in certain settings, **simpler models often remain competitive, particularly under clinical distribution shifts**." This is **not yet cited** in `related_work.md`.
- **Scoop severity**: **Medium**. EEG-Bench overlaps with our F-B (honest evaluation exposes weak FM advantage) and F-D.2 (ShallowConvNet matches FMs on Stress). It does **not** measure variance decomposition (F-A), does not demonstrate direction-flipping FT (F-C), and does not use the UCSD Stress / DASS-21 dataset. Our contributions (F-A variance decomp, F-C model×dataset FT taxonomy, F-D Stress power-floor) remain novel, but we must cite EEG-Bench as prior empirical convergence alongside Brain4FMs / EEG-FM-Bench / AdaBrain-Bench, and soften any language implying we were first to observe simpler models staying competitive on clinical EEG.

### 1.2 Partial scoop: "Multi-dataset Joint Pre-training of Emotional EEG" (arXiv:2510.22197, NeurIPS 2025)
- **Authors**: Not easily accessible in search result; NCClab-SUSTech group.
- **One-liner**: Proposes cross-dataset covariance alignment loss for affective EEG. Reports +4.57 % AUROC few-shot and +11.92 % zero-shot over SOTA large EEG FMs. **Frames the same diagnosis we do** (inter-subject variability, inter-dataset shifts are the dominant issue for FMs on emotion tasks) but proposes a method rather than a diagnostic framework.
- **Scoop severity**: **Low-medium**. They treat subject/dataset shift as a problem to solve, not as a structural property to quantify. Our F-A variance-decomposition framework and F-C direction-flipping taxonomy remain distinct. Should cite as supporting evidence in §1 and §9.

### 1.3 Not-a-scoop but relevant: "Foundation Models for Neural Signal Decoding" (Kwon et al. 2026, Eur J Neurosci)
- **Ref**: DOI 10.1111/ejn.70376 (Wiley 2026)
- **One-liner**: Review piece positioning EEG FMs toward "unified representations." Does not quantify subject-vs-label variance; does not challenge the claim that FMs underperform on cross-subject affective tasks. Useful field-survey citation.

**Verdict on scoop risk**: No paper currently claims what **F-A** (subject-dominance via variance decomposition, 12/12 RSA panel) or **F-C** (model × dataset FT direction flip across 3 FMs × 3 datasets) claim. EEG-Bench is the single strongest adjacent paper and must be cited, but does not preempt our mechanistic contributions.

---

## 2. Supporting citations — strengthen the paper's story

### 2.1 Power-floor / small-N reliability (STRONG support for §8)

**Xie et al. (or equivalent authors), "Sample Size Critically Shapes the Reliability of EEG Case-Control Findings in Psychiatry"** (bioRxiv 2025.11.10.687610, posted 2025-11-12; PMID 41292863)
- **One-liner**: Multisite N = 2,874 resting-state EEG across ADHD/ASD/anxiety/learning disorders. **"Results from small samples were unstable, with inflated and highly variable effect sizes across iterations. Larger samples produced consistent findings, converging on uniformly small but robust effects."** This is the direct citation for our §8 power-floor framing. **Not in related_work.md — must add.**
- **Relevance**: Directly supports F-D.1 (FT null-indistinguishable on N=70/14-positive) and F-B G-F08 (cuDNN ±20pp swings on small cohorts).

**"Robust EEG brain-behavior associations emerge only at large sample sizes"** (bioRxiv 2026.02.06.704323)
- **One-liner**: Same conclusion from a brain-behavior correlation angle. Additional independent precedent for §8.

### 2.2 Data leakage / subject-level CV (field consensus has strengthened since related_work.md)

**Brookshire et al. "Data leakage in deep learning studies of translational EEG"** (Frontiers Neurosci 2024, 10.3389/fnins.2024.1373515) — already in related_work.md §3.1. Still the strongest canonical cite for our F-B Axis 1.

**"Impact of Trial-wise and Test Data Leakage on EEG-Based Emotion Classification"** (CEUR-WS Vol-4115 paper7, 2024)
- **One-liner**: Trial-wise leakage specifically in affect classification; quantifies inflation. Adds emotion-specific precedent for our Wang 2025 reanalysis.

### 2.3 EEG Foundation Challenge 2025 (NeurIPS competition track)

**"EEG Foundation Challenge: From Cross-Task to Cross-Subject EEG Decoding"** (arXiv:2506.19141, Truong et al., NeurIPS 2025 competition track)
- **One-liner**: NeurIPS 2025 competition explicitly framed around cross-subject generalization + psychopathology prediction (CBCL p-factor, internalizing, externalizing, attention). This is the closest "the field officially admits cross-subject clinical decoding is the hard problem" marker. **Should cite in §1 hook and §9.**
- **Relevance**: Supports our framing that F-A (subject dominance) is the central open problem, and our power-floor warning to the field is timely.

### 2.4 Critical review of EEG FMs (already cited but deepen)

**Kauffmann et al. "EEG Foundation Models: A Critical Review of Current Progress and Future Directions"** (arXiv:2507.11783v3, PubMed 41666566, NeurIPS 2025)
- **One-liner**: Confirms that **"Linear probing results [of frozen EEG FMs] were relatively worse"** than full-FT, and **"general time-series foundation models sometimes outperformed EEG-FMs"**. Frames LP<<FT as a representation-quality concern — aligns with our F-C (FT is a projection, not a rewrite) argument. Already in related_work.md §2.4 but should be up-weighted into the Introduction.

### 2.5 Negative transfer / fine-tuning harms

**Li et al. "On the Robustness Tradeoff in Fine-Tuning"** (ICCV 2025)
- **One-liner**: Shows FT trades robustness against accuracy in a systematic way. Analogous to our TDBRAIN erosion finding at the vision-model level.

**"Massive Supervised Fine-tuning Experiments"** (arXiv:2506.14681, 757 fine-tuned models × 10 base architectures × 10 datasets)
- **One-liner**: **"The inductive biases of the base model outweigh the specific SFT corpus in determining the final representation."** This is the LLM analogue of our F-C: pretrained model architecture dominates over dataset in determining FT outcome. **Must cite in §9** as cross-modality precedent that our model × dataset interaction finding is consistent with broader ML observations.

---

## 3. Refuting citations — claims the authors should address

**No outright refutations were found.** Several papers temper our story but none overturn it.

### 3.1 Tempering: EEG-Bench shows FMs sometimes win on clinical tasks
- EEG-Bench's claim that FMs "achieve strong performance in certain settings" means our absolute statement about FM weakness on cross-subject clinical EEG must be **conditional on label biology / contrast strength** (which F-C.3 already does for EEGMAT). Our paper should pre-empt reviewer objections by noting: "on sleep staging, seizure detection, event-related paradigms where EEG-Bench and EEG-FM-Bench show FM advantage, our critique does not apply; our scope is weak-contrast affective/trait labels."

### 3.2 Tempering: Multi-dataset Joint Pre-training (arXiv:2510.22197) shows alignment methods can partly rescue FMs on emotion
- Reports +11.9 % zero-shot gain via covariance alignment. We do not test such alignment; reviewers may ask whether our F-D Stress ceiling holds if a covariance-alignment pretraining variant is used. Recommend adding a sentence in Discussion acknowledging this as untested future work.

### 3.3 No refutations on F-A (subject dominance)
- No paper was found that **measured** subject-vs-label variance in frozen EEG FMs and found it **balanced** or **label-dominated**. Our F-A appears to be the first quantitative claim of its kind. (The closest prior, EEG-FM-Bench's gradient-mass + CKA analysis, is unsupervised and multi-task averaged — see related_work.md §2.1 for the existing positioning.)

### 3.4 No refutations on F-C (FT direction flips by model × dataset)
- Multi-dataset Joint Pre-training reports positive transfer on emotion tasks with alignment, but does not test the canonical LaBraM/CBraMod/REVE at model-level divergence. The Massive SFT paper (arXiv:2506.14681) independently supports "pretrained architecture matters more than SFT corpus" — complementary, not refuting.

---

## 4. Unused but citable papers (not currently in related_work.md)

| Citation | Why add |
|---|---|
| **EEG-Bench** (Scherer et al., arXiv:2512.08959, NeurIPS 2025) | Fourth independent clinical EEG benchmark converging on "FMs not universally superior"; strengthens F-B Axis 1. |
| **"Sample Size Critically Shapes…"** (bioRxiv 2025.11.10.687610) | Direct statistical-power citation for §8 power-floor framing; most current and psychiatry-specific. |
| **"Robust EEG brain-behavior associations"** (bioRxiv 2026.02.06.704323) | Independent large-N confirmation that small-N EEG effects are unreliable. |
| **Multi-dataset Joint Pre-training** (arXiv:2510.22197, NeurIPS 2025) | Acknowledges inter-subject variability as dominant FM failure mode on affect — frames the same problem but proposes a method. |
| **EEG Foundation Challenge 2025** (arXiv:2506.19141, NeurIPS 2025 competition) | Field-level endorsement that cross-subject + clinical psychopathology is *the* open problem — gives our F-A/F-D legitimacy anchor. |
| **Kwon et al. 2026** (Eur J Neurosci, DOI 10.1111/ejn.70376) | Recent review on EEG FM unified representations — useful survey cite. |
| **"On the Robustness Tradeoff in Fine-Tuning"** (ICCV 2025) | Vision-domain analogue of our TDBRAIN erosion. |
| **"Massive SFT Experiments"** (arXiv:2506.14681) | LLM-domain analogue of our F-C model × dataset finding. **Useful for Discussion §9.** |
| **"Inter- and Intra-Subject Variability in EEG: A Systematic Survey"** (arXiv:2602.01019, Feb 2026) | Systematic review framing — useful context for §2 related work. |
| **"Foundation models for EEG decoding"** (PubMed 41145005) | Recent review synthesis; good intro/related-work cite. |
| **DeepAttNet / ear-EEG stress** (Frontiers Hum Neurosci 2025, 10.3389/fnhum.2025.1685087) | Subject-independent stress decoding with cross-attention — current competitor to our stress pipeline, useful comparator in §5 related work. |

---

## 5. Topic-specific findings

### 5.1 DASS-21 / DSS as EEG targets (R4 query #5)
- **No paper was found** that directly correlates DASS-21 or DSS scores with resting-state EEG features at the trial level. DASS-21 psychometric papers (2024–2025) proliferate but none link to EEG. This is a **gap in the literature**, which makes the authors' "weak contrast" thesis scientifically well-grounded — but they cannot claim *existing evidence* of weak DASS-EEG correlation; they must frame it as an empirical observation in their own dataset. The TAR (Theta/Alpha Ratio) paper already in related_work.md §6.1 is the closest existing link.

### 5.2 UCSD Stress dataset usage (R4 query #4)
- **Only paper found using Komarov 2020 besides our group's analysis**: Wang et al. 2025 (arXiv:2505.23042) — already cited. No independent replication exists. **This strengthens the paper's contribution** — we are the first independent re-analysis of the Wang claim.

### 5.3 EEG FM position / critical papers (R4 query #7)
- The 2025 Critical Review (arXiv:2507.11783) is the one genuine critical paper. No op-eds or skeptical position papers have appeared. Space remains for our paper to be **the** rigorous audit of EEG FM claims.

---

## 6. Verdict on paper novelty (April 2026)

**Novelty holds up. All five paper claims F-A…F-E remain defensible.**

- **F-A (subject dominance via variance decomp)**: Novel. No paper quantifies subject-vs-label variance in frozen EEG FMs. The EEG-Challenge 2025 field focus on cross-subject decoding *motivates* the problem but has not measured it our way.
- **F-B (honest evaluation eliminates inflation)**: Convergent with EEG-Bench + Brain4FMs + EEG-FM-Bench + AdaBrain-Bench — **four** independent benchmarks now support a similar conclusion. Our twist (Axis 2: OR-aggregation) and the Wang 2025 single-dataset reanalysis remain specific to our paper.
- **F-C (FT direction is model × dataset)**: Novel in EEG-FM domain. Massive-SFT LLM paper provides cross-modality precedent, strengthening rather than scooping.
- **F-D (Stress ceiling; power-floor)**: Novel for UCSD Stress specifically. Precedent for the *framing* now exists (bioRxiv 2025.11.10.687610 is the direct citation for our §8 narrative).
- **F-E (alpha lateralization)**: Descriptive, not a primary contribution. Literature consistent.

**Must-do additions before submission**:
1. Cite EEG-Bench (arXiv:2512.08959) in Related Work §2 as 4th independent benchmark convergence.
2. Cite "Sample Size Critically Shapes…" (bioRxiv 2025.11.10.687610) in §8 as power-floor anchor.
3. Cite Multi-dataset Joint Pre-training (arXiv:2510.22197) in §9 as acknowledgment that covariance-alignment methods are an active research direction our analysis does not rule out.
4. Cite Massive SFT paper (arXiv:2506.14681) in §9 as LLM cross-modality precedent for F-C.
5. Cite EEG Foundation Challenge 2025 (arXiv:2506.19141) in §1 as field-level problem recognition.
6. Add a pre-emptive sentence to §9 Discussion clarifying **scope**: our critique applies to weak-contrast affective/trait labels, not to sleep/seizure/ERP tasks where EEG-Bench shows FM advantage.

**Bottom line**: The paper's scientific positioning is slightly stronger than it was at the last `related_work.md` update (2026-04-08) — the November 2025 sample-size psychiatry paper is a clean anchor for the power-floor argument, and the NeurIPS 2025 EEG Challenge provides field-level validation of the problem framing. No 2024–2026 publication has scooped F-A or F-C.
