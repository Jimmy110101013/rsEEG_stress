# Paper Strategy & Reframing Notes

**Date:** 2026-04-07
**Status:** Active — start drafting in parallel with FiLM experiments
**Companion docs:** `progress.md` (results), `related_work.md` (citations)

This document captures the strategic decisions about paper narrative, the
honest reframing of our results, and the experimental directions still worth
trying. Written as a snapshot so we don't lose the reasoning under deadline
pressure.

---

## 1. The Honest Reframe (read this first)

### What it feels like
"Resting-state EEG can't predict stress. We failed."

### What the data actually says
| Method | Subject-level BA (70-rec) | Above chance |
|---|---|---|
| Classical RF on band power | **0.666** | +0.166 |
| Fine-tuned LaBraM | **0.656** | +0.156 |
| Chance | 0.500 | — |

Two completely different methods (hand-crafted features and a 200M-param
pretrained transformer) **converge to the same number above chance**. If the
signal didn't exist, both would collapse to 0.5. They don't.

→ **There IS a real cross-subject resting-state stress signature in our data.**
It is small-to-moderate (~0.16 above chance with N=17 subjects), and it is
saturated by classical features. The ceiling is **dataset-limited and
framing-limited, not method-limited.**

### The correct one-line summary
> "There exists a modest cross-subject resting-state stress signal (~0.66 BA),
> and both classical features and 200M-parameter foundation models converge to
> it. The ceiling is method-independent under the binary cross-subject framing."

This is **not a negative result** — it is a *characterization* of where the
ceiling sits and why FMs cannot exceed it. That distinction matters for the
paper, for our motivation, and for what experiments are still worth running.

---

## 2. Why Drafting Now (not later) Is the Right Move

### Decision: start drafting immediately, even though FiLM is unfinished.

**Rationale:**
1. **The diagnosis story is fully written-up-able with current data.** 5 figures
   + 4 tables of evidence are already locked.
2. **Drafting reveals which experiments matter.** Writing §4.3 will show
   exactly which TDBRAIN comparison plot we need — and only that one.
3. **Negative results decay if undocumented.** Adversarial λ-sweep config
   details are still fresh; will be stale in a month.
4. **Iteration is asymmetric.** Slotting in a paragraph when FiLM lands = 30
   min. Reconstructing the intro under deadline = 2 days.
5. **FiLM is one section, not the foundation.** Treat it as a bounded
   experimental hypothesis with two write-up branches, both publishable.

### Lock-vs-Leave-Open partition

| Section | Status | Evidence source |
|---|---|---|
| Abstract (provisional) | Lock | progress.md §1 |
| §1 Introduction | Lock | — |
| §2 Related Work | Lock | docs/related_work.md (already exists) |
| §3 Methods | Lock | — |
| §4.1 Trial vs subject CV gap | Lock | progress.md §4.2 |
| §4.2 Variance decomposition (η²) | Lock | progress.md §4.3, fm_diagnosis/ |
| §4.3 Cross-dataset signal strength | Lock after TDBRAIN FT | cross_dataset/ notebook |
| §4.4 Classical baseline parity | Lock | classical_subject-dass run |
| §4.5 Frozen vs FT η² shift | Lock | progress.md §4.4 |
| §5.1 Failed mitigation: adversarial | Lock | progress.md §6 |
| **§5.2 Spectral-guided FiLM** | **OPEN** | — pending |
| §6 Discussion | Lock FiLM-agnostic version, refine after FiLM lands | — |

Either FiLM outcome publishes:
- **If FiLM works:** "Diagnosis + working fix" upgraded paper.
- **If FiLM doesn't:** Joins §5.1 as another negative result, *strengthening*
  F4 ("the diagnostic signal is too weak to amplify").

---

## 3. Paper Title Options (constructive framings)

Avoid takedown framings. Use titles that report what we found AND point at
what to do next.

1. **"Subject Identity Dominance Bounds Foundation-Model Performance on
   Resting-State Affective EEG: Diagnosis and Implications for Personalized
   Modeling"** (preferred — implies productive next step)

2. "Cross-Subject Resting-State Stress Detection Hits a Method-Independent
   Ceiling: A Diagnosis of Subject Identity Dominance in EEG Foundation
   Models"

3. "Signal Strength Determines EEG Foundation Model Utility: A Cross-Dataset
   Diagnosis of Subject Identity Dominance" (current progress.md §8 version —
   slightly more "takedown" framed)

The first framing closes with "personalized modeling is the productive path
forward" — which bridges naturally to the within-subject experiments below.

---

## 4. Precedent: Negative / "Baselines Win" Papers That Got Published

### High-impact ML
- **Zeng et al., "Are Transformers Effective for Time Series Forecasting?"**
  (AAAI 2023, outstanding paper) — DLinear crushes the entire transformer
  time-series literature.
- **Musgrave et al., "A Metric Learning Reality Check"** (ECCV 2020) — Years
  of metric losses give ~zero gain over simple contrastive baseline.
- **Lipton & Steinhardt, "Troubling Trends in ML Scholarship"** (2018) — Pure
  methodological critique paper, no model. Hugely influential.
- **Dodge et al., "Show Your Work: Improved Reporting of Experimental
  Results"** (EMNLP 2019) — Reported gains are mostly hyperparameter-tuning
  artifacts. Same flavor as our trial-leakage point.

### EEG-specific
- **Roy et al.**, "Deep learning-based electroencephalography analysis: a
  systematic review" (J. Neural Eng. 2019) — 156 papers reviewed, DL gives
  marginal improvement over classical features. Essentially our F3.
- **Brain4FMs / EEG-FM-Bench / AdaBrain-Bench** (2025–2026) — Three
  independent benchmark papers all reporting EEG FMs fail or barely move on
  cross-subject affective/cognitive tasks. **Already cited in
  related_work.md.** We are part of a 2026 consensus, AND we have the
  mechanistic explanation (η² subject dominance) that those benchmarks lack.

### Venues that explicitly welcome this kind of work
- **NeurIPS Datasets & Benchmarks Track** — designed for it
- **TMLR** (Transactions on ML Research) — explicitly accepts negative/null
  results, no novelty bar
- **"I Can't Believe It's Not Better" NeurIPS workshop** — literally the name
- **J. Neural Engineering**, **IEEE TNSRE** — EEG-friendly, value rigor
- **Nature Sci. Data** — descriptive/methodological

---

## 5. Directions Where the Hope Is *Legitimately* Alive

The 65% ceiling is for one specific framing: **binary, single-recording,
generic-feature, cross-subject zero-shot prediction.** None of the following
contradict our current findings, and all are experimentally tractable.

### 5.1 Within-subject longitudinal modeling ⭐ **highest leverage**
- We have multiple recordings per subject. Reframe the question:
  - From "is this person stressed?" (cross-subject)
  - To "has this person's EEG drifted from their own baseline?" (within-subject)
- Matches the *actual clinical use case* (pre-diagnose stress in *this
  individual* over time).
- Sidesteps subject identity dominance entirely — subject identity *is* the
  reference frame, not the noise.
- **Most underexplored direction in our dataset and aligns directly with the
  research vision.**

### 5.2 Personalized fine-tuning (few-shot adaptation)
A handful of calibration samples per subject. Standard in clinical BCIs.

### 5.3 Spectral-guided FiLM conditioning
Original CLAUDE.md plan. Still worth running. Even +3 BA is a publishable
improvement on top of the diagnosis.

### 5.4 Continuous label regression
Use the actual DASS score or trajectory rather than thresholded
normal/increase. Binary thresholding throws away most of the variance.

### 5.5 Multimodal fusion
EEG + HRV / actigraphy if available. EEG's modest signal may be detectable
when combined with autonomic markers.

### 5.6 Self-supervised contrastive pretraining on our own data
Subject-as-positive contrastive — explicitly *encode* subject as a feature,
then condition on it. Flip the diagnosis on its head.

---

## 6. The Half-Day Experiment to Run BEFORE Drafting

**Goal:** Within-subject longitudinal proof of concept on existing data.

**Procedure:**
1. For each subject, compute their own baseline EEG features from the
   earliest recording (or recordings labeled "normal").
2. For each follow-up recording, compute the deviation from that personal
   baseline (Δ-features).
3. Predict stress *change* (not absolute label) from the deviation features.
4. Use the same StratifiedGroupKFold protocol for fairness — but the model
   sees Δ-features, not absolute features.

**Decision tree:**
- **If BA ≥ 0.72** → reframe paper title to lead with the within-subject
  finding: *"Generic FMs fail because the question is wrong; reframing as
  deviation-from-self produces a working detector."* This becomes the paper
  we *actually want to write.*
- **If BA stays at 0.66** → still informative. The diagnostic paper publishes
  unchanged. We've eliminated one more hypothesis cleanly.

**Cost:** ~half a day. Extremely high information-per-hour.

---

## 7. Concrete Next Actions (in order)

1. **Tonight:** Wait for TDBRAIN FT (PID 609824) to finish. Update notebook
   cell 22 with actual results dir, regenerate cross-dataset figures.
2. **Tomorrow morning (~half day):** Run §6 within-subject longitudinal
   experiment. Single notebook, no infra changes needed.
3. **Tomorrow afternoon (~2h):** Create `paper/` directory with `main.tex` or
   `paper/draft.md`. Lock §1 + §2 + §3 outline. Port abstract from
   progress.md §1 verbatim as a stub.
4. **Day 2 (~3h):** Write §4.1, §4.2, §4.4 from existing tables. Use
   placeholder figure boxes — don't generate final figures yet.
5. **Day 2–3:** Lock §4.3 (cross-dataset, including TDBRAIN row) and §4.5
   (frozen vs FT).
6. **Day 3:** Write §5.1 (adversarial negative result). While writing it,
   note exactly what control conditions FiLM needs to be comparable.
7. **Then start FiLM experiments** with a clear evaluation protocol matching
   the existing tables, so results slot in.
8. **§5.2 + Discussion** finalize after FiLM converges (positive *or*
   negative).

---

## 8. Statistical Hardening (do once, cite everywhere)

Before drafting, run these to harden every claim in §4 against reviewer
pushback. Each is 1–2 hours of work.

- **Permutation test on η²** — to claim "11–13× is significant, not noise"
- **Bootstrap CIs on subject-level BA across folds** — for every table cell
- **Sign test on the trial vs subject gap across the 3 FMs** — for §4.1

These are listed as TODO in `progress.md §9`. Move them up in priority — they
make every table reviewer-proof.

---

## 9. Emotional / Mindset Note

Hitting a ceiling is not failing. **Understanding *why* a task is hard, with
quantitative diagnosis, is genuine science.** The η² methodology *is* a
contribution. Three independent 2026 benchmarks already say what we're
saying — we're not lonely; we have the mechanistic explanation they lack.

The hope that resting-state EEG can pre-diagnose stress is **compatible with
the data.** RF 0.666 is the evidence the signal exists. The next move is not
to abandon the dream — it's to reframe the question (within-subject
longitudinal) so that the modest signal can actually be detected.

We are not abandoning the vision. We are *correctly diagnosing why the
obvious method doesn't reach it,* and pointing at the right next thing.
