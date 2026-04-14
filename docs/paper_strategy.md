# Paper Strategy & Reframing Notes

**Date:** 2026-04-07 → reframed 2026-04-10 → updated 2026-04-11
**Status:** Active — multi-model HP sweep completed, narrative updated
**Companion docs:** `progress.md` (§4.6 for 2026-04-10 reclassification), `related_work.md` (citations)

This document captures the strategic decisions about paper narrative. The
original (2026-04-07) "method-independent ceiling" framing was superseded
on 2026-04-10 after the Stress dataset was reclassified from representation
no-op → behavioral erosion. This top section is rewritten; §3 onward is
mostly original (2026-04-07 snapshot) with updated numbers where necessary.

---

## 1. The Honest Reframe (read this first — updated 2026-04-10)

### What we thought (2026-04-07)
"Resting-state stress has a modest ~0.66 BA ceiling, and classical features
match FMs. The ceiling is method-independent — not a failure, just a
characterization."

### What the data actually says (after 2026-04-11 HP sweep)

The 0.656 LaBraM and 0.666 classical RF numbers were both computed under
`--label subject-dass`, an OR-aggregation that turns the task into
subject-identity broadcast. Under the honest per-recording DASS protocol
(matches Lin et al. 2025), multi-seed evaluation, and **comprehensive HP
sweep across 3 FMs** (3 learning rates × 2 encoder LR scales × 3 seeds =
54 total runs):

| Model | Frozen LP (8-seed) | Best FT (3-seed) | Δ | Mode |
|---|---|---|---|---|
| **LaBraM** | **0.605 ± 0.030** | 0.524 ± 0.008 | **−8.1 pp** | erosion |
| **CBraMod** | 0.452 ± 0.030 | **0.548 ± 0.026** | **+8.3 pp** | injection |
| **REVE** | 0.494 ± 0.017 | **0.577 ± 0.041** | **+8.0 pp** | injection |

LaBraM canonical recipe (lr=1e-5, elrs=0.1): 0.443 ± 0.068 (3 seeds),
erosion gap is 16.2 pp. With best HP (lr=1e-4, elrs=1.0), gap narrows
to 8.1 pp — still erosion, but less dramatic.

→ **Erosion is LaBraM-specific on Stress, not model-universal.** LaBraM's
frozen representation is strong (0.605); FT degrades it. CBraMod and REVE
have weaker frozen representations (0.452, 0.494); FT improves them.
This suggests erosion occurs when the frozen representation already
captures the signal well — FT on weak labels then overwrites it.

**Important caveat:** The cross-dataset taxonomy (ADFTD=injection,
TDBRAIN=erosion, EEGMAT=mild injection) was tested with **LaBraM only**.
We do not yet know if CBraMod/REVE would show the same pattern on those
datasets. The claim "FT mode is driven by label–biomarker strength" is
supported by cross-dataset LaBraM evidence but the cross-model evidence
currently only covers Stress.

### The correct one-line summary (2026-04-11)
> "On UCSD Stress, LaBraM fine-tuning degrades a strong frozen
> representation by 8–16 pp (depending on HP), while CBraMod/REVE
> fine-tuning improves weaker frozen representations by ~8 pp. Erosion
> is conditional on frozen representation quality, not universal across
> FM architectures."

### Evidence package
- HP sweep: `results/hp_sweep/20260410_dass/` (54 runs, 3 models)
- LaBraM erosion analysis: `results/studies/exp03_stress_erosion/analysis.json`
- Frozen LP multi-seed: `results/studies/exp03_stress_erosion/frozen_lp/{labram,cbramod,reve}_multi_seed.json`
- Stale numbers (canonical 0.656, classical 0.666, subject-dass anything)
  are retained only as "what we thought before 2026-04-09" historical context.

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
| §4.2 Variance decomposition (η²) | Re-run needed for Stress row (per-rec dass) | progress.md §4.3 + `results/archive/2026-04-05_fm_diagnosis/` (historical) |
| §4.3 Cross-dataset signal strength | Lock (ADFTD/TDBRAIN/EEGMAT) | `results/studies/exp01_cross_dataset_signal/` |
| §4.4 Classical baseline parity | STALE — classical RF needs re-run under per-rec dass | `results/archive/2026-04-05_classical_baselines_subjectdass/` |
| §4.6 Behavioral erosion on Stress (new) | Lock | `results/studies/exp03_stress_erosion/` |
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
