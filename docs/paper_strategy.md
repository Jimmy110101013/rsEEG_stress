# Paper Strategy

**Date**: 2026-04-17 (synced to tex) / 2026-04-15 (ID migration) / 2026-04-14 (full rewrite)
**Status**: Active — SDL narrative implemented in tex; HHSA added as Discussion evidence
**Companion docs**: `findings.md` (F-A…F-E + F-HHSA paper claims), `methodology_notes.md` (guardrails + internal notes + archived), `TODO.md` (D1/D2 open decisions), `related_work.md`, `eta_squared_pipeline_explanation.md`

This document captures the current paper narrative, structure, and figure plan.
**The tex files (`paper/sections/*.tex`) are authoritative for the actual paper content.**
This doc tracks strategy and maps claims to sections.

> **Sync note (2026-04-17)**: The old "three-pillar" narrative (Methodology traps /
> Representation diagnosis / FT interaction) is **superseded** by the SDL
> (Subject-Dominance Limit) framing now implemented in the tex. F-C (FT direction
> reversal) is not in the tex main results. See §2 below for the current structure.

---

## 1. Title

**Subject Dominance Limits: A Mechanistic Diagnostic for EEG Foundation Models**

(Updated in tex. Old title "Beyond Accuracy: What EEG Foundation Models Encode,
and Why Fine-Tuning Direction Depends on Model × Dataset Interactions" was
superseded when F-C direction reversal was removed from the main narrative.)

Venue: **IEEE TNSRE** (primary target). Backup: J. Neural Engineering,
NeurIPS D&B Track, TMLR.

---

## 2. Paper structure (matches tex as of 2026-04-17)

The paper has **three Results sections + Discussion**, not three pillars.
The logic is a single deductive chain:

```
§3.1 Subject dominance is universal (F-A)
  → representations are subject-dominated in all 12 FM×dataset cells
  → any classifier must work against this structural constraint
       ↓
§3.2 The anchor-contrast diagnostic (F-D + F-NEURO)
  → CV gap intro (F-B, brief)
  → Paired experiment: EEGMAT rescue (0.73) vs Stress fail (≤ chance)
  → Band-stop corroboration: alpha convergence (EEGMAT) vs divergence (Stress)
  → Architecture ceiling: 7 architectures in 0.44-0.58 band
  → Conclusion: contrast strength, not architecture, governs rescue
       ↓
§3.3 FM value within the ceiling (partial F-C)
  → Frozen LP +5pp over classical (real but modest)
  → Two regimes: anchored (FT adds rescue) vs bounded (FT doesn't help)
  → No FT direction reversal narrative
       ↓
Discussion: SDL diagnostic protocol + TDBRAIN intermediate case + HHSA gradient
```

### What is NOT in the tex (archived from old narrative)

- **F-C direction reversal** (LaBraM vs REVE opposite on ADFTD/TDBRAIN) — valid data but removed to avoid defocusing the SDL argument
- **Master table cross-dataset FT taxonomy** — §3.3 uses a simplified 3-dataset table (Stress/EEGMAT/ADFTD), not the full 3×4 grid
- **Stress as power-floor case study** (old §8) — absorbed into §3.2 as the "bounded" arm of the paired comparison

---

## 3. Paper structure (synced to tex 2026-04-17)

### §1 Introduction
- Hook: Wang et al. (2025) 0.9047 BA → our subject-level 0.52–0.58
- Define SDL: FM performance bounded by within-subject neural contrast / subject variance ratio
- Fig 1 (SDL Spectrum): datasets ordered by anchor strength, showing the two regimes
- Pre-benchmark diagnostic question: does the label have a within-subject neural contrast?

### §2 Related work + §3 Methods
- Datasets: UCSD Stress (primary), EEGMAT (paired comparator), ADFTD + TDBRAIN (variance decomposition coverage)
- Three sample-size conventions (70/55/14) clearly documented
- Contrast-strength anchoring protocol (3 dimensions: published correlate, empirical cluster, locus concordance)
- Variance decomposition: pooled fractions, RSA, cluster bootstrap

### §3.1 Results — Subject-dominated frozen representations (F-A)
- Variance atlas: 12/12 FM×dataset cells show subject >> label (10–50×)
- Three independent measurements: pooled variance, RSA, cluster bootstrap
- Fig (variance_atlas): 3-panel triangulation
- Table (subject_atlas): all 12 cells with exact fractions

### §3.2 Results — The anchor-contrast diagnostic (F-D core)
- **CV gap** (intro): trial vs subject gap on Stress (brief, Fig cv_gap)
- **Permutation null**: LaBraM real FT inside null distribution (p=0.70)
- **Paired experiment** (core):
  - Arm 1 EEGMAT: LaBraM 0.731, REVE 0.727, CBraMod 0.620 — rescue succeeds
  - Arm 2 Stress longitudinal DSS: all 9 FM×classifier cells ≤ chance — rescue fails
  - "Projection, not rewrite": label fraction doesn't increase under FT
- **Within-subject trajectory geometry**: EEGMAT positive cosine (0.06–0.15), Stress zero
- **Band-stop corroboration** (F-NEURO integrated): alpha convergence on EEGMAT, divergence on Stress
- **Architecture ceiling**: 7 architectures (3k–1.4B params) all in 0.44–0.58 band on Stress

### §3.3 Results — FM value within the ceiling
- Classical vs Frozen LP vs FT across Stress/EEGMAT/ADFTD
- Two regimes: anchored (FT adds rescue) vs bounded (frozen LP +5pp, no FT rescue)
- Note: simplified table, NOT the full 3×4 master table

### §4 Discussion
- **SDL two regimes** + pre-benchmark protocol (3-step)
- **TDBRAIN intermediate case**: empirical cluster survives but wrong locus → 19pp FM spread
- **HHSA contrast gradient** (F-HHSA, new): model-independent evidence that contrast is a continuum, not binary
- What SDL does not claim (3 explicit non-claims)
- Implications for pretraining design (subject-adversarial objectives)

### §5 Limitations
- Cohort size + single-seed fragility
- Single anchor on rescue arm (→ HHSA partially addresses this)
- Architecture panel breadth

### §6 Conclusion

---

## 4. Figure & table plan (synced to tex 2026-04-17)

The tex already has figures referenced. Below maps to what exists in the tex.

### Main figures (as referenced in tex)

| Fig | tex label | Title | Source | Status |
|---|---|---|---|---|
| 1 | `fig_sdl_spectrum` | SDL Spectrum — datasets ordered by anchor strength | `paper/figures/main/fig_sdl_spectrum.pdf` | **IN TEX** |
| 2 | `fig_variance_atlas` | 3-panel variance atlas (fractions + RSA + bootstrap) | `paper/figures/main/fig_variance_atlas.pdf` | **IN TEX** |
| 3 | `fig_cv_gap` | Trial vs subject CV gap on Stress | `paper/figures/main/fig_cv_gap.pdf` | **IN TEX** |
| 4 | `fig_within_subj_direction` | Within-subject trajectory (UMAP + cosine) | `paper/figures/main/fig_within_subject_direction.pdf` | **IN TEX** |
| 5 | `fig_paired_contrast` | Paired diagnostic: EEGMAT rescue vs Stress fail | `paper/figures/main/fig_paired_contrast.pdf` | **IN TEX** |
| 6 | `fig_band_diagnostic` | Band-stop cross-model consensus | `paper/figures/main/fig_band_diagnostic.pdf` | **IN TEX** |
| 7 | `fig_ceiling` | Architecture ceiling forest plot (7 architectures) | `paper/figures/main/fig_ceiling.pdf` | **IN TEX** |

### Main tables (as referenced in tex)

| Table | tex label | Title | Status |
|---|---|---|---|
| 1 | `tab_subject_atlas` | Variance decomposition 12 cells | **IN TEX** |
| 2 | `tab_stress_longitudinal` | Within-subject DSS classification (9 cells) | **IN TEX** |
| 3 | `tab_architectures` | 7 architectures BA on Stress | **IN TEX** |
| 4 | `tab_fm_value_cross_dataset` | Classical vs Frozen vs FT (3 datasets) | **IN TEX** |

### Figures NOT in tex (archived or available for reviewer response)

| Old # | Title | Status |
|---|---|---|
| Cross-dataset FT taxonomy (F-C) | 3×4 master table direction reversal | **Not in tex** — available if reviewers ask |
| Matched-N curves | LaBraM-only N-invariance | **Not in tex** |
| Interpretability topomap + band RSA | Spatial + correlational (exp14) | **Not in tex** (only band-stop causal in tex) |

### Potential additions

| Fig | Title | Source | Status |
|---|---|---|---|
| Discussion | HHSA holospectral contrast gradient (4 datasets) | `results/hhsa/cross_dataset_comparison/` | **NEW** — condition contrast t-maps + summary bar chart |
| Discussion | Sleep Dep FM BA vs HHSA contrast (if exp_newdata confirms) | Pending exp_newdata | **PENDING** |

---

## 5. Status (synced to tex 2026-04-17)

| Section | Status | Notes |
|---|---|---|
| §1 Introduction | **Written** | SDL framing, Fig 1 SDL Spectrum |
| §2 Related work | **Written** | |
| §3 Methods | **Written** | Anchoring protocol, variance decomposition, HP selection |
| §3.1 Variance Atlas | **Written** | Fig variance_atlas, Table subject_atlas |
| §3.2 Anchor-Contrast Diagnostic | **Written** | Core section — CV gap, paired comparison, band-stop, ceiling |
| §3.3 FM Value Within Ceiling | **Written** | Table fm_value_cross_dataset |
| §4 Discussion | **Written** | SDL protocol, TDBRAIN intermediate, pretraining implications |
| §5 Limitations | **Written** | Cohort size, single anchor, architecture panel |
| §6 Conclusion | **Written** | |
| Appendices | **Written** | PSD anchor, ADFTD split, perm null |

### Open items

- **HHSA paragraph in Discussion**: Add holospectral contrast gradient evidence (F-HHSA) as empirical support for graded anchor-strength refinement. Figures ready in `results/hhsa/cross_dataset_comparison/`.
- **Sleep Dep FM results** (exp_newdata running): If FM performance on Sleep Dep aligns with HHSA ranking, add as third data point in Discussion.
- **Advisor alignment**: Confirm SDL framing is acceptable for TNSRE submission.

---

## 6. Precedent venues and papers

### EEG-specific
- Roy et al., "Deep learning-based EEG analysis: systematic review" (J.
  Neural Eng. 2019) — DL gives marginal improvement over classical features.
- Brain4FMs / EEG-FM-Bench / AdaBrain-Bench (2025–2026) — independent
  benchmark convergence, already cited in `related_work.md`.

### Methodology / negative-result precedents
- Zeng et al., "Are Transformers Effective for Time Series Forecasting?"
  (AAAI 2023 outstanding paper).
- Musgrave et al., "A Metric Learning Reality Check" (ECCV 2020).
- Dodge et al., "Show Your Work" (EMNLP 2019) — HP-tuning artifacts,
  same flavor as our F08.

### Venues welcoming this work
- **IEEE TNSRE** (primary) — EEG-friendly, rigor-valued, interpretability
  matters.
- J. Neural Engineering, NeurIPS D&B Track, TMLR.
- "I Can't Believe It's Not Better" NeurIPS workshop as fallback.

---

## 7. Ruled out — do not re-propose

- Subject-adversarial GRL/DANN / LEAD-style subject CE loss (F12)
- `--label subject-dass` (F07)
- Within-subject longitudinal DSS reframing (F14 negative)
- Sparse-label-subspace hypothesis (refuted)
- Spectral-guided FiLM conditioning (dropped; bounded experimental budget
  reallocated to cross-dataset taxonomy validation)
- Canonical 0.656 as "LaBraM ceiling" (F08 cuDNN noise)

---

## 8. Historical framings (archived, do NOT cite)

| Version | Framing | Superseded by |
|---|---|---|
| 2026-04-07 | "Method-independent ceiling at 0.66" — stress as main result | F07 (subject-dass deprecated), F16 (classical also OR artifact) |
| 2026-04-10 | "Stress erosion main finding; LaBraM−8pp the headline" | F17 (cross-dataset taxonomy is the real story), F06/F19 (null borderline) |
| 2026-04-11 | "FT mode driven by label biology" | F17 (model × dataset interaction, not label biology alone) |
| All pre-04-14 | "Within-subject longitudinal highest leverage" | F14 (all 3 FMs fail on within-subject DSS) |
| All pre-04-14 | "FiLM as §5.2" | Scope-cut; not run |
| 2026-04-14 | "Three-pillar narrative (Methodology traps / Rep diagnosis / FT interaction)" | SDL framing in tex — single deductive chain, not three pillars |
| 2026-04-15 | "F-C direction reversal as main result" | Removed from tex; ADFTD/TDBRAIN used only for variance decomposition + Discussion |

---

## 9. Mindset note

The paper's contribution is the **SDL diagnostic**: a mechanistic explanation
for why EEG FMs sometimes fail, grounded in the ratio of task contrast to
subject variance. Three independent 2026 benchmarks converge on the empirical
observation (F11); our addition is the mechanistic explanation (subject
dominance via variance decomposition) and the **paired controlled comparison**
(EEGMAT rescue vs Stress fail) that isolates contrast strength as the
governing variable.

The tex implements this as a single deductive chain (subject dominance →
paired diagnostic → architecture ceiling → FM value within ceiling), not
as multiple pillars or a cross-dataset taxonomy.
