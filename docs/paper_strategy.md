# Paper Strategy

**Date**: 2026-04-14 (full rewrite — supersedes 2026-04-07/10/11 versions)
**Status**: Active — narrative locked pending advisor alignment on D1/D2
**Companion docs**: `findings.md` (F01–F20 evidence), `TODO.md` (D1/D2 open decisions), `related_work.md`, `eta_squared_pipeline_explanation.md`

This document captures the current paper narrative, structure, and figure plan.
It integrates all 20 findings, not just recent results. Historical framings
(FiLM-centric, within-subject longitudinal, Stress-erosion-main, LaBraM-only
taxonomy) are **superseded** and retained only in §8 as context.

---

## 1. Title

**Beyond Accuracy: What EEG Foundation Models Encode, and Why Fine-Tuning
Direction Depends on Model × Dataset Interactions**

Chosen over earlier "…why fine-tuning fails on weak psychiatric labels"
because F17 shows FT direction is **bidirectional** (LaBraM injects on ADFTD,
erodes on TDBRAIN; REVE the opposite), not a universal failure on weak
labels. The new title covers both halves of the paper — representation
diagnosis and FT interaction — without overclaiming.

Venue: **IEEE TNSRE** (primary target). Backup: J. Neural Engineering,
NeurIPS D&B Track, TMLR.

---

## 2. Three-pillar narrative

The paper integrates F01–F20 across three pillars. Stress is not a pillar;
it is a **stress test** that runs through all three pillars as a
statistical-power-floor case study.

| Pillar | Claim | Core findings |
|---|---|---|
| **A. Methodology traps** | EEG FM literature is systematically inflated by subject leakage, OR-aggregated labels, and single-seed cuDNN noise | F01, F07, F08, F11, F16 |
| **B. Representation diagnosis** | Frozen FM representations primarily encode subject identity, not diagnostic signal — but they still beat classical features under honest labels | F02, F03, F10, F13, F16 |
| **C. Fine-tuning interaction** | FT direction is a model × dataset interaction, not a property of either alone; within-subject contrast strength (not just design type) determines whether within-subject framing rescues FM | F09, F14, F17 |

**Stress as cross-pillar stress test** (§8): F05 headline numbers look like
injection/erosion, but **five** independent lines of evidence converge on
power-floor noise: F06 (p=0.70), F19 (p=0.10), F08 (±20pp swings), F20
(ShallowConvNet matches FMs), and cross-model band-importance divergence
(§7). F14 within-subject longitudinal failure is the 5th line, paired with
F09 EEGMAT success to establish "within-subject contrast strength" as the
operative factor. This is the paper's central caveat to the field: benchmark
design must report statistical power before claiming FM superiority.

**Master table** (`paper/figures/source_tables/master_frozen_ft_table.md`):
3 FMs × 4 datasets × 2 phases (frozen LP / FT), subject-level BA, sample std.
This table is the foundation for Results §5 (Pillar C behavioral evidence).

---

## 3. Paper structure

Section numbering below is for the **English paper**. Chinese reading guide
(`paper_narrative_zh.md`) folds Related Work into the Introduction; English
paper keeps them separate per TNSRE convention.

### §1 Introduction
- Hook: Wang et al. (2025, arXiv:2505.23042) — **same lab as the Stress
  dataset paper (Komarov et al. 2020, TNSRE)** — reports 90.47% BA on this
  dataset under trial-level CV. Our subject-level, honest-per-rec-DASS
  reproduction falls to 0.45–0.60 frozen LP / 0.52–0.58 best-HP FT.
- Three questions: (a) What do frozen EEG FMs encode? (b) When does FT help,
  when does it hurt, and why? (c) Can small-sample EEG benchmarks support FM
  superiority claims?

### §2 Related work
- Dataset: Komarov, Ko, Jung (2020, TNSRE 28(4):795) — Taiwan-recorded
  longitudinal resting-state EEG with DASS-21 + DSS per recording.
- EEG FMs: LaBraM (Jiang et al. 2024 ICLR), CBraMod, REVE.
- Wang et al. 2025 — inflated baseline we re-evaluate.
- Brain4FMs / EEG-FM-Bench / AdaBrain-Bench convergence (F11).
- Subject leakage history (Roy et al. 2019 systematic review).

### §3 Methods
- Dataset: 70 recordings / 17 subjects with **both DASS-21 and DSS labels**.
  (No 400 s duration filter — that was a documentation artifact.)
- Pipeline: StressEEGDataset → FM backbone → global pool → classifier.
- Per-model normalization (LaBraM zscore; CBraMod / REVE none) — silently
  destroys runs if wrong.
- Window size is a first-class factor: LaBraM / CBraMod 5 s; REVE 10 s
  (matches pretraining). All frozen-vs-FT comparisons are window-matched.
- Subject-level StratifiedGroupKFold(5) (primary) vs trial-level
  StratifiedKFold(5) (literature reference only).
- cuDNN determinism + multi-seed protocol (≥ 3 seeds for all Stress claims).
- Permutation null via recording-level label shuffling (`--permute-labels`).
- Variance decomposition: pooled label fraction, mixed-effects ICC, matched
  subsampling, cluster bootstrap.
- **All std values reported with sample convention (n−1 divisor).**
  Bootstrap 95% CIs labelled `[low, high]`.

### §4 Results — Pillar A: Methodology traps
- F01 trial-vs-subject CV gap: on Stress, ~30+ pp drop from literature 0.90
  (Wang 2025, same lab, trial-level) to our subject-level 0.45–0.60.
- F08 cuDNN ±20pp single-seed swings on 70 rec / 14 pos.
- F16 classical RF 0.666 was also an OR-aggregation artifact; under honest
  per-rec DASS, all classical methods at or below chance (RF 0.44 etc.).
- F07 subject-dass deprecation rationale referenced; **no subject-dass
  numbers cited** (per revision 2026-04-14).

### §5 Results — Pillar B: Representation diagnosis
- F13 RSA subject r > label r in 12/12 frozen model×dataset combinations.
- F02 subject identity dominates 71% of EEGMAT representation variance.
- F03 classical band-power collapses under per-rec dass; LaBraM frozen LP
  retains 0.605 BA on Stress — FM advantage over classical is real but the
  ceiling is low.
- F10 alpha lateralization (right hemisphere) is the dominant classical
  stress feature — consistent with neuroscience literature.

### §6 Results — Pillar C: Cross-dataset × cross-model FT taxonomy
**Headline section. Four subsections.**

**§6.1 Representation-level pp Δ (F17)** — 3 models × 2 datasets × 3 seeds:
| Model | ADFTD Δ | TDBRAIN Δ |
|---|---|---|
| LaBraM | +1.03 ± 0.74 | −1.56 ± 0.28 |
| CBraMod | +0.83 ± 3.35 (σ > |μ|) | −0.02 ± 0.04 |
| REVE | −1.53 ± 0.28 | +0.44 ± 0.32 |

LaBraM / REVE directions are opposite on same labels — model × dataset
interaction, not label biology.

**§6.2 Behavioral-level (Master Table, 3×4)** — frozen LP + FT BA across all
four datasets. Source: `master_frozen_ft_table.md`. Will appear as **Table 1**
in the paper. Two cells missing: CBraMod / REVE EEGMAT FT (TODO).

**§6.3 N-invariance (F04 matched-N)** — LaBraM-only; taxonomy persists at
matched N=17.

**§6.4 Within-subject design is not a universal rescue (F09 + F14)** — new.
EEGMAT paired rest/task: LaBraM FT 0.731 ± 0.021 (success). Stress
within-subject DSS trajectory (F14): all 3 FMs × 3 classifiers ≤ chance.
Principle: contrast strength matters, not just within-subject framing.

### §7 Neuroscience interpretability (exp14 triad + cross-model consistency)
TNSRE-flavor section. Reframed 2026-04-14 around **cross-model consistency
as a signal-vs-noise test**.
- Spatial: channel-importance topomap (gradient-based).
- Correlational spectral: per-band RSA.
- Causal spectral: band-stop ablation across 4 bands (delta/theta/alpha/beta).
- **Cross-model convergence test**: on EEGMAT, LaBraM and REVE both peak at
  alpha → real neural signature. On Stress, LaBraM peak=beta, REVE peak=alpha
  → divergence → noise floor. CBraMod always peaks at delta (architecture
  artifact, noted separately).

Claim: FMs capture physiological structure when the contrast is clean; the
divergence on Stress is the 4th line of power-floor evidence (§8).

### §8 Stress as statistical power floor (cross-pillar case study)
**Five** independent lines of evidence on 70 rec / 14 positive:
1. F08 cuDNN ±14 pp single-seed swings (non-reproducibility).
2. F06 LaBraM null-indistinguishable (p=0.70).
3. F19 CBraMod/REVE null borderline (p=0.10, 10-perm floor).
4. §7 band-stop cross-model divergence on Stress (alpha/beta/delta split vs
   EEGMAT's alpha convergence).
5. F20 ShallowConvNet from scratch hits 0.557, within FM range — architecture
   confers no advantage; **F14** within-subject DSS trajectory classification
   fails for all 3 FMs × 3 classifiers (failure points to Stress contrast
   weakness, confirmed by EEGMAT success in §6.4).

**Conclusion of §8**: On 70 rec / 14 positive, no method claim
(FM superiority, FT direction, architecture, band causality, within-subject)
is distinguishable from noise. This is a benchmark-design caveat, not a
statement about any specific model.

### §9 Discussion
- Why FT direction depends on model × dataset: CBraMod's criss-cross
  attention may interact differently with spectral structure; REVE's linear
  patch embedding is scale-sensitive; LaBraM's neural tokenizer may discard
  subtle spectral differences under FT.
- Window size as a first-class factor (F18, but kept in supplementary per
  2026-04-14 decision; methods §3 notes it).
- Ruled-out alternatives: F12 GRL/DANN/LEAD, F14 within-subject longitudinal,
  sparse-label-subspace hypothesis, spectral-FiLM.
- Recommendations for the field: (1) subject-level CV as default; (2) report
  statistical power (n_subjects, n_positive, multi-seed std, permutation
  null, bootstrap CI).
- Clinical implication (TNSRE angle): frozen FM representations retain
  diagnostic value (e.g. LaBraM 0.605 on Stress beats classical), but
  deployment requires per-individual calibration given subject dominance.

### §10 Conclusion + limitations
- Untested: CBraMod / REVE on EEGMAT FT (2 cells of Master Table); cross
  3-FM × 4-dataset matrix otherwise complete.
- Mechanism of model × dataset interaction remains a hypothesis requiring
  architectural ablation.
- HNC private dataset (308+400 subjects) as potential high-powered
  validation if power-floor narrative becomes central.

---

## 4. Figure & table plan

**Target**: 1 main table + 8 main figures + 7 supplementary. TNSRE norm is
6–9 main figures; each figure must carry exactly one claim.

### Main table

| # | Title | Source | Status |
|---|---|---|---|
| **Table 1** | Master frozen-LP + FT BA across 3 FMs × 4 datasets | `paper/figures/source_tables/master_frozen_ft_table.{md,json}` | **READY** (2 missing cells: CBraMod / REVE EEGMAT FT — TODO) |

Will appear in §6.2 as the canonical behavioral-level evidence. Sample std
convention; LaTeX version in `paper/sections/table1_master.tex`.

### Main figures

| # | Title | File | Status | Notes |
|---|---|---|---|---|
| 1 | Pipeline + CV protocol schematic | `paper/figures/main/fig1_pipeline.pdf` | **MISSING** | Conceptual vector diagram |
| 2 | Trial vs subject CV gap | `paper/figures/main/fig2_cv_gap.pdf` | **MISSING** | Bar: Wang 2025 0.90 vs our 0.45–0.60 frozen / 0.52–0.58 FT, with asymmetric-protocol caveat in caption |
| 3 | Subject dominance (composite) | `fig3a_fitness_heatmap.pdf` + `fig3b_tsne_cross_dataset.png` | **REWORK** | Merge into single 2-panel figure |
| 4 | Classical vs FM under honest labels | `fig4_classical_vs_fm.pdf` | **READY** (verified 70-rec) | — |
| 5 | **Cross-dataset × model FT taxonomy** (headline) | `fig5_cross_dataset_taxonomy.pdf` | **READY** (rebuilt with sample-std CIs 2026-04-14) | CBraMod×ADFTD `σ > \|μ\|` annotation |
| 6 | Matched-N subsample curves | `fig6_matched_n_curves.pdf` | **READY** | LaBraM-only caveat in caption |
| 7 | Interpretability triad + cross-model consistency | `fig7a_topomap.pdf` / `fig7b_band_rsa.pdf` / `fig7c_band_stop.pdf` | **READY** (7c rebuilt 2026-04-14: EEGMAT converges at alpha, Stress diverges) | — |
| 8 | Stress power floor (composite) | `fig8a_stress_erosion.pdf` / `fig8b_stat_hardening.pdf` / `fig8c_non_fm_baselines.pdf` | **READY** (8c new 2026-04-14) | Caption must point to §8 power-floor framing |

### Supplementary

| S# | Title | Source | Status |
|---|---|---|---|
| S1 | Signal-strength spectrum | `exp01/signal_strength_spectrum` | **READY** |
| S2 | Permutation null histograms | `exp03/ft_null_{labram,cbramod,reve}` | **MISSING** — create overlays |
| S3 | Matched-N BA curve alt view | `exp09/matched_n_ba_curve` | **READY** |
| S4 | Within-subject longitudinal failure (F14) | `exp11/feature_space_analysis`, `within_subject_supplementary` | **READY** — promoted narrative weight via §6.4 |
| S5 | REVE window robustness (5s vs 10s, F18) | `exp12` vs `exp07`, `f18_*_window.json` | **MISSING** — plot from JSON |
| S6 | EEGMAT paired-design supporting (F09) | `exp04_eegmat_feat_multiseed` | **MISSING** |
| S7 | GRL/DANN/LEAD full λ sweep (F12) | F12 table (text-only or small bar) | **MISSING** |

### Work remaining (prioritized by paper-review severity)
1. Fig 1 pipeline schematic (conceptual, vector)
2. Fig 2 trial-vs-subject bar + protocol asymmetry caveat
3. Fig 3 composite merge (a)+(b)
4. S2 / S5 / S6 / S7 supplementary plots
5. CBraMod + REVE EEGMAT FT (2 missing Master Table cells) — requires GPU

---

## 5. Lock / pending / open

| Section | Status | Blocker |
|---|---|---|
| Title, §1–§5, §7 | Lock (content; outline LaTeX in progress) | — |
| §6 (Pillar C + Master Table) | Lock; §6.2 awaiting CBraMod+REVE EEGMAT FT | GPU job |
| §8 (power floor case study) | Lock pending D1 advisor sign-off | Does advisor agree Stress is power-floor, not main result? |
| §9 (discussion) | Draft after §8 locked | — |
| §10 (limitations, HNC note) | Draft; HNC inclusion pending D1 | — |

**D1 (narrative framing)** — tentative: Option B+C hybrid (task-property +
cross-dataset taxonomy), Stress as power floor. **Needs advisor.**

**D2 (REVE window presentation)** — tentative: 10s-matched primary (native
REVE config), 5s-matched supplementary. Window moved out of Results main
text into Methods + supplementary per 2026-04-14 decision. **Needs advisor.**

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

---

## 9. Mindset note

Understanding *why* a task is hard — with quantitative diagnosis,
cross-dataset taxonomy, and statistical-power awareness — is the
contribution. Three independent 2026 benchmarks converge on our empirical
claim (F11); our addition is the mechanistic explanation (subject dominance
via variance decomposition) and the model × dataset FT interaction (F17)
they do not have.

Stress stays in the paper as the cautionary tale, not as evidence of
anything positive or negative about any specific model.
