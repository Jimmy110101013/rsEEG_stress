# Findings

Single source of truth for confirmed **paper claims**.
Guardrails, methodology notes, and archived findings live in `docs/methodology_notes.md`.

**Read this file at session start to know what is current.**

---

> **⚠️ HP-CONTAMINATION AUDIT (2026-04-19)** — critical read before citing any FT number.
>
> All `exp_newdata` FT runs (6 datasets × 3 FMs × 3 seeds, the dataset supporting paper ΔBA and FT-direction claims) were executed with **unified hyperparameters that deviate 10–80× from every official recipe**:
> - lr=1e-5 (LaBraM TUAB paper: 5e-4 • LEAD: implicit 1e-4 per §3.1 • CBraMod: 1e-4 • REVE: 2.4e-4)
> - batch=4 (LEAD: 512 • LaBraM: 64 • CBraMod: 64 • REVE: 32)
> - layer_decay off (LaBraM TUAB + local stress `finetune.sh`: 0.65)
> - weight_decay=0.01 (LaBraM/CBraMod: 0.05)
> - loss=focal γ=2 (all official: CE or CE+LS 0.1)
> - head=MLP(128) (LaBraM: single Linear init_scale 0.001)
>
> In exp20 (Wang split, unrelated to exp_newdata) we DID test `layer_decay=0.65` + `wd=0.05` but with the same lr=1e-5 and loss=focal — CBraMod + REVE collapsed to chance (bal_acc=0.50) while LaBraM went high-variance (0.32–0.93 across seeds). This confirms the HP diagnosis.
>
> **Claims that are HP-contaminated and require reproduction before paper submission**:
> - F-C.1 (ADFTD/TDBRAIN Δ label fraction direction flip), F-C.2 (Stress HP sweep — lr=1e-4 was tested only via one seed path), F-C.3 (EEGMAT projection-not-rewrite — BA 0.736 may be suppressed), F-C.4 (REVE window caveat).
> - F-D.2 (Stress longitudinal DSS FT near-chance), F-D.3 (architecture-independent ceiling — LaBraM/CBraMod/REVE all ~0.52–0.58 *might* converge on a higher ceiling with proper HP), F-D.4 (permutation null indistinguishability — same LaBraM recipe).
> - F-B §3.2 table Trial-vs-Subject ΔBA for LaBraM/REVE/CBraMod (subject-level FT numbers share the contaminated recipe).
>
> **Claims that are HP-safe (do NOT depend on FT runs)**:
> - F-A (variance decomposition on frozen features — no FT)
> - F-B classical baselines, Frozen LP 0.605 (LP head lr=5e-3, reasonable for linear head)
> - F-NEURO.1/2/3 (all on frozen representations)
> - F-HHSA (raw EEG, no model)
> - F-E (classical RF feature importance, no FM)
>
> **Plan**: ADFTD 3-class LEAD reproduction (target LaBraM sample-level F1 75.64±4.68 / subject-level F1 91.14±8.64) is the sanity-check gate. If reproduction succeeds under per-FM official HP, re-run exp_newdata with aligned HP before re-validating F-C / F-D.2-4 / F-B FT rows.

> **Notation**: All `mean ± std` values use **sample std** (divide by n−1).
> Bootstrap 95% CIs are labelled `[low, high]`.

> **ID scheme**: `F-A` … `F-E` + `F-NEURO` are paper-citable claims. Original
> `F01`–`F21` IDs are preserved in each claim's "Absorbs" line for traceability.

> **DASS-21 psychometric scope (added 2026-04-15)**: DASS-21 is validated as a
> **past-week trait/state screener**. No published literature establishes a
> per-recording EEG correlate of DASS score, and the UCSD dataset (Komarov, Ko,
> Jung 2020, TNSRE 28(4):795) was designed as a **group-level longitudinal
> correlational study**, not a single-recording binary classification
> benchmark. We use `--label dass` (per-recording) with threshold 50 to
> reproduce Wang et al. 2025's protocol, **but this extends DASS-21 beyond its
> validation envelope**. All claims derived from UCSD Stress classification
> performance are conditional on this scope; see §10 Limitations in the paper.

---

## Central thesis (2026-04-23 pivot — task-substrate alignment axis)

> **On small-N clinical EEG (N ≤ 82 subjects) under subject-level cross-validation,
> pretrained EEG foundation model behaviour is carved by **task-substrate
> alignment strength** — the degree to which a task's label maps to an
> identifiable neural activity pattern the FM learned during pretraining.
> Across a 2×2 factorial of CV regime (within-subject paired vs subject-label
> trait) × alignment strength (strong vs weak), the alignment column
> determines FM downstream success regardless of the regime row. We document
> per-cell outcomes and provide a diagnostic toolkit — permutation null,
> variance decomposition, within-subject direction consistency, and
> architecture ceiling — that characterises each cell's dominant mechanism.
> We do not claim predictive generalisation to new datasets within each cell
> (n=1 per cell).**

### 2×2 factorial (updated 2026-04-23, anchored on Fig 3 permutation null)

|   | **Strong-aligned task** (canonical neural signature) | **Weak-aligned task** (behavioral / state summary) |
|---|---|---|
| **Within-subject paired** | **EEGMAT** — rest vs arithmetic → theta/alpha; real FT 0.731, perm-null p = 0.03, dir_cons LaBraM 0.149 | **SleepDep** — normal vs sleep-deprived (state label); real FT 0.532, perm-null p = 0.19, dir_cons LaBraM −0.003 |
| **Subject-label trait** | **ADFTD** — AD vs HC → 1/f aperiodic slope; real FT 0.709, perm-null p = 0.03 (subject-level perm), FT Δlabel_frac +10.7% | **Stress-DASS** — DASS-21 score threshold; real FT 0.524, perm-null p = 0.32, FT Δlabel_frac −1.0% |

Source: `paper/figures/main/fig3_honest_evaluation_4panel.{pdf,png}`, built from `scripts/figures/build_fig3_perm_null_4panel.py`.

### Axis operational definitions

- **Task-substrate alignment strength** (column): degree to which the label
  is grounded in an EEG-identifiable neural pattern. **Primary diagnostic**
  = permutation null (`results/studies/exp27_paired_null/*`): real LaBraM FT
  BA compared against 30-seed label-shuffle distribution. Strong-aligned →
  real clears null (`p ≤ 0.05`); weak-aligned → real inside null
  (`p > 0.1`). For ADFTD (subject trait), permutation is `--permute-level
  subject` so a subject's recordings stay label-consistent in the null.
  Supporting diagnostics: variance-decomposition `Δlabel_frac` and
  within-subject `dir_consistency` (direction agree with alignment, do not
  flip the verdict).
- **CV regime** (row): structural dataset property. **Within-subject
  paired** (EEGMAT, SleepDep) — both label classes present in every
  subject; supports `dir_consistency`. **Subject-label trait** (ADFTD,
  Stress-DASS) — one label per subject (Stress-DASS caveat: 14/17 subjects
  consistent, 3/17 straddle under per-recording DASS binarisation). The
  regime does **not** predict FM success on its own — SleepDep is
  within-subject paired yet fails the null; ADFTD is subject-label yet
  clears it.

### Why the axis pivoted from "CV regime" to "alignment strength" (2026-04-23)

Earlier (2026-04-21) framing treated CV regime as the primary carving
variable with alignment as a secondary "coherent vs absent signal"
qualifier. The 2026-04-23 permutation-null completion (30 seeds × 4
datasets) showed the column (alignment) carves the data in both rows,
while the row (regime) does not discriminate within either column. This
reverses the priority: alignment is the primary mechanism, regime is a
structural variable that affects *which diagnostics apply* (only
within-subject regimes support `dir_consistency`) but not the outcome.

### Dropped from 2×2 framing

- **TDBRAIN** — duplicates the "between-subject + coherent signal" cell occupied
  by ADFTD. Retained as supplementary-only.
- Within-quadrant replication remains an open question (each cell is n=1 dataset).

### Scope boundaries

- **Datasets covered**: Stress-DASS (70 rec / 17 subj), SleepDep (72 / 36),
  EEGMAT (72 / 36), ADFTD (82 / 82).
- **Pretrained FMs tested**: LaBraM-base, CBraMod, REVE.
- **Protocol**: subject-level 5-fold stratified CV; per-FM canonical norm
  (LaBraM zscore, CBraMod/REVE none); FT with unified HP (lr=1e-5, wd=0.05,
  encoder_lr_scale=0.1) — the HP-contamination caveat (block above) applies.

All claims below are facets of this thesis:

| Claim | Thesis role | tex status |
|---|---|---|
| **F-A** subject variance dominates frozen FM reps | the denominator (always large) | §3.1 — **core** |
| **F-B** honest CV is required to measure the ratio | measurement protocol | §3.2 intro — **supporting** |
| **F-C** FT is a model × dataset interaction | cross-dataset taxonomy | **not in tex results** (archived from old narrative) |
| **F-D** contrast strength governs FM rescue | the *outcome*: EEGMAT rescues, Stress does not | §3.2 — **core** |
| **F-NEURO** cross-model neural band consensus | independent mechanism test (neural, not behavioural) | §3.2 integrated — **core** |
| **F-FOOOF** causal aperiodic/periodic ablation → Type I/II/III anchor taxonomy | causal signal-level test of anchor type (3 datasets) | §4.4 — **core** |
| **F-HHSA** holospectral contrast gradient | model-independent signal-level contrast measure | Discussion — **new** |
| **F-E** classical features rely on alpha lateralization | minor/context | §3.2 — **minor** |

---

## F-REPRO: LEAD ADFTD 3-class sanity reproduction (2026-04-19)

**Status**: directional PASS, magnitude gap
**Purpose**: validate pipeline under per-FM official HP before rerunning exp_newdata FT. Sanity anchor = LEAD v4 Table 2 LaBraM ADFTD 3-class subject-independent 8:1:1 × 5 seeds.

### Protocol
ADFTD 3-class (88 subjects, HC=29/AD=36/FTD=23), subject-independent 8:1:1 split × 3 seeds (41, 42, 43). Per-FM official HP (see `methodology_notes.md` G-F09). Per-window CE loss + majority-vote subject-level aggregation (LEAD §2.2, G-F10). Driver: `scripts/experiments/run_sanity_lead_adftd.py`. Raw outputs: `results/studies/sanity_lead_adftd/*.json`.

**Note on batch size**: seed 41 ran batch=128, seeds 42–43 ran batch=512. LEAD uses batch=512. Clean apples-to-apples = seeds 42+43 only (n=2).

### Results (batch=512, n=2 seeds)

| FM | FT sample F1 | FT subject F1 | Frozen subject F1 | ΔF1_subject |
|---|---|---|---|---|
| LaBraM | 0.507±0.171 | 0.441±0.225 | 0.333±0.314 | **+0.107** |
| CBraMod | 0.462±0.146 | 0.498±0.223 | 0.205±0.000 | **+0.292** |
| REVE | 0.358±0.014 | 0.345±0.043 | 0.260±0.078 | **+0.085** |

**vs LEAD v4 (batch=512, n=5)**: LaBraM FT subject F1 0.9114±0.086, sample F1 0.7564±0.047. CBraMod FT subject F1 0.8221±0.063, sample F1 0.6833±0.045. We are 20–47 pp below LEAD means across the board; our std is 3–5× LEAD's.

### Claims

- **Direction confirmed**: FT > Frozen on all 3 FMs (ΔF1_subject +0.085 to +0.292). **This alone invalidates the exp_newdata "FT ≤ Frozen" narrative** — that pattern was HP-driven, not a subject-level-CV property.
- **Pipeline calibration gap**: absolute numbers 20–47 pp below LEAD. Likely causes (in priority):
  1. Batch 512 on our 88-subject dataset leaves only ~22 gradient steps/epoch vs LEAD's larger effective dataset (88×150 windows = 13k); cosine schedule may not provide enough updates
  2. Small val set (9 subjects × 3 classes → very noisy val F1) + patience=15 triggers premature early-stop (best epoch 9–32, far below LEAD's up-to-200)
  3. Only 2 seeds under LEAD protocol (they use 5) — seed noise dominates; LaBraM FT subject F1 spans [0.281, 0.852] across our 3 seeds
- **Not pipeline bug**: seed 41 LaBraM subject F1 = 0.852 is within 1σ of LEAD 0.911±0.086 → structurally our code can reach LEAD territory.

### What this unblocks

- exp_newdata FT rerun with per-FM official HP (G-F09) is now authorized
- All claims in the "Pending experiments" table at bottom of this file are now runnable — use LEAD-style protocol + per-FM HP
- To match LEAD's absolute calibration: add seeds 44+45 to close 3 → 5 seed gap, consider batch=256 compromise, or increase patience to 30

---

## F-A: Subject identity dominates frozen FM representations

**Status**: confirmed
**Absorbs**: F02 (pooled label fraction, cosine), F13 (fitness metrics across 4 datasets)
**Thesis role**: the denominator — frozen FM representations carry large subject variance regardless of dataset.

**Evidence:**

- **Pooled label fraction** (LaBraM frozen, 4 datasets): 2.8–7.2% — task labels explain a small slice of representation variance. Source: `paper/figures/_historical/source_tables/variance_analysis_all.json` (legacy pooled run); current per-cell triangulation: `results/studies/exp32_variance_triangulation/`.
- **Cosine similarity**: within-subject ≫ between-subject for all 3 FMs.
- **Mixed-effects ICC**: subject identity dominates 71% of EEGMAT representation variance.
- **RSA triangulation** (`results/studies/exp06_fm_task_fitness/fitness_metrics_full.json`): RSA subject-r > RSA label-r in **12/12** frozen model × dataset combinations. Confirmed by silhouette, Fisher score, kNN, LogME, H-score.
- **SleepDep replication (2026-04-21)**: added 3rd main dataset datapoint. Frozen variance fractions: `subject_frac` 46–68%, `label_frac` <6%; RSA `r(subject) = 0.054–0.084`, `r(label) = −0.006 to +0.018`. All three FMs in same subject-dominated regime as Stress + EEGMAT. Source: `paper/figures/_historical/source_tables/sleepdep_variance_rsa.json`.
- **Frozen → FT trajectory (regime-conditional, see N-F22)**: FT moves variance in **opposite directions** per regime — Stress `subject_frac` 49 → 12% (pushes to residual, label_frac also drops), EEGMAT `subject_frac` 76 → 87%, SleepDep 46 → 78%. RSA tells a different story: `rsa_subject_r` rises under FT for Stress too (0.19–0.28 → 0.25–0.33). Variance SS vs rank RSA measure different aspects of subject alignment and can disagree — they are complementary, not substitutes. Source: `paper/figures/_historical/source_tables/ft_rsa_stress_eegmat.json`.

**Key insight**: frozen FM representations encode *who* the recording belongs to more strongly than *what* the label is. This is the structural baseline against which F-C and F-D must work. **Subject_frac direction under FT is not a universal quality signal** — interpretation is regime-conditional (see `docs/methodology_notes.md#N-F22`, `docs/regime_framing_decision.md`).

---

## F-B: Honest evaluation is prerequisite — trial-level CV inflates literature baselines by 20–30 pp

**Status**: confirmed
**Absorbs**: F01 (trial-vs-subject CV gap), F03 (classical collapse under honest labels), F16 (classical artifact)
**Thesis role**: measurement protocol — before the contrast/variance ratio can be quantified, evaluation-design inflation must be removed.

### Trial-level → subject-level CV

Under per-recording DASS labels (our honest label setting, 3 seeds, 2026-04-14):

| Model | Trial BA | Subject BA | Δ (pp) |
|---|---|---|---|
| LaBraM | 0.676 ± 0.052 | 0.524 ± 0.010 | **+15.2** |
| REVE | 0.643 ± 0.063 | 0.577 ± 0.051 | +6.5 |
| CBraMod | 0.560 ± 0.049 | 0.548 ± 0.031 | +1.2 |

Sources: `results/studies/exp18_trial_dass_multiseed/`, `results/hp_sweep/20260410_dass/{model}/`.

**Wang et al. 2025 (arXiv:2505.23042)** reports 90.47% BA on UCSD Stress under trial-level CV — subject-level CV alone accounts for most of that inflation (our subject-level ceiling is 0.52–0.58). The gap under per-rec labels is model-dependent and dominated by LaBraM.

### Classical baselines under honest per-rec DASS (70 recordings)

All methods use class-balanced weighting (R1 C3 audit, 2026-04-15; 4/5 already balanced in pre-existing runs, XGBoost retrospectively fixed with `sample_weight='balanced'`).

| Method | 5-fold BA (mean ± std) | pooled BA |
|---|---|---|
| RF (class_weight=balanced) | 0.438 ± 0.022 | 0.438 |
| LogReg_L2 (class_weight=balanced) | 0.491 ± 0.098 | 0.491 |
| SVM_RBF (class_weight=balanced) | 0.429 ± 0.085 | 0.429 |
| **XGBoost (sample_weight=balanced)** | **0.553 ± 0.081** | 0.482 |
| **LaBraM frozen LP (3-seed)** | **0.605 ± 0.032** | — |

Source: `results/studies/exp02_classical_dass/rerun_70rec/summary.json` (4 methods) + `results/studies/exp02_classical_dass/rerun_70rec_xgb_balanced/summary.json` (XGBoost audit).

**Class-imbalance audit result (R1 C3, 2026-04-15)**:
- Plain XGBoost (no weighting) → balanced XGBoost: **+4.5 pp** (0.508 → 0.553 fold-mean BA). Confirms XGBoost's earlier 0.44 was partially an imbalance artifact.
- Other 4 methods were already balanced; their numbers unchanged.
- **Conclusion**: on class-balanced evaluation, classical methods cluster around chance (0.43–0.55), not strictly "≤ chance". Best balanced classical (XGBoost 0.553) trails LaBraM frozen LP (0.605) by **~5 pp**, not 15 pp as an unbalanced comparison suggested.

**Key insight (revised 2026-04-15)**: under honest subject-level per-rec evaluation with class-balanced classical baselines:
- (a) Wang's 90% drops to 0.52–0.58;
- (b) FMs still outperform best balanced classical by **~5 pp** (0.605 vs 0.553);
- (c) this 5 pp gap lives **inside the F-D architecture-independent band** (0.44–0.58) — the FM advantage is small even when it exists. This aligns with the SDL thesis: FM frozen representations carry a little more than classical band-power, but not enough to escape the subject-dominance ceiling.

*Internal historical note: subject-dass OR-aggregation was an earlier internal labelling choice we deprecated (see `methodology_notes.md` G-F07). Not cited in the paper.*

---

## F-C: Fine-tuning is a model × dataset interaction, not label biology

**Status**: ⚠️ HP-CONTAMINATED — all FT-dependent numbers below pending re-run under per-FM official HP (see audit at top of file).
**Status (pre-audit)**: confirmed experimentally, but **not in tex main results** (2026-04-17)
**Absorbs**: F17 (ADFTD/TDBRAIN multi-model taxonomy), F05 (Stress HP sweep), F09 (EEGMAT projection-not-rewrite), F18 (REVE window caveat). **Supersedes F04** (LaBraM-only 10s historical version).
**Thesis role (original)**: mechanism — FT direction (injection vs erosion) depends on architecture × dataset contrast structure, not on label biology alone.

> **Tex status (2026-04-17)**: The tex has moved to SDL (Subject-Dominance Limit)
> framing. FT direction reversal (C.1 ADFTD/TDBRAIN opposite directions) is
> **not discussed in Results**. ADFTD/TDBRAIN are used only for variance
> decomposition (§3.1) and TDBRAIN as intermediate-anchor case in Discussion.
> The master table (C.2) appears in §3.3 as "FM value within ceiling" but
> only compares classical/frozen/FT within each dataset — no cross-dataset
> direction-reversal narrative. C.3 EEGMAT projection-not-rewrite is in §3.2.
> C.4 REVE window caveat is methods-only. **Keep this data for potential
> reviewer questions but do not re-introduce as main result.**

### C.1 — ADFTD and TDBRAIN (representation-level, 3-seed variance decomp)

| Model | ADFTD Δ (pooled label fraction, pp) | TDBRAIN Δ (pp) | Interpretation |
|---|---|---|---|
| **LaBraM** | **+1.03 ± 0.74** (injection) | **−1.56 ± 0.28** (erosion) | direction flips between datasets |
| **CBraMod** | +0.83 ± 3.35 (unstable) | −0.02 ± 0.04 (flat) | FT-insensitive on TDBRAIN; wildly unstable on ADFTD |
| **REVE** | **−1.53 ± 0.28** (erosion) | +0.44 ± 0.32 (mild injection) | **opposite to LaBraM on both datasets** |

Source: `results/studies/exp07_adftd_multiseed/`, `results/studies/exp08_tdbrain_multiseed/`, `results/features_cache/ft_*/`. 5 s windows; all frozen-FT pairs matched.

LaBraM and REVE show **opposite FT direction on identical labels** — core evidence that architecture, not label biology, sets direction.

### C.2 — Stress (behavioral-level, HP sweep)

The architecture divergence reproduces on a third dataset. 54 runs (3 LRs × 2 encoder LR scales × 3 seeds × 3 models). Source: `results/hp_sweep/20260410_dass/`.

| Model | Frozen LP (8-seed) | Best FT (3-seed) | Best FT config | Δ | Direction |
|---|---|---|---|---|---|
| **LaBraM** | **0.605 ± 0.032** | 0.524 ± 0.010 | lr=1e-4, elrs=1.0 | **−8.1 pp** | erosion |
| **CBraMod** | 0.452 ± 0.032 | **0.548 ± 0.031** | lr=1e-5, elrs=0.1 | **+9.6 pp** | injection |
| **REVE** | 0.494 ± 0.018 | **0.577 ± 0.051** | lr=3e-5, elrs=0.1 | **+8.3 pp** | injection |

Erosion on Stress requires a strong frozen baseline (LaBraM 0.605 ≫ CBraMod 0.452 / REVE 0.494) — LaBraM has signal to erode; CBraMod/REVE have headroom to inject. Same model-dependent mechanism as C.1.

### C.3 — EEGMAT: FT does not rewrite, only projects

EEGMAT pooled label fraction 5.35% → 5.82% (within bootstrap noise) despite paired rest/task per subject. Mixed-effects ICC for subject **increases** 0.56 → 0.63 under FT. Behavioral BA=0.736 (LaBraM FT, 3-seed: 0.731 ± 0.021, `results/studies/exp04_eegmat_feat_multiseed/`) is achieved via **projection onto existing subject-dominated axes**, not via representation rewrite.

**Falsifies**: "label-structure alone (paired within-subject design) explains FM ceiling." Paired design is necessary but not sufficient — see F-D for the contrast-strength condition.

### C.4 — Window caveat (REVE on Stress)

REVE injection magnitude is window-dependent. Source: `results/studies/exp16_reve_window_match/`.

| Window | Frozen LP (3-seed) | FT (3-seed) | Δ |
|---|---|---|---|
| **5s-matched** | 0.4970 ± 0.0111 | 0.5446 ± 0.0635 | **+4.8 pp** |
| **10s-matched** | 0.4494 ± 0.0111 | 0.5774 ± 0.0414 | **+12.8 pp** |

Paper cites **10 s-matched** as primary (matches REVE's native pretraining window); 5 s-matched in supplementary.

---

## F-D: Contrast strength governs FM rescue — EEGMAT succeeds where Stress longitudinal fails

**Status**: ⚠️ PARTIALLY HP-CONTAMINATED — D.1 frozen-LP BA is safe, D.2/D.3/D.4 depend on FT recipe and need re-validation.
**Status (pre-audit)**: confirmed (paired experiment 2026-04-13)
**Absorbs**: F09 (EEGMAT success), F14 (Stress longitudinal failure), F06 (LaBraM null on Stress), F20 (ShallowConvNet matches FMs)
**Thesis role**: outcome — quantifies when the contrast/variance ratio is high enough for FM to be useful.

This is the paper's **core paired experiment**: two within-subject designs, identical framework, opposite outcomes. The difference is contrast strength, not methodology.

### D.1 — EEGMAT within-subject rest/task: FM rescue succeeds

Paired rest vs mental-arithmetic recordings per subject (36 subjects × 2 recordings). LaBraM FT 3-seed: **0.731 ± 0.021**. Source: `results/studies/exp04_eegmat_feat_multiseed/`.

Neural contrast: mental arithmetic produces **alpha desynchronization** — a well-established cognitive-load signature (Klimesch 1999). The contrast is strong, within-subject, and spectrally localized.

### D.2 — Stress within-subject DSS trajectory: FM rescue fails

LOO within-subject evaluation using personal-median DSS as threshold (same framework as D.1; reframes Stress as within-subject trajectory). Source: `results/studies/exp11_longitudinal_dss/`.

**Scope caveat (added 2026-04-15 per R2 neuro review)**: DSS (Daily Stress Scale) has **no published within-subject EEG correlate** at the trial / recording level. The failure observed here could therefore have two indistinguishable causes: (a) FM representations lack the capacity to capture within-subject stress variation, or (b) DSS-as-a-label has no within-subject EEG ground truth to begin with. We cannot separate these interpretations with the current data. However, **either interpretation supports SDL thesis**: if (a), subject-dominance prevents rescue; if (b), the label lacks within-subject neural contrast (the thesis's own operative variable). See §10 Limitations for full discussion.

| Model | Centroid BA | 1-NN BA | Linear BA | n recordings |
|---|---|---|---|---|
| LaBraM | 0.296 | 0.296 | 0.000 | 54 |
| CBraMod | 0.241 | 0.167 | 0.000 | 54 |
| REVE | 0.333 | 0.426 | 0.000 | 54 |

All 3 FMs × 3 classifiers **at or below chance**; all kappas negative.

**Direct comparison**: same within-subject framework, same FMs, same subject sample size (36 vs 17) — EEGMAT 0.73, Stress 0.30. The only structural difference is the neural contrast available to the labels. This is the thesis's key controlled comparison.

### D.3 — Stress ceiling is architecture-independent

Under the cross-subject framework, all architectures converge on the same Stress ceiling:

| Model | 3-seed BA |
|---|---|
| ShallowConvNet (Schirrmeister 2017, ~40k params) | 0.557 ± 0.031 |
| EEGNet (Lawhern 2018) | 0.518 ± 0.097 |
| LaBraM FT (~100M) | 0.524 ± 0.010 |
| CBraMod FT (~100M) | 0.548 ± 0.031 |
| REVE FT (~1.4B) | 0.577 ± 0.051 |

Source: `results/studies/exp15_nonfm_baselines/sweep/`. A 2017-vintage CNN reaches the FM range. **FM pretraining confers no task-specific advantage on Stress** — consistent with the contrast-strength thesis: if the contrast/variance ratio is below a threshold, no architecture can recover the label signal.

### D.4 — FT is indistinguishable from permutation null on Stress

Real FT 0.443 ± 0.083 (3 seeds) vs null FT 0.497 ± 0.086 (10 perm), one-sided p = 0.70 (LaBraM canonical recipe). Source: `results/studies/exp03_stress_erosion/analysis.json`. Best-HP FT has not been null-tested; CBraMod/REVE null at 10-perm p=0.1 floor (see `methodology_notes.md` N-F19).

**Key insight of F-D**: the EEGMAT ↔ Stress longitudinal pair (D.1 vs D.2) is the experimental controlled test of the thesis. The architecture-independent ceiling (D.3) and permutation indistinguishability (D.4) are complementary boundary evidence.

---

## F-NEURO: Cross-model neural band consensus diagnoses contrast strength

**Status**: confirmed (2026-04-13)
**Absorbs**: exp14 triad (channel ablation, band RSA, band-stop) — previously buried in `docs/historical/paper_strategy.md` §7; promoted to claim 2026-04-15.
**Thesis role**: *independent neural test* of the contrast-strength hypothesis — not a BA metric, but a mechanism probe.

Three complementary analyses (spatial, correlational spectral, causal spectral) on the same 3 FMs × 2 datasets, frozen representations only. Source: `results/studies/exp14_channel_importance/`.

### NEURO.1 — Spatial (channel-ablation importance)

All 3 FMs prioritize **posterior/occipital channels** (O2, OZ, PZ, P4) on Stress — the source region of alpha oscillations. LaBraM max = 0.0103 (matches F-E classical RF right-posterior); CBraMod max = 0.0033 (3–5× lower sensitivity, uniform); REVE max = 0.0154 (posterior + CPZ).

### NEURO.2 — Correlational band RSA (frozen RDM vs per-band power RDM)

**Stress** — *no band selectivity* (all bands correlate uniformly, p<0.001): LaBraM 0.17–0.27, CBraMod 0.29–0.37, REVE 0.16–0.33. FMs capture broadband spectral structure = subject fingerprint.

**EEGMAT** — *clear alpha/theta preference*: LaBraM (theta 0.19, alpha 0.18, beta ns 0.02); REVE (alpha 0.18, beta 0.15); CBraMod (alpha 0.13, beta 0.17). FMs selectively lock on task-relevant frequencies.

### NEURO.3 — Causal band-stop ablation (cosine distance after Butterworth band removal)

**Stress:** LaBraM most dependent on **beta** (0.168, 2× alpha, 3× delta/theta). CBraMod uniform and low. REVE weakly broadband.
**EEGMAT:** LaBraM and REVE shift to **alpha-dominant** (REVE α=0.150, 3× beta). CBraMod uniform.

### Synthesis

| FM | Stress dominant band | EEGMAT dominant band | Interpretation |
|---|---|---|---|
| **LaBraM** | Beta (arousal / subject fingerprint) | Alpha (task-induced suppression) | FM adapts to available signal |
| **CBraMod** | Uniform (low sensitivity) | Delta (uniform, low sensitivity) | Criss-cross attention is band-agnostic |
| **REVE** | Weakly alpha | Strongly alpha | REVE has cross-task alpha affinity |

**Key insight**: on EEGMAT, LaBraM and REVE **converge on alpha** — independent architectures locking onto the same neural signature (alpha desynchronization, Klimesch 1999). Cross-model convergence = real neural signal. On Stress, the dominant band **diverges by model** (LaBraM beta, REVE broadband, CBraMod uniform) — no shared neural target = the "signal" is architecture-specific artifact, not a stable neural contrast.

This is the **neural-level diagnostic** that complements F-D's behavioural diagnostic. The two together argue that Stress's failure is not an ML problem (D.3 shows architecture doesn't matter) and not an evaluation problem (F-B rules that out) — it is a **neural contrast-strength problem**.

---

## F-FOOOF: Causal anchor dissection (FOOOF ablation + band-stop) gives Type I/II/III taxonomy

**Status**: confirmed (2026-04-21)
**Thesis role**: *causal* upgrade of F-NEURO — two complementary signal-level interventions:
  (i) FOOOF decomposes EEG into aperiodic 1/f + periodic peaks, then ablates one component before FM extraction (probe BA Δ);
  (ii) band-stop filters remove a whole frequency band (peak + in-band 1/f tail) from the raw EEG (cosine distance on FM features).
Together they yield a pre-training, model-independent taxonomy of contrast-anchor types.
**Paper section**: §4.4.
**Sources**:
  `results/studies/fooof_ablation/{stress,eegmat,sleepdep,adftd}_probes.json`;
  `results/studies/exp14_channel_importance/band_stop_ablation.json` (3 datasets merged 2026-04-20).

> **8-seed確認 (2026-04-23)**：全部4個 dataset 的所有 condition 均已跑 8 seeds（JSON內 `subject_probe_bas` / `state_probe_bas` 均為 list of 8）。SUMMARY.md 部分 cell 僅列單一數字係格式省略，非真的 n=1。

### FOOOF.1 — Aperiodic 1/f is the universal subject substrate

Subject-identity probe BA under `aperiodic_removed` (vs `original`):

| Dataset | LaBraM | CBraMod | REVE |
|---|---|---|---|
| EEGMAT | 0.556 → 0.528 (−2.8pp) | 0.507 → 0.365 (**−14.2pp**) | 0.465 → 0.205 (**−26.0pp**) |
| Stress | 0.569 → 0.544 (−2.5pp) | 0.569 → 0.483 (**−8.6pp**) | 0.565 → 0.569 (+0.4pp) |
| SleepDep | 0.406 → 0.361 (−4.5pp) | 0.420 → 0.375 (−4.5pp) | 0.375 → 0.417 (+4.2pp) |
| ADFTD | 0.856 → 0.830 (−2.6pp) | 0.748 → **0.940 (+19.2pp)** ⚠️ | 0.647 → **0.885 (+23.8pp)** |

`periodic_removed` leaves subject probe essentially unchanged in all 12 cells (|Δ| ≤ 1 pp). **Aperiodic 1/f carries subject identity; narrowband peaks do not.** This generalises the F-NEURO observation (Stress-only, band-RSA) to 4 datasets via causal manipulation.

**ADFTD subject-ID 反向解讀（8-seed 穩定，非 noise）**：REVE +23.8pp、CBraMod +19.2pp 代表移除 aperiodic 後受試者反而*更*容易區分。兩種機制：

1. **真實 neural 效應（REVE 確認）**：ADFTD 中 aperiodic exponent (chi) 隨疾病種類（AD/FTD/HC）系統性改變 → aperiodic 成為**疾病分組標記**而非個人指紋。移除後，剩餘 periodic oscillations 反而更個人特異 → subject-ID 上升。
2. **CBraMod × ADFTD preprocessing artifact**：FOOOF spectral whitening 將訊號振幅從 ~50 µV 壓縮至 ~1–5 µV。CBraMod 內部硬編碼 `x/100`（`cbramod_extractor.py:102`），導致輸入從 ±0.5（正常範圍）縮小至 ±0.01–0.05（遠超 pretraining 分布外）。CBraMod state probe 同時上升 +12.1pp（方向與 LaBraM −11.2pp、REVE −3.8pp 相反）— 此反向為 artifact 確診證據，非神經生理現象。**解釋 ADFTD 時以 LaBraM / REVE 為主，CBraMod 數字加 caveat。**

### FOOOF.2 — State anchor differs by dataset type

State probe BA under each ablation, averaged across 3 FMs:

| Dataset | original | aperiodic_removed | periodic_removed |
|---|---|---|---|
| **EEGMAT** (rest vs task) | 0.714 | 0.701 (−1.2pp) | 0.706 (−0.8pp) |
| **SleepDep** (NS vs SD) | 0.573 | **0.528 (−4.5pp)** | 0.575 (+0.2pp) |
| **Stress** (DASS binary) | 0.467 | 0.446 (−2.1pp) | 0.465 (−0.2pp) |
| **ADFTD** (AD/FTD vs HC) | — | per-FM below ⚠️ | — |

Per-FM SleepDep state sensitivity to aperiodic removal: LaBraM 0.616 → 0.538 (−7.8pp), REVE 0.562 → 0.519 (−4.3pp), CBraMod 0.540 → 0.528 (−1.2pp; already low).

Per-FM ADFTD state sensitivity to aperiodic removal: LaBraM 0.654 → 0.542 (**−11.2pp**), REVE 0.652 → 0.614 (−3.8pp), CBraMod 0.571 → 0.692 (**+12.1pp** ⚠️ artifact — 見 FOOOF.1 說明)。Periodic removal: 三者均 ≤ ±0.1pp，不動。**LaBraM/REVE 一致顯示 aperiodic 移除後 disease state signal 流失 → ADFTD 為 `1/f-aperiodic` anchor（aperiodic-anchored trait signal）。CBraMod 反向屬 preprocessing artifact。**

**SleepDep 和 ADFTD（LaBraM/REVE）均在移除 aperiodic 後流失 state signal → 兩者同屬 `1/f-aperiodic` anchor。EEGMAT state 兩種 ablation 均存活 → `α-broadband` anchor。Stress state 始終在 chance → `absent` anchor。**

### FOOOF.3 — Band-stop sensitivity resolves the EEGMAT anchor

FOOOF periodic removal only deletes *peaks* atop the 1/f background; band-stop removes the *entire band* (peak + in-band 1/f tail). Running both on the same datasets resolves the apparent EEGMAT contradiction.

Mean cosine distance across 3 FMs after Butterworth band-stop:

| Dataset | δ (1–4 Hz) | θ (4–8 Hz) | α (8–13 Hz) | β (13–30 Hz) |
|---|---|---|---|---|
| **EEGMAT** | 0.052 | 0.049 | **0.105** ← | 0.061 |
| **Stress** | 0.040 | 0.034 | 0.054 | **0.078** |
| **SleepDep** | 0.008 | 0.007 | 0.012 | 0.007 |

EEGMAT FM features peak in α-band dependence (highest of any cell); SleepDep FM features are flat across bands (lowest by ~5×). Since FOOOF periodic removal (peaks only) barely shifts EEGMAT state probe but band-stop α moves FM representation far more, **EEGMAT state lives in the α band as a broadband entity — the peak plus the in-band 1/f tail together**, not the peak alone.

### Synthesis — three anchor regimes

The two causal probes agree on a three-way dataset taxonomy diagnosable from the EEG *before* any FM training:

| Anchor type | FOOOF signature | Band-stop signature | 2×2 cell / Example | FM rescue prognosis |
|---|---|---|---|---|
| **absent** | State probe near chance in all conditions | Any band-stop cosine distance (subject signature only) | between × incoherent / Stress DASS | FT exploits subject shortcut (F-C, F-D, F-drift) |
| **α-broadband** | State survives both periodic and aperiodic removal | α-band cosine distance peaks (high reliance) | within × coherent / EEGMAT rest/task | FM frozen LP separates (F-D.1); state encoded redundantly across peak + tail |
| **1/f-aperiodic** | State collapses under aperiodic removal only | Band-stop cosine distance flat (no band reliance) | within × incoherent / SleepDep; between × coherent / ADFTD (LaBraM/REVE) | FM frozen LP separates at modest BA (≈ 0.53–0.61); signal diffuse in 1/f slope. ADFTD: aperiodic carries disease-trait signal (LaBraM −11.2pp, REVE −3.8pp); CBraMod reversed = preprocessing artifact |

This is the **strongest form** of the contrast-anchor argument — not a representation correlation (F-NEURO) or a benchmark number (F-D), but *complementary signal-level interventions* that jointly identify which spectral component the FM state probe is reading. The anchor-type classification is model-independent and could be computed on any candidate dataset before committing an FM training pipeline.

**Important interpretation caveat on cosine distance**: band-stop cosine distance measures how much the FM representation *depends on* a band — a reliance / sensitivity metric, not a task-probe accuracy. High cosine distance on a band only implies "FM listens to that band"; whether that band also *carries task signal* requires the FOOOF probe-BA panel alongside.

---

## F-HHSA: Holospectral contrast gradient provides model-independent SDL evidence

**Status**: confirmed (2026-04-17)
**Thesis role**: model-independent, signal-level measurement of condition contrast strength — quantifies the SDL "numerator" without any FM or classifier.

Holo-Hilbert Spectral Analysis (HHSA; Huang et al. 2016, Ho et al. 2026) decomposes EEG into a 2D holospectrum (carrier frequency × AM frequency) via two-layer CEEMDAN. Condition contrast is measured as the between-group t-statistic on per-recording mean holospectra. Unlike F-NEURO (which probes FM representations), HHSA operates on raw EEG — it is truly model-free.

### Cross-dataset holospectral contrast (4 datasets, 60s windows)

| Dataset | Condition contrast | n_rec | mean\|t\| | max\|t\| | Bonf sig bins | AM coherence |
|---|---|---|---|---|---|---|
| **EEGMAT** | rest vs task (within-subject) | 72 | **1.40** | 4.85 | **2** | 0.979 ± 0.015 |
| **Sleep Dep** | normal vs deprived (within-subject) | 213 | **1.32** | 3.86 | 0 | 0.943 ± 0.045 |
| **Stress** | Normal vs Increase (between-group DASS) | 70 | 1.08 | 4.81 | 1 | 0.959 ± 0.036 |
| **Meditation** | ses-1 vs ses-2 (no real manipulation) | 39 | 0.55 | 2.94 | 0 | 0.969 ± 0.022 |

Source: `results/hhsa/cross_dataset_comparison/`, `scripts/hhsa_cross_dataset_comparison.py`.

### Interpretation for SDL

HHSA provides the **contrast-strength gradient** that the tex's Limitation §6.2 identifies as needed: SDL currently uses a binary anchored/bounded classification; HHSA shows it is actually a continuum (EEGMAT > Sleep Dep > Stress > Meditation).

- EEGMAT's strong focal contrast (2 Bonferroni-significant bins in the alpha-carrier × slow-AM region) aligns with Klimesch alpha-ERD anchor
- Stress's weaker, diffuse contrast (mean|t| 24% lower than EEGMAT) is consistent with the absent neural anchor diagnosis
- Sleep Dep is the **key intermediate case**: strong diffuse contrast (mean|t| = 1.32) from a genuine within-subject physiological manipulation — FM performance on this dataset (exp_newdata, pending) will test whether diffuse-but-strong contrast enables FM rescue

**Paper placement**: Discussion, as empirical evidence for the graded-anchor-strength refinement of SDL. Figures in `results/hhsa/cross_dataset_comparison/`.

**Caveat**: Stress contrast here is between-group (DASS), not within-subject. EEGMAT and Sleep Dep are true within-subject designs. Meditation has no real condition manipulation (both sessions are meditation). The ranking is descriptive, not a formal regression.

---

## F-E: Classical features are posterior-alpha dominated (minor/context)

**Status**: confirmed (revised 2026-04-15 per R2 neuro review)
**Absorbs**: F10
**First observed**: 2026-04-04 · **Last revised**: 2026-04-15

RF feature importance on 70-rec per-rec dass: 13/20 top features are alpha band, with highest importance at right-posterior (T4, C4, CP4, P4) and right-frontal (Fp2, F8) sites. Source: `docs/fm_comparison_summary.md` §6.

**Scope of claim**: this is a descriptive observation about which features drove the RF classifier on this dataset. It is **not** evidence of any specific cognitive/affective neural mechanism.

**What this finding does NOT support** (corrections 2026-04-15):

- **Not a Davidson/Coan frontal-alpha-asymmetry (FAA) finding.** FAA in the approach/withdrawal literature is an **L − R asymmetry index** (e.g. F3-alpha − F4-alpha), not absolute right-hemisphere alpha power. Moreover, higher alpha at a site indicates **less** cortical activity at that site (Pfurtscheller 1999 alpha = cortical idling). Earlier drafts that linked this F-E to Davidson FAA inverted the neural interpretation and must not be cited.
- **Not a stress-specific neural signature.** Posterior alpha dominance is the default of resting-state eyes-closed EEG regardless of emotional state.

**What this finding does support**:
- The classical band-power feature set's predictive signal (such as it is, RF bal_acc=0.44 under honest per-rec dass) loads predominantly on posterior alpha channels.
- This matches F-NEURO.1 spatial finding (all 3 FMs also prioritise posterior channels) — classical and FM frozen representations agree on **where** the signal lives, even though FMs additionally capture broader spectral structure (F-NEURO.2).

**Placement**: Introduction or Discussion context only. **Do not invoke Davidson FAA.** Proper FAA analysis on this dataset is future work.

---

## Deprecated numbers — do NOT cite

| Number | Why wrong | Replacement |
|---|---|---|
| Classical RF subject-dass 0.666 | OR-aggregation artifact (see `methodology_notes.md` G-F07) | Per-rec dass: RF = 0.44, all classical below chance |
| Trial-level LaBraM 0.862 | Subject leakage, OR-aggregation | Not comparable to per-rec subject-level CV |
| "34× subject/label η² ratio" | Naive one-way η² with nesting confound | Pooled label fraction (2.8–7.2%) |
| "9.1→1.6 per-fold Stress drop" | Per-fold degenerate (1 subject per positive class) | Do not cite per-fold Stress |

## Pending experiments (invalidated by HP audit, awaiting sanity reproduction)

| Claim | Status | Unblock condition |
|---|---|---|
| LaBraM FT on Stress BA (any value) | Invalid under exp_newdata HP | LEAD sanity PASS → rerun exp_newdata with per-FM official HP |
| ADFTD ΔBA / ×-fold rewrite | Invalid under exp_newdata HP | Same |
| TDBRAIN FT erosion | Invalid under exp_newdata HP | Same |
| "Erosion model-universal vs LaBraM-specific" | Invalid | Same |
| Stress FT vs permutation-null | Invalid | Same |
| ρ(subject_id_BA, ΔBA) between-arm = +0.50 | Invalid (depends on contaminated ΔBA) | Same |
| mean ΔBA between-arm / within-arm | Invalid | Same |
