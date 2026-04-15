# Findings

Single source of truth for confirmed **paper claims**.
Guardrails, methodology notes, and archived findings live in `docs/methodology_notes.md`.

**Read this file at session start to know what is current.**

> **Notation**: All `mean ± std` values use **sample std** (divide by n−1).
> Bootstrap 95% CIs are labelled `[low, high]`.
> Audit script: `scripts/audit_std_convention.py`.

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

## Central thesis

> **EEG FM downstream performance is governed by the ratio of task contrast to
> subject variance.** When a dataset's labels correspond to a neural contrast
> that dominates the within-subject spectrum (e.g. alpha suppression under
> EEGMAT rest/task), FMs recover the signal and fine-tuning helps. When the
> label contrast is weak relative to subject variance (e.g. UCSD DASS-based
> stress state), FMs revert to subject-fingerprint representations and no
> architecture — pretrained FM or from-scratch CNN — exceeds the ceiling.

All six claims below are facets of this thesis:

| Claim | Thesis role |
|---|---|
| **F-A** subject variance dominates frozen FM reps | the denominator (always large) |
| **F-B** honest CV is required to measure the ratio | measurement protocol (removes evaluation inflation) |
| **F-C** FT is a model × dataset interaction | the *mechanism* by which architecture + contrast interact |
| **F-D** contrast strength governs FM rescue | the *outcome*: EEGMAT rescues, Stress does not |
| **F-NEURO** cross-model neural band consensus diagnoses contrast | *independent mechanism test* (neural, not behavioural) |
| **F-E** classical features rely on alpha lateralization | minor/context |

---

## F-A: Subject identity dominates frozen FM representations

**Status**: confirmed
**Absorbs**: F02 (pooled label fraction, cosine), F13 (fitness metrics across 4 datasets)
**Thesis role**: the denominator — frozen FM representations carry large subject variance regardless of dataset.

**Evidence:**

- **Pooled label fraction** (LaBraM frozen, 4 datasets): 2.8–7.2% — task labels explain a small slice of representation variance. Source: `paper/figures/variance_analysis.json`.
- **Cosine similarity**: within-subject ≫ between-subject for all 3 FMs.
- **Mixed-effects ICC**: subject identity dominates 71% of EEGMAT representation variance.
- **RSA triangulation** (`results/studies/exp06_fm_task_fitness/fitness_metrics_full.json`): RSA subject-r > RSA label-r in **12/12** frozen model × dataset combinations. Confirmed by silhouette, Fisher score, kNN, LogME, H-score.

**Key insight**: frozen FM representations encode *who* the recording belongs to more strongly than *what* the label is. This is the structural baseline against which F-C and F-D must work.

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

**Status**: confirmed (3-dataset triangulation)
**Absorbs**: F17 (ADFTD/TDBRAIN multi-model taxonomy), F05 (Stress HP sweep), F09 (EEGMAT projection-not-rewrite), F18 (REVE window caveat). **Supersedes F04** (LaBraM-only 10s historical version).
**Thesis role**: mechanism — FT direction (injection vs erosion) depends on architecture × dataset contrast structure, not on label biology alone.

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

**Status**: confirmed (paired experiment 2026-04-13)
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
**Absorbs**: exp14 triad (channel ablation, band RSA, band-stop) — previously buried in paper_strategy.md §7; promoted to claim 2026-04-15.
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
| LaBraM FT subject-dass 0.656 | Single-seed, cuDNN noise, OR-aggregation | Multi-seed per-rec dass: 0.443 ± 0.083 (canonical) or 0.524 ± 0.010 (best HP) |
| Classical RF subject-dass 0.666 | OR-aggregation artifact (see `methodology_notes.md` G-F07) | Per-rec dass: RF = 0.44, all classical below chance |
| Trial-level LaBraM 0.862 | Subject leakage, OR-aggregation | Not comparable to per-rec subject-level CV |
| "34× subject/label η² ratio" | Naive one-way η² with nesting confound | Pooled label fraction (2.8–7.2%) |
| "ADFTD ×2.76 rewrite" | N-inflation artifact | +5 pp additive (N-invariant) |
| "9.1→1.6 per-fold Stress drop" | Per-fold degenerate (1 subject per positive class) | Do not cite per-fold Stress |
| "Erosion is model-universal" | Refuted by HP sweep | LaBraM-specific on Stress; see F-C.2 |
| Stress variance row (7.23%→7.24%) | Computed under subject-dass | Needs re-run with `--label dass --save-features` |
