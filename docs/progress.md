# Research Progress Report

**Project**: EEG Foundation Model Evaluation for Resting-State Stress Classification
**Period**: April 2–10, 2026
**Author**: Jimmy Lin

---

> ### ⚠️ 2026-04-10 reclassification notice (read before citing any number below)
>
> Sections §1–§4.5 were written between 2026-04-02 and 2026-04-08 under the
> now-deprecated `--label subject-dass` OR-aggregation protocol. On
> 2026-04-09/10 we discovered that subject-dass is a **trait memorization
> artifact** — it turns stress classification into subject-identity
> broadcast. Under the honest per-recording DASS protocol (matches Lin et
> al. 2025), the Stress dataset's "no-op" classification is reclassified to
> **behavioral erosion** and the canonical "0.656 LaBraM FT" is a
> non-reproducible single-seed lucky draw under cuDNN non-determinism.
>
> See **§4.6** for the 2026-04-10 findings:
> - Frozen LP: **0.605 ± 0.030** (8 seed)
> - Real FT:   **0.443 ± 0.068** (3 seed)
> - Null FT (shuffled labels): 0.497 ± 0.081 (10 perm)
> - Delta Frozen − FT = **+16.2 pp** erosion
>
> Full evidence: `results/studies/2026-04-10_stress_erosion/analysis.json`.
> The representation-level Stress row (7.23% → 7.24%) in §4.5 and in
> `paper/figures/variance_analysis.json` is **stale** — it was computed
> under subject-dass. Needs regeneration before citation.

---

## 1. Executive Summary

We conducted a systematic evaluation of EEG foundation models (FMs) for
cross-subject resting-state EEG classification across four datasets:
UCSD Stress (17 subjects), EEGMAT (36, within-subject), ADFTD (65),
TDBRAIN (359). Our reviewer-ready metric is the pooled label fraction
$SS_{\text{label}} / SS_{\text{total}}$ in the LaBraM 200-d
representation. Fine-tuning produces **three qualitatively distinct
modes** whose effect directions are **invariant under N-controlled
subsampling** (100 draws per rung, 4-5 rungs per dataset, verified
against a subject-level label-permutation null):

1. **Stress (DASS)** — *no change*. Pooled label fraction 7.23% → 7.24%.
   Classifier BA=0.66 is achieved by reading a pre-existing projection
   through an unchanged backbone.
2. **EEGMAT (task)** — *also no change* (5.35% → 5.82%, within bootstrap
   noise), falsifying the hypothesis that within-subject labels rescue
   FT rewriting.
3. **ADFTD (AD)** — *clean label-signal injection* of +5 pp stable across
   N ∈ {17, 25, 35, 50, 65}. The canonical 2.79% → 7.70% (×2.76) ratio is
   an artifact of the pooled-label-fraction denominator inflating at
   small N; the **N-invariant additive effect is +5 pp** and exceeds the
   permutation null by 5–8×.
4. **TDBRAIN (MDD)** — *active label-signal erosion* of −1.5 pp stable
   across N ∈ {35, 65, 150, 300, 359}. FT training dynamics strip label
   variance that the frozen representation carried. 100% of draws at
   N=300 have FT < frozen, and the observed delta sits on the *opposite*
   side of zero from the permutation null — FT is not merely failing to
   improve, it is actively degrading.

**Headline correction (2026-04-08 evening, after N-controlled experiment):**
the previous "dataset-size-dependent taxonomy" framing conflated N with
label biology. Under matched N=17 (Stress-comparable), ADFTD still gains
+5 pp while Stress gains 0 pp, so the three-mode contrast is a property
of the label content, **not** the training set size. See §4.5 for the
N-control results and `paper/figures/matched_subsample_curves.pdf` for
the figure.

---

## 2. Datasets

| Dataset | Subjects | Recordings | Protocol | Labels | Signal Strength |
|---------|----------|------------|----------|--------|----------------|
| **UCSD Stress** (ours) | 17 | 70 | 5-min eyes-open resting, longitudinal | DASS (binary: normal/increase) | Weak (d=0.2–0.4) |
| **ADFTD** (OpenNeuro ds004504) | 65 (36 AD, 29 HC) | 195 (3 splits/subject) | ~14-min eyes-closed resting | Clinical diagnosis (AD vs healthy) | Strong (spectral slowing) |

- ADFTD: Each ~14-min recording split into 3 contiguous ~4.5-min pseudo-recordings to enable within-subject variance analysis
- 19 common 10-20 channels shared across both datasets for frozen feature analysis
- Each dataset uses its full channel set for fine-tuning (Stress: 30ch, ADFTD: 19ch)

---

## 3. Foundation Models Evaluated

| Model | Architecture | Embed Dim | Status |
|-------|-------------|-----------|--------|
| **LaBraM** (ICLR 2024) | ViT + channel position embeddings | 200 | **Primary** — best performer |
| REVE (2024) | ViT + 4D Fourier PE + MAE | 512 | Ruled out — poor cross-subject |
| CBraMod (ICLR 2025) | Criss-Cross Transformer + ACPE | 200 | Ruled out — chance level (48.8% BA) |

---

## 4. Experimental Results

### 4.1 Subject-Level CV Performance (StratifiedGroupKFold, 5-fold)

| Method | Stress BA | ADFTD BA |
|--------|-----------|----------|
| **LaBraM Fine-tuned** | **0.656** | **0.752** |
| RF on band power (full ch) | 0.666 | 0.753 |
| RF on band power (19ch) | 0.536 | 0.753 |
| LaBraM + GRL (adversarial) | 0.537 | — |
| Classical LogReg | 0.624 | — |
| Classical SVM | 0.580 | — |

**Key observation**: Fine-tuned LaBraM matches classical RF on both datasets. FM pretraining provides no advantage over hand-crafted band-power features.

### 4.2 Trial-Level vs Subject-Level CV Gap (UCSD Stress only)

| Model | Trial-Level BA | Subject-Level BA | Gap |
|-------|---------------|-----------------|-----|
| LaBraM | 0.862 | 0.656 | **-20.6 pts** |
| REVE | 0.770 | 0.553 | -21.7 pts |
| CBraMod | 0.712 | 0.488 | -22.4 pts |

Prior work (Wang et al., arXiv:2505.23042) reports 90% BA on the same dataset using trial-level splits with subject leakage.

### 4.3 Cross-dataset variance decomposition (pooled label fraction)

Reviewer-ready metric: $SS_{\text{label}} / SS_{\text{total}}$ summed over
the 200 LaBraM dims, computed on the pooled fine-tuned feature matrix
(one OOF embedding per recording). Full methodology and rationale in
`docs/eta_squared_pipeline_explanation.md`.

| Dataset | n rec / n subj | Label type | BA | Frozen label frac | FT pooled label frac | Change | N-controlled Δ (§4.5) |
|---|---|---|---|---|---|---|---|
| Stress  | 70 / 17  | between | 0.656 | **7.23%** | **7.24%** | → unchanged | n/a (only 17 subjects) |
| **EEGMAT** | **72 / 36**  | **within** | **0.736** | **5.35%** | **5.82%** | **→ unchanged** | n/a (crossed design) |
| ADFTD   | 195 / 65 | between | 0.752 | 2.79% | **7.70%** | ×2.76 ratio †  | **+5.22 to +5.68 pp across N ∈ [17, 65]** |
| TDBRAIN | 734 / 359| between | 0.681 | 2.97% | 1.47% | ×0.49 ratio †  | **−1.06 to −1.58 pp across N ∈ [35, 359]** |

† **The ratio framing for ADFTD and TDBRAIN is N-inflated** by the
pooled-label-fraction denominator. The honest, N-controlled effect size
is the **additive delta in pp**, which is stable across N — see §4.5
and `paper/figures/matched_subsample_curves.pdf`. At Stress-comparable
N=17, ADFTD's true multiplier is 1.87× (frozen 6.50%, FT 12.18%), not
2.76×.

**Canonical sources**: `paper/figures/variance_analysis.json` (Stress,
ADFTD, TDBRAIN) and `paper/figures/variance_analysis_eegmat.json`
(EEGMAT, separate file because the within-subject design needs the
crossed-decomposition path documented in
`docs/eta_squared_pipeline_explanation.md` §3 and
`scripts/analyze_eegmat.py`).

Supporting metrics (cluster-bootstrapped CIs, REML mixed-effects ICC,
subject-level PERMANOVA) all live in the same JSON and move in the same
directions.

### 4.4 Interpretation

Fine-tuning's effect on the LaBraM representation is dataset-size
dependent, **not** label-structure dependent and **not** a uniform
"reduces subject dominance" story:

- **Stress (n=70, between-subject)**: no measurable change in
  representation structure. Classifier BA=0.66 is achieved by reading a
  pre-existing projection through an unchanged representation.
- **EEGMAT (n=72, within-subject)**: same projection-only behavior
  even though every subject contributes both rest and arithmetic
  recordings. Subject identity dominates 71% of the representation
  variance both before and after FT (mixed-effects ICC 0.56 → 0.63).
  Classifier BA=0.736 is achieved the same way as on Stress — the
  head reads an existing projection. **This rules out the "labels are
  confounded with subject" alternative explanation for the small-data
  failure.**
- **ADFTD (n=195, 65 subjects)**: clean label-signal injection; the
  only dataset where fine-tuning genuinely rewrites LaBraM's variance
  structure. PERMANOVA drops from p=0.05 to p=0.001.
- **TDBRAIN (n=734, 359 subjects)**: the five OOF fold-models drift
  from each other and dilute the global label signal when pooled;
  individual folds show mild label-signal injection that doesn't
  survive pooling.

The original "fine-tuning reduces subject dominance 5-6×" narrative was
based on per-dim averaged ω² and per-fold Stress ratios, both of which
turned out to be measurement artifacts. See §10 of
`docs/eta_squared_pipeline_explanation.md` for the correction history.
The follow-up "within-subject labels rescue FT" hypothesis (added 2026-04-08
morning when EEGMAT was queued as a positive control) was falsified by
the EEGMAT result. The subsequent "size-only taxonomy" (2026-04-08
afternoon) was then itself refined by the N-controlled matched-subsample
experiment (2026-04-08 evening, §4.5), which showed the three-mode
contrast is N-invariant and therefore driven by label biology, not size.

### 4.5 N-controlled matched subsampling (primary robustness check)

To test whether the three-mode taxonomy is an artifact of varying
training-set size across datasets, we ran a paired frozen-vs-FT
subsampling experiment: for each draw, pick the same set of M subjects
(label-stratified, preserving the source dataset's own class ratio),
apply the resulting subject mask to **both** frozen and FT-pooled feature
matrices, and compute the pooled label fraction on each. 100 draws per
rung, 4-5 rungs per dataset. Script:
`scripts/run_variance_analysis.py --matched all --matched-n-draws 100
--permute-labels-check`; output:
`paper/figures/variance_analysis_matched.json`; figure:
`paper/figures/matched_subsample_curves.pdf`.

**ADFTD (AD vs HC)** — FT rewrite stable across N:

| N | Frozen (%) | FT (%) | Δ (pp) | frac FT > frozen |
|---:|---:|---:|---:|---:|
| 17 | 6.50 ± 4.44 | 12.18 ± 4.22 | **+5.68 ± 4.89** | 86% |
| 25 | 5.04 ± 3.24 | 10.53 ± 3.10 | **+5.49 ± 3.52** | 95% |
| 35 | 3.82 ± 2.11 | 9.13 ± 2.30 | **+5.31 ± 2.36** | 98% |
| 50 | 3.14 ± 1.08 | 8.36 ± 1.45 | **+5.22 ± 1.41** | 100% |
| 65 (full) | 2.79 | 7.70 | **+4.91** | — |

**TDBRAIN (MDD vs HC)** — FT erosion stable across N:

| N | Frozen (%) | FT (%) | Δ (pp) | frac FT > frozen |
|---:|---:|---:|---:|---:|
| 35 | 4.76 ± 3.17 | 3.70 ± 1.66 | **−1.06 ± 2.64** | 39% |
| 65 | 3.76 ± 2.34 | 2.51 ± 1.17 | **−1.25 ± 1.72** | 24% |
| 150 | 3.39 ± 1.36 | 1.81 ± 0.62 | **−1.58 ± 0.96** | 2% |
| 300 | 3.03 ± 0.54 | 1.51 ± 0.25 | **−1.53 ± 0.40** | 0% |
| 359 (full) | 2.97 | 1.47 | **−1.50** | — |

**Key readings:**

1. **Both frozen and FT inflate as N shrinks** (ADFTD frozen:
   2.79% → 6.50%; TDBRAIN frozen: 2.97% → 4.76%). This is the
   N-sensitivity of the pooled label fraction denominator — at small N,
   there are fewer between-subject residual directions to absorb
   variance, so SS_label is a larger share of SS_total.

2. **The absolute delta (FT − frozen) is N-invariant.** ADFTD Δ stays at
   +4.91 to +5.68 pp across N ∈ [17, 65]. TDBRAIN Δ stays at −1.06 to
   −1.58 pp across N ∈ [35, 359]. This is the single most important
   robustness check: if the effects were size artifacts the deltas would
   drift with N.

3. **The canonical 2.76× multiplier is inflated** by the N-sensitivity
   of the denominator. At matched N=17 (Stress-comparable), the true
   multiplier is 12.18 / 6.50 = **1.87×**, not 2.76×. The additive
   +5 pp framing is the honest effect size; the ratio framing should be
   used only at full N for comparison with the canonical literature.

4. **At literally matched N=17 with nearly identical frozen baselines
   (~7%)**, ADFTD gains +5 pp from fine-tuning and Stress gains 0 pp.
   This is the strongest available evidence that Stress's FT failure is
   not a sample-size problem: given exactly as many subjects as Stress
   has, ADFTD still works. The difference is in the label biology.

5. **Permutation null (subject-level label shuffle) confirms the
   effects are not metric artifacts.** Under the null:
   - ADFTD Δ drops from +5.2 pp (observed) to ≈+0.7 pp at N=50 and
     +1.5 pp at N=17, i.e. observed is 4–8× the null depending on N.
     The null has a small positive bias from finite-sample
     between-subject-label correlation (shrinks with N as expected).
   - TDBRAIN Δ is **on the opposite side of zero from the null**. At
     N=300 the observed Δ is −1.53 pp while the null Δ is +0.12 pp.
     FT is not just failing to improve; it is actively moving label
     signal in the wrong direction more strongly than label
     permutation would move it.

6. **The three modes are N-invariant and pre-permutation-resistant.**
   - ADFTD (AD): +5 pp across all N, clearly above null at every rung.
   - Stress (DASS): Δ ≈ 0 at the only N we have (17).
   - TDBRAIN (MDD): −1.5 pp across all N, on opposite side of null at
     every rung.

See `paper/figures/matched_subsample_curves.pdf` for the two-panel
figure and `paper/figures/variance_analysis_matched.json` for the full
per-draw JSON (500 observed draws + 500 permuted null draws, with chosen
subject IDs saved per draw for reproducibility).

### 4.6 Stress reclassification (2026-04-09/10): subject-dass artifact + behavioral erosion

The §4.3–§4.5 Stress results (pooled label fraction 7.23% → 7.24%, "no-op"
mode) were computed under `--label subject-dass`, which OR-aggregates a
subject's recordings: if any of a subject's recordings is labelled
"increase", all of that subject's recordings are treated as "increase".
Only 3 of 17 subjects (pids 3, 11, 17 — 9 recordings) actually had
within-subject DASS class transitions; subject-dass artificially boosts
the label signal by turning a per-recording stress-state task into a
subject-identity broadcast task. This was not recognized until we read
Lin et al. 2025 carefully on 2026-04-09 and noticed their protocol uses
per-recording DASS directly.

**Honest protocol (`--label dass`)**. We re-ran LaBraM FT on the
per-recording DASS labels (same as Lin 2025) with the canonical recipe
(lr=1e-5, encoder_lr_scale=0.1, llrd=1.0, epochs=50, patience=15,
warmup_freeze_epochs=1, loss=focal, aug_overlap=0.75) across 3 seeds,
plus a label-permutation null with 10 shuffled-label runs, plus an 8-seed
Frozen LP baseline for context. Full evidence package:
`results/studies/2026-04-10_stress_erosion/analysis.json`.

| Condition | Subject BA | std | n | Interpretation |
|---|---|---|---|---|
| **Frozen LP** (LogReg on 200-d LaBraM features) | **0.605** | 0.030 | 8 seed | primary — pretrained encoder has signal |
| **Real FT** (canonical recipe on real labels) | **0.443** | 0.068 | 3 seed | behavioral erosion vs frozen |
| **Null FT** (canonical recipe on shuffled labels) | **0.497** | 0.081 | 10 perm | real FT ≈ null |

**Findings**:
1. **Frozen LP > Real FT by 16.2 pp**, and Frozen LP's seed variance
   (±0.030) is less than half of Real FT's (±0.068). The pretrained
   representation has a stable, linearly-separable stress signal that FT
   *destroys*.
2. **Real FT is indistinguishable from training on shuffled labels**.
   One-sided p(null ≥ real) = 0.70, and two permutation seeds (s4=0.607,
   s8=0.643) produced BAs *higher* than any real-label FT seed.
3. **cuDNN non-determinism dominates the canonical 0.656 number**. At
   HEAD, running the exact canonical recipe (subject-dass, seed=42) on
   the same code that produced 0.6559 on 2026-04-05 now produces 0.4505
   — a 20.5 pp single-seed swing with zero code drift in the Stress
   training path. Root cause: `train_ft.py` seeds numpy+torch but does
   not set `torch.backends.cudnn.deterministic=True`, and the
   70-recording / 14-positive class-imbalanced regime amplifies
   microscopic init differences into 10–20 pp BA swings.

**Implications**:
- The "Stress 0.656 LaBraM ceiling" is **not a reproducible ceiling** —
  it's a lucky-tail draw from a high-variance seed distribution whose
  true mean under subject-dass is around 0.55–0.61 (from the 2026-04-09
  126-run sweep) and whose true mean under per-rec dass is **below
  chance at 0.44**.
- The "no-op" classification for Stress in §4.3–§4.5 and in
  `paper/figures/variance_analysis.json` is **stale**. Representation-
  level regeneration under per-rec dass is a TODO (requires re-running
  Stress FT with `--label dass --save-features` to produce
  `fold*_features.npz`).
- EEGMAT's behavioral result (Frozen 0.671 → FT 0.736, +6.5 pp) was
  also reclassified from "no-op" (representation level) to **mild
  injection** at the behavioral level.

**Updated four-dataset taxonomy**:

| Dataset | Representation Δ | Behavioral Δ (Frozen LP → FT) | Mode |
|---|---|---|---|
| ADFTD (AD) | +5 pp | +8.3 pp | **injection** |
| EEGMAT (task) | ~0 | +6.5 pp | **mild injection** |
| TDBRAIN (MDD) | −1.5 pp | ±0.2 pp | **silent erosion** (representation drops, BA unchanged) |
| Stress (DASS) | stale (re-run TODO) | **−16.2 pp** | **behavioral erosion** |

The taxonomy axis is **label–biomarker correspondence strength**, not
resting vs task-evoked: ADFTD (resting, AD has strong EEG biomarker) and
EEGMAT (task, well-known cognitive load biomarker) both show injection,
while TDBRAIN (resting, MDD biomarker is weak/contested) and Stress
(resting, DASS has no accepted EEG biomarker) both show erosion.

---

## 5. Key Findings

### F1: Subject leakage inflates results by 20+ points
All 3 FMs show consistent ~21-point BA drop from trial-level to subject-level CV. Published results using trial-level CV dramatically overestimate generalization.

### F2: Frozen FM features encode subject identity, not diagnostic signal
Frozen LaBraM allocates only 2.8–7.2% of its representation variance to
the label across the three datasets, while classifier BA is 0.66–0.75.
The representation is subject-dominated; the classifier extracts whatever
small label component exists via a supervised projection.

### F3: Classical features match FM performance
Random Forest on 156 hand-crafted features (band power + ratios + asymmetry) achieves 0.666 BA on stress — matching fine-tuned LaBraM (0.656). On ADFTD, RF (0.753) also matches FT LaBraM (0.752). No GPU, no pretraining needed.

### F4: Fine-tuning produces three N-invariant modes driven by label biology, not dataset size
Under N-controlled matched subsampling (100 draws/rung; see §4.5),
ADFTD's FT rewrite (Δ = +5 pp), Stress's no-op (Δ ≈ 0), and TDBRAIN's
erosion (Δ = −1.5 pp) are all **stable across N**: ADFTD delta varies
only within [+4.91, +5.68] across N ∈ [17, 65]; TDBRAIN delta varies
within [−1.06, −1.58] across N ∈ [35, 359]. A subject-level label-
permutation null confirms both effects exceed chance by ≥ 4× (ADFTD)
or lie on the *opposite* side of zero from the null (TDBRAIN). At
literally matched N=17, ADFTD with ~7% frozen baseline gains +5 pp from
FT while Stress with ~7% frozen baseline gains 0 pp — the three-mode
contrast is therefore a property of the *label*, not the training set
size. The earlier "dataset-size-dependent taxonomy" framing (2026-04-08
afternoon) and "fine-tuning helps strong signals but not weak ones"
framing were both superseded. The original canonical "2.76× ADFTD
rewrite" ratio is also an N-inflation artifact: the honest multiplier
at Stress-comparable N=17 is 1.87×, and the **additive +5 pp** framing
is the effect size that survives N-control.

### F5: The ~65% BA ceiling on stress is signal-limited, not method-limited
Both FMs and classical ML converge to ~65% BA on stress with 17 subjects. Three consistently misclassified subjects (P2, P5, P14) have clean signal quality — biological overlap, not artifacts.

### F6: Three independent FM benchmarks confirm our findings
Brain4FMs (2026), EEG-FM-Bench (2025), and AdaBrain-Bench (2025) all report FM failure on cross-subject affective/cognitive tasks, supporting our diagnosis.

---

## 6. Methods Attempted and Ruled Out

| Method | Result | Status |
|--------|--------|--------|
| Subject-adversarial training (GRL) | BA drops 0.656→0.537 on stress | Ruled out |
| LEAD-style subject loss | BA drops to 0.439 | Ruled out |
| Various adversarial lambda sweeps (0.01–1.0) | All worse than baseline | Ruled out |
| 82-rec dataset (imputed labels) | All methods degrade vs 70-rec | Use 70-rec only |

---

## 7. Infrastructure Built

| Component | Description |
|-----------|-------------|
| `pipeline/adftd_dataset.py` | ADFTD dataset loader with n_splits pseudo-recording support |
| `pipeline/tuab_dataset.py` | TUAB dataset loader (ready for when needed) |
| `pipeline/common_channels.py` | 19-channel common infrastructure with alias mapping |
| `train_ft.py --dataset adftd` | Cross-dataset fine-tuning support |
| `train_ft.py --save-features` | Test-fold feature extraction for post-hoc analysis |
| `train_classical.py` | Classical ML baseline (RF/SVM/LogReg/XGBoost) |
| `notebooks/Cross_Dataset_Signal_Strength.ipynb` | Unified cross-dataset diagnosis (eta², t-SNE, cosine similarity, RF) |
| `notebooks/FM_Representation_Diagnosis.ipynb` | Single-dataset FM diagnosis (3 FMs) |
| `notebooks/Recording_Quality_Analysis.ipynb` | Signal quality vs misclassification analysis |
| `scripts/download_adftd.sh` | ADFTD download from OpenNeuro S3 |
| `docs/related_work.md` | Living literature review (11 key citations) |
| `docs/fm_comparison_summary.md` | Comprehensive results summary |

---

## 8. Paper Narrative (reframed 2026-04-08 after EEG-FM-Bench prior-art pressure test, refined after EEGMAT positive control, corrected 2026-04-08 evening after N-controlled matched-subsample experiment)

**Title direction**: "Three Modes of Foundation-Model Fine-Tuning on
Clinical EEG: Injection, No-Op, and Erosion — Driven by Label Biology,
Not Training-Set Size" (revised working title; the previous "subject
identity dominance" framing is preempted by EEG-FM-Bench §4.3's
gradient-mass observation, and the "dataset-size-dependent" claim was
itself superseded after the matched-subsample experiment showed the
three effect directions are N-invariant and at matched N=17 ADFTD and
Stress have nearly identical frozen baselines yet opposite FT
responses).

**Closest methodological prior**: EEG-FM-Bench (Xiong et al. 2025,
arXiv:2508.17742). Their gradient-mass analysis (§4.3, Fig. 7 left)
observes that under pretrained-FT, gradient norms concentrate on the
input Temporal Embedding while the backbone receives small gradients.
Their CKA/RSA panels (Fig. 7 mid/right) compare Scratch-vs-Pretrained
trajectories — they are unsupervised, multi-task averaged, and not
sensitive to per-fold or per-dataset behavior. They never analyze
representation evolution as a function of dataset size, never use a
label-aware metric, and never observe fold-model drift.

**Core argument**: We introduce a label-aware variance decomposition —
the pooled label fraction $SS_\text{label}/SS_\text{total}$ over all
LaBraM dims, with cluster bootstrap CIs over subjects, a subject-level
PERMANOVA, and a paired frozen-vs-FT matched-subsample design with
100 draws per rung and a subject-level label-permutation null — that
complements EEG-FM-Bench's input-side gradient diagnostic on the
*output side*. Applied across **four** EEG datasets (three
between-subject clinical, one within-subject task-evoked), the metric
reveals **three qualitatively distinct fine-tuning modes whose effect
directions are invariant under N-controlled resampling** and therefore
properties of the label biology, not the training set size:

- **Small (Stress, n=70, between-subject)** — *projection only*. The
  pooled label fraction does not change (7.23% → 7.24%). Classifier BA
  still rises to 0.66 because the head learns a linear projection
  through an unchanged backbone. Output-side quantitative replication
  of EEG-FM-Bench's gradient-mass observation, restricted to a single
  small clinical dataset and using a label-aware metric.
- **EEGMAT (n=72, 36 subjects, within-subject)** — *also projection only*.
  Pooled label fraction 5.35% → 5.82% (×1.09, within bootstrap noise).
  Mixed-effects ICC for subject *increases* slightly under FT
  (0.56 → 0.63). Classifier BA = 0.736. Even though every EEGMAT
  subject contributes both rest and arithmetic recordings (so the
  label is biologically separable per subject and is *not* confounded
  with subject identity), fine-tuning still does not measurably
  rewrite the representation. **This falsifies the hypothesis that
  within-subject labels rescue FT rewriting: the difference between
  ADFTD's +5 pp injection and Stress/EEGMAT's 0 pp is not explained
  by label structure.** EEGMAT's crossed subject×label design is
  incompatible with the pure-label nested decomposition used for the
  matched-subsample experiment, so it has no matched-N rung — it
  appears only as a canonical full-N point in the taxonomy.
- **ADFTD (n=195, 65 subjects, between-subject)** — *clean label-signal
  injection*. Pooled label fraction goes 2.79% → 7.70% at full N;
  **under matched N ∈ {17, 25, 35, 50}, the absolute delta is stable
  at +5.22 to +5.68 pp** (ADFTD frozen at N=17 = 6.50%, FT = 12.18%).
  PERMANOVA p drops 0.050 → 0.001. The canonical 2.76× multiplier is
  inflated by the N-sensitivity of the pooled-fraction denominator —
  the **N-invariant +5 pp additive effect** is the honest effect size.
  This is the only dataset where FT genuinely rewrites the LaBraM
  representation, and it does so regardless of how many subjects the
  FT set contains.
- **TDBRAIN (n=734, 359 subjects, between-subject)** —
  ***active label-signal erosion***. Pooled label fraction drops
  2.97% → 1.47% at full N; **under matched N ∈ {35, 65, 150, 300},
  the absolute delta is stable at −1.06 to −1.58 pp**. At N=300, 100%
  of draws have FT < frozen, and the observed delta sits on the
  *opposite* side of zero from the permutation null (observed
  −1.53 pp vs null +0.12 pp) — FT is not merely failing to improve,
  it is actively removing label signal the frozen representation
  already carried. Classifier BA stays at 0.68 because the large
  sample size still gives the head enough local signal to read. This
  failure mode is invisible to any pipeline that averages over folds
  before computing the diagnostic and is a genuinely novel finding.

Classical band-power features match or beat fine-tuned LaBraM on both
Stress (RF 0.666 vs FT 0.656) and ADFTD (RF 0.753 vs FT 0.752),
reinforcing that the fine-tuning gains in the small-data regime are
projection-shaped, not representation-shaped.

**Contributions** (revised, in order of novelty):

1. **Three-mode N-invariant FT taxonomy**: *injection* (ADFTD,
   +5 pp stable across N ∈ [17, 65]), *no-op* (Stress and EEGMAT,
   Δ ≈ 0), *erosion* (TDBRAIN, −1.5 pp stable across N ∈ [35, 359],
   on opposite side of permutation null). Verified by paired
   matched-subsample resampling with 100 draws per rung and a
   subject-level label-permutation null. Inverts both the naive
   "more data → cleaner adaptation" expectation (TDBRAIN gets *worse*
   with FT) and the prior "size-dependent taxonomy" framing (the three
   modes persist at matched N=17, so the differences are label-biology
   not size). TDBRAIN active erosion is the headline.
2. **Label-aware, output-side, per-fold-aware variance decomposition**
   (pooled label fraction + subject-level cluster bootstrap +
   subject-level PERMANOVA) as the diagnostic that makes the taxonomy
   visible. Positioned explicitly as the complement to EEG-FM-Bench's
   gradient-mass and CKA/RSA analyses, not as a replacement.
3. **Empirical demonstration that classical band-power features match
   fine-tuned FMs on both Stress and ADFTD**, supporting the
   interpretation that small-data FT gains are projection-shaped.
4. **Per-fold nested-decomposition degeneracy guard** for small-
   subject datasets (the 1-subject-per-positive-class problem on
   Stress). Relegated to a methods footnote / supplement; the
   underlying df-collapse is elementary stats but the explicit
   programmatic guard is a practical contribution for downstream
   pipelines.
5. **EEGMAT within-subject negative control** showing that the
   projection-only failure mode at small n is *not* specific to
   between-subject labels. Strengthens the dataset-size-dependence
   claim by ruling out an alternative explanation that a reviewer
   would naturally raise ("FT fails on resting-state stress because
   the label IS the subject; try a within-subject task and FT will
   work").

---

## 9. Next Steps

### Immediate (high priority)
1. **Private dementia dataset** (~420 subjects, eyes-open resting-state, dementia/MCI/normal)
   - Provides 3 signal-strength levels within one dataset
   - Largest resting-state clinical EEG dataset in our study
   - Need to verify: multiple recordings per subject? Channel montage?
2. **TUAB consideration**: Not resting-state (clinical routine with HV/photic). Dropped from comparison. Could use as "clinical EEG" reference if needed.

### §4.7 Multi-model HP sweep on Stress (2026-04-11)

Comprehensive hyperparameter sweep across **3 FMs** (LaBraM, CBraMod, REVE) on
UCSD Stress with per-recording DASS labels. Grid: 3 learning rates (1e-5, 3e-5,
1e-4) × 2 encoder LR scales (0.1, 1.0) × 3 seeds (42, 123, 2024) = 18 runs per
model, 54 total. All runs completed.

**Results — Frozen LP vs Best FT:**

| Model | Frozen LP (8-seed) | Best FT (3-seed) | Best FT config | Δ |
|---|---|---|---|---|
| LaBraM | 0.605 ± 0.030 | 0.524 ± 0.008 | lr=1e-4, elrs=1.0 | −8.1 pp (erosion) |
| CBraMod | 0.452 ± 0.030 | 0.548 ± 0.026 | lr=1e-5, elrs=0.1 | +8.3 pp (injection) |
| REVE | 0.494 ± 0.017 | 0.577 ± 0.041 | lr=3e-5, elrs=0.1 | +8.0 pp (injection) |

**Key finding:** Erosion on Stress is **LaBraM-specific**. CBraMod and REVE show
injection (FT improves over frozen). This refutes the earlier hypothesis that
"erosion isn't LaBraM-specific" (CLAUDE.md §4 priority 3). The pattern suggests
erosion requires a strong frozen representation — LaBraM's 0.605 frozen BA is
much higher than CBraMod's 0.452 or REVE's 0.494.

**Implications for cross-dataset taxonomy:** The ADFTD/TDBRAIN/EEGMAT modes were
established with LaBraM only. Whether CBraMod/REVE show the same modes on those
datasets is an open question. The claim "FT mode is driven by label–biomarker
strength" is supported by cross-dataset LaBraM evidence, but cross-model evidence
currently only covers Stress.

**Note on LaBraM erosion magnitude:** The canonical recipe (lr=1e-5, elrs=0.1)
produces 0.443 ± 0.068 (−16.2 pp gap). The best HP config (lr=1e-4, elrs=1.0)
produces 0.524 ± 0.008 (−8.1 pp gap). Both are erosion, but the magnitude
depends on HP. The permutation null comparison (real FT ≈ null) was done with
the canonical recipe only.

**EEGMAT 3-seed study:** LaBraM FT on EEGMAT: BA = 0.731 ± 0.017 (seeds 42,
123, 2024). Features saved for variance analysis at
`results/studies/2026-04-10_eegmat_feat_multiseed/`.

### Paper preparation
3. Figures done: `cross_dataset_signal_strength.pdf` (pooled label
   fraction) and `label_subspace.pdf` (3×3 diagnostic + t-SNE).
4. Write methods and results sections using the corrected narrative.

### Task-evoked comparison (open research question)
5. **Should we add a task-evoked EEG dataset as a positive control?**
   Our current three datasets all have subject-level labels (AD or not,
   MDD or not, chronic stress or not). In task-evoked paradigms
   (motor imagery, emotion, mental arithmetic) each subject has
   recordings of multiple classes, so label is within-subject. The
   hypothesis to test: "on within-subject labels, the FM representation
   encodes the task strongly and fine-tuning works cleanly — the
   subject-dominance story is specific to between-subject labels."
   Candidate datasets: EEGMAT (mental arithmetic, paired rest/task per
   subject), SAM40 (stress arithmetic, paired), SEED emotion (within-
   subject), BCI Competition IV 2a (motor imagery, within-subject).
   This would turn the paper's negative finding into a mechanistic
   positive claim and is worth discussing with the advisor.

### Optional extensions
6. Within-subject longitudinal modeling (documented in
   `project_paper_strategy`).
7. Spectral-guided FiLM conditioning (proposed but not implemented).
