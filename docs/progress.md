# Research Progress Report

**Project**: EEG Foundation Model Evaluation for Resting-State Stress Classification
**Period**: April 2–8, 2026
**Author**: Jimmy Lin

---

## 1. Executive Summary

We conducted a systematic evaluation of EEG foundation models (FMs) for
cross-subject resting-state EEG classification across three datasets:
UCSD Stress (17 subjects), ADFTD (65), TDBRAIN (359). Our reviewer-ready
metric is the pooled label fraction $SS_{\text{label}} / SS_{\text{total}}$
in the LaBraM 200-d representation. Fine-tuning only measurably rewrites
the representation on the medium-sized ADFTD (2.8% → 7.7%, +2.76×); on
the small Stress dataset it does not change the representation at all
(7.2% → 7.2%), and on the large-but-subject-saturated TDBRAIN it actually
dilutes the pooled label fraction (3.0% → 1.5%). Classifier BA rises
modestly on Stress (0.66) despite no representation change, because the
classifier head learns to read a pre-existing projection. See
`docs/eta_squared_pipeline_explanation.md` for the full methodology.

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

| Dataset | n rec / n subj | BA | Frozen label frac | FT pooled label frac | Change |
|---|---|---|---|---|---|
| Stress  | 70 / 17  | 0.656 | **7.23%** | **7.24%** | → unchanged |
| ADFTD   | 195 / 65 | 0.752 | 2.79% | **7.70%** | **↑ 2.76×** |
| TDBRAIN | 734 / 359| 0.681 | 2.97% | 1.47% | ↓ 0.49× |

**Canonical source**: `paper/figures/variance_analysis.json`.

Supporting metrics (cluster-bootstrapped CIs, REML mixed-effects ICC,
subject-level PERMANOVA) all live in the same JSON and move in the same
directions.

### 4.4 Interpretation

Fine-tuning's effect on the LaBraM representation is dataset-size
dependent, not a uniform "reduces subject dominance" story:

- **Stress (n=70)**: no measurable change in representation structure.
  Classifier BA=0.66 is achieved by reading a pre-existing projection
  through an unchanged representation.
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

### F4: Fine-tuning's effect is dataset-size dependent, not signal-strength dependent
Only ADFTD shows a clean FT-induced increase in the pooled label fraction
(2.8% → 7.7%, +2.76×). Stress is unchanged (7.2% → 7.2%) and TDBRAIN is
diluted (3.0% → 1.5%). The earlier "fine-tuning helps strong signals but
not weak ones" framing was based on per-dim averaged ω², which did not
survive the switch to the reviewer-ready pooled metric.

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

## 8. Paper Narrative

**Title direction**: "Subject Identity Dominance Bounds Foundation-Model
Performance on Resting-State EEG: Diagnosis and Implications for
Personalized Modeling" (working title in `project_paper_strategy`).

**Core argument**: In resting-state EEG, the label is a subject-level
property (each subject is either AD or HC, each subject is either MDD or
HC, each subject has a fixed DASS score). Foundation models pretrained
on masked-patch prediction learn representations where subject identity
dominates the variance (label fraction <8% in all three datasets).
Fine-tuning only measurably rewrites this representation on the medium-
sized ADFTD dataset; on small (Stress, n=70) and very large but subject-
saturated (TDBRAIN, 359 subjects) datasets, fine-tuning does not change
the representation's variance structure and classifier BA gains come
from the head learning a supervised projection, not from representation
reshaping. Classical band-power features match or beat fine-tuned LaBraM
on Stress and ADFTD, questioning the added value of FM pretraining for
resting-state subject-level classification.

**Contributions**:
1. Reviewer-ready variance-decomposition methodology (pooled label
   fraction + cluster bootstrap + subject-level PERMANOVA + mixed
   effects ICC) for measuring whether FM fine-tuning rewrites
   representations.
2. Identification of a per-fold degeneracy in small-subject datasets
   (Stress's 1-subject-per-positive-class problem) with a guard that
   flags it in future analyses.
3. Cross-dataset evidence that fine-tuning effects on FM representations
   are dataset-size dependent, not uniformly "helpful".
4. Evidence that classical features match fine-tuned FMs on both
   Stress and ADFTD.
5. Open question: does the pattern hold on task-evoked EEG, where
   labels are within-subject rather than subject-level? (see §9.)

---

## 9. Next Steps

### Immediate (high priority)
1. **Private dementia dataset** (~420 subjects, eyes-open resting-state, dementia/MCI/normal)
   - Provides 3 signal-strength levels within one dataset
   - Largest resting-state clinical EEG dataset in our study
   - Need to verify: multiple recordings per subject? Channel montage?
2. **TUAB consideration**: Not resting-state (clinical routine with HV/photic). Dropped from comparison. Could use as "clinical EEG" reference if needed.

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
