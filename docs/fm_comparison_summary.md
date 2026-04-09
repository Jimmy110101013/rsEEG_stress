# Foundation Model Comparison for Resting-State EEG Stress Classification

**Dataset**: UCSD Stress — 18 subjects, 70 recordings (with DSS) / 82 recordings (400s filter), 5-min resting-state EEG @ 200Hz, 30 channels  
**Task**: Binary stress classification (increase vs normal) using subject-DASS labels  
**Date**: April 2–5, 2026

> ### ⚠️ Deprecated 2026-04-10 — historical reference only
>
> Every number in this document is a **single-seed** result under
> **`--label subject-dass`**, which was reclassified as a trait-memorization
> artifact on 2026-04-09. Canonical numbers (LaBraM FT 0.656, trial
> LaBraM 0.862, classical RF 0.666) are not reproducible under the
> honest per-recording DASS protocol and were further affected by
> cuDNN non-determinism amplifying single-seed noise into ±10 pp
> swings. Multi-seed Stress FT under per-rec DASS lands at
> **0.443 ± 0.068** (erosion), and Frozen LP on the same features hits
> **0.605 ± 0.030**. See:
>
> - `CLAUDE.md` headline finding (current state)
> - `docs/progress.md` §4.6 (2026-04-10 reclassification writeup)
> - `results/studies/2026-04-10_stress_erosion/analysis.json` (numbers)
>
> This file is kept for historical context — do not cite from it directly.

---

## 1. Models Evaluated

| Model | Paper | Architecture | Embed Dim |
|-------|-------|-------------|-----------|
| **LaBraM** | Jiang et al., ICLR 2024 | ViT (12L, 10H) + channel position embeddings | 200 |
| **REVE** | Elouayas et al., 2024 | ViT + 4D Fourier PE + MAE pretraining | 512 |
| **CBraMod** | Wang et al., ICLR 2025 | Criss-Cross Transformer (12L, 8H) + ACPE | 200 |
| **Classical ML** | — | RF / SVM / LogReg on 156 hand-crafted features | — |

---

## 2. Evaluation Paradigms

- **Subject-Level CV** (primary): `StratifiedGroupKFold` by patient_id (5 folds). No subject leakage.
- **Trial-Level CV** (comparison): `StratifiedKFold` on recordings (5 folds). Subject leakage allowed.

---

## 3. Main Results

### 3.1 Subject-Level CV (70 recordings — primary dataset)

| Model | Mode | Bal Acc | Acc | Kappa |
|-------|------|---------|-----|-------|
| **RF (classical)** | 156 features | **0.666** | 0.686 | 0.318 |
| **LaBraM** | FT | **0.656** | 0.657 | 0.286 |
| LogReg L2 (classical) | 156 features | 0.624 | 0.629 | 0.227 |
| XGBoost (classical) | 156 features | 0.612 | 0.657 | 0.223 |
| LaBraM + adv=1.0 | FT+GRL | 0.558 | 0.586 | 0.110 |
| LaBraM + adv=0.1 | FT+GRL | 0.537 | 0.557 | 0.069 |
| SVM RBF (classical) | 156 features | 0.580 | 0.600 | 0.150 |

### 3.2 Subject-Level CV (82 recordings — with imputed labels)

| Model | Mode | Bal Acc | Acc | Notes |
|-------|------|---------|-----|-------|
| LaBraM + adv=0.1 | FT+GRL | 0.564 | 0.561 | Best on 82-rec |
| LaBraM + adv=1.0 | FT+GRL | 0.562 | 0.561 | |
| REVE | FT | 0.553 | 0.549 | |
| LaBraM | FT | 0.537 | 0.537 | |
| CBraMod | FT | 0.488 | 0.451 | Near chance |
| LaBraM + LEAD loss | FT | 0.439 | 0.427 | Worse than baseline |
| RF (classical) | 156 features | 0.423 | 0.427 | Collapsed |

**Insight**: 12 extra recordings with imputed subject-DASS labels degrade all methods. 70-rec is the clean dataset.

### 3.3 Trial-Level CV (82 recordings — paper comparison)

| Model | Bal Acc | Acc | vs Reference Paper |
|-------|---------|-----|-------------------|
| **LaBraM** | **0.862** | 0.854 | +5.2 pts over paper's 0.810 |
| REVE | 0.770 | 0.768 | |
| CBraMod | 0.712 | 0.707 | |

### 3.4 Subject Leakage Gap

| Model | Trial BA | Subject BA | Gap |
|-------|----------|-----------|-----|
| LaBraM | 0.862 | 0.656 | **-20.6 pts** |
| REVE | 0.770 | 0.553 | **-21.7 pts** |
| CBraMod | 0.712 | 0.488 | **-22.4 pts** |

---

## 4. Representation Diagnosis

### 4.1 Variance decomposition — SUPERSEDED

The original naive one-way η² subject/label ratio was structurally biased
(subjects are pure-label, so subject-SS mechanically contains label-SS).
The reviewer-ready reformulation uses the **pooled label fraction**
$SS_{\text{label}} / SS_{\text{total}}$ summed over feature dims. Only
LaBraM was carried forward to the cross-dataset analysis; CBraMod and
REVE were ruled out for stress and not re-run with the corrected method.

**Canonical current numbers**: `paper/figures/variance_analysis.json` and
`docs/eta_squared_pipeline_explanation.md`. For Stress frozen LaBraM the
pooled label fraction is 7.23%, and fine-tuning does not change it.

### 4.2 Effect Size Comparison

| Feature Source | Cohen's d (normal vs increase) |
|---------------|-------------------------------|
| FM LDA projection | 2.9–5.2 (inflated by high-dim overfitting) |
| **Theta band power** | **1.14** (honest 1D feature) |
| Alpha band power | 0.37 |
| Beta band power | 0.22 |
| Theta/Alpha ratio | 0.01 |

### 4.3 Cosine Similarity

| FM | Within-Subject | Between-Subject | Subject Gap | Label Gap (cross-subj) |
|----|---------------|----------------|-------------|----------------------|
| LaBraM | 0.981 | 0.944 | 0.038 | -0.005 |
| REVE | 0.922 | 0.840 | 0.082 | -0.019 |
| CBraMod | 0.999 | 0.996 | 0.002 | -0.001 |

Label gap near zero = no stress signal in cross-subject features.

### 4.4 Consistently Misclassified Subjects

| Subject | Label | Error Rate | Signal Quality |
|---------|-------|------------|---------------|
| P2 | Normal | 100% (all 3 FMs wrong) | Normal |
| P5 | Normal | 86% | Normal |
| P14 | Normal | 83% | Normal |

ICA was applied by dataset provider. Residual artifact analysis (frontal eye blink, temporal muscle) shows no quality differences vs correctly classified subjects. Misclassification is biological, not artifactual.

---

## 5. Methods Attempted for Cross-Subject Improvement

| Method | Bal Acc | Effect |
|--------|---------|--------|
| Baseline FT | 0.656 | — |
| Subject-adversarial (GRL, λ=0.1) | 0.537 | -12 pts (strips useful signal) |
| Subject-adversarial (GRL, λ=1.0) | 0.558 | -10 pts |
| LEAD-style subject loss | 0.439 | -22 pts (82-rec) |
| Classical ML (RF, 156 features) | 0.666 | +1 pt (matches FM with 0 GPU) |

---

## 6. Classical ML Feature Importance (RF)

Top features from Random Forest (70-rec):
1. Alpha power at right hemisphere (T4, C4, CP4, P4) — stress lateralization
2. Alpha power at frontal (Fp2, F8) — frontal asymmetry
3. Beta power at temporal/parietal (T6, P4, Pz)
4. Theta/Alpha ratio at TP8

**Alpha band dominates** (13 of top 20 features), consistent with stress neuroscience literature.

---

## 7. Key Findings

### F1: Subject leakage inflates results by 20+ points
All 3 FMs show consistent ~21 point BA drop from trial-level to subject-level CV. Published results using trial-level CV dramatically overestimate real-world generalization.

### F2: FM features encode subject identity, not stress
Frozen LaBraM allocates 7.23% of its total representation variance to the
stress label on UCSD Stress (pooled label fraction from
`paper/figures/variance_analysis.json`), and fine-tuning does not change
this fraction (7.23% → 7.24%). The classifier achieves BA=0.656 by
reading a projection through an unchanged representation, not by
reshaping it. The original "34–37× subject/label ratio" figure was a
naive one-way η² artifact and is no longer cited.

### F3: Classical features match FM performance
Random Forest on 156 hand-crafted features (band power + ratios + asymmetry) achieves 0.666 BA — matching LaBraM's 0.656 BA. No GPU, no pretraining, no fine-tuning needed.

### F4: The ~65% ceiling is dataset-limited, not method-limited
Both FMs and classical ML converge to ~65% BA with 18 subjects. Subject-adversarial training, LEAD-style loss, and various regularization strategies all fail to break this ceiling. Three Normal subjects (P2, P5, P14) with clean signal quality are consistently misclassified — biological overlap, not data quality.

### F5: Dataset composition matters more than method
70-rec (clean DASS labels) gives 0.656 BA. Adding 12 imputed recordings drops to 0.537. The 82-rec imputed labels hurt more than extra data helps.

### F6: Alpha lateralization is the dominant stress feature
RF feature importance and FM diagnosis both point to right-hemisphere alpha power as the most discriminative feature for resting-state stress classification.
