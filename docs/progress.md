# Research Progress Report

**Project**: EEG Foundation Model Evaluation for Resting-State Stress Classification
**Period**: April 2–7, 2026
**Author**: Jimmy Lin

---

## 1. Executive Summary

We conducted a systematic evaluation of EEG foundation models (FMs) for cross-subject resting-state stress classification. Our findings reveal that FMs encode subject identity 11–13x more than task-relevant diagnostic signals — a limitation that persists across both weak-signal (chronic stress) and strong-signal (Alzheimer's disease) resting-state EEG tasks. Fine-tuning partially recovers diagnostic signal for strong-signal tasks but fails for weak-signal tasks, establishing signal strength as the fundamental bottleneck.

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

### 4.3 Frozen Feature Variance Decomposition (eta-squared)

**Frozen LaBraM features (19 common channels):**

| Dataset | N subj | eta²(subject) | eta²(label) | Ratio |
|---------|--------|---------------|-------------|-------|
| UCSD Stress | 17 | 0.649 | 0.059 | **11.0x** |
| ADFTD full | 65 | 0.783 | 0.024 | 33.0x |
| ADFTD N-matched (10 draws) | 17 | 0.748 ± 0.042 | 0.062 ± 0.016 | **13.0x ± 3.9x** |

When controlling for subject count (N=17), both datasets show similar subject-to-label ratio (~11–13x). Frozen FM features are equally uninformative for both tasks.

### 4.4 Frozen vs Fine-Tuned Feature Comparison

| Dataset | Mode | eta²(subject) | eta²(label) | Ratio | BA |
|---------|------|---------------|-------------|-------|----|
| Stress (N=17) | Frozen | 0.649 | 0.059 | 11.0x | — |
| Stress (N=17) | **FT** | 0.874 | 0.069 | **12.8x** | 0.656 |
| ADFTD matched (N=17) | Frozen | 0.748 | 0.062 | 13.0x | — |
| ADFTD matched (N=17) | **FT** | 0.930 | 0.125 | **7.7x** | 0.752 |

**Critical finding**: Fine-tuning increases both subject and label encoding. For Alzheimer's, label encoding grows faster (ratio drops 13x→7.7x). For stress, the ratio stays flat (11x→12.8x) — fine-tuning cannot create signal that isn't in the raw EEG.

---

## 5. Key Findings

### F1: Subject leakage inflates results by 20+ points
All 3 FMs show consistent ~21-point BA drop from trial-level to subject-level CV. Published results using trial-level CV dramatically overestimate generalization.

### F2: Frozen FM features encode subject identity, not diagnostic signal
Variance decomposition shows subject identity explains 11–13x more feature variance than task labels. This is consistent across architecturally different FMs (LaBraM, REVE, CBraMod) and across both weak-signal (stress) and strong-signal (Alzheimer's) tasks.

### F3: Classical features match FM performance
Random Forest on 156 hand-crafted features (band power + ratios + asymmetry) achieves 0.666 BA on stress — matching fine-tuned LaBraM (0.656). On ADFTD, RF (0.753) also matches FT LaBraM (0.752). No GPU, no pretraining needed.

### F4: Fine-tuning helps strong signals but not weak ones
Fine-tuning reduces the subject-to-label dominance ratio from 13x to 7.7x for Alzheimer's (eta²(label) doubles). For stress, the ratio remains at ~12x — the diagnostic signal is too weak to amplify.

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

**Title direction**: "Signal Strength Determines EEG Foundation Model Utility: A Cross-Dataset Diagnosis of Subject Identity Dominance in Resting-State Classification"

**Core argument**: EEG foundation models pretrained on masked patch prediction learn representations dominated by subject identity (11–13x over diagnostic signal). Fine-tuning can partially recover strong clinical signals (Alzheimer's: ratio drops to 7.7x, BA=75%) but fails for weak psychological signals (chronic stress: ratio stays at 12.8x, BA=65%). Classical band-power features match FM performance in both cases, questioning the added value of foundation model pretraining for resting-state EEG classification.

**Contributions**:
1. First rigorous multi-FM cross-subject evaluation for resting-state stress detection
2. Novel quantitative diagnosis of subject-identity dominance via variance decomposition (eta-squared)
3. Cross-dataset validation showing the pattern is universal, not dataset-specific
4. Frozen vs fine-tuned comparison revealing signal-strength-dependent adaptation
5. Evidence that classical features match FMs with zero computational overhead

---

## 9. Next Steps

### Immediate (high priority)
1. **Private dementia dataset** (~420 subjects, eyes-open resting-state, dementia/MCI/normal)
   - Provides 3 signal-strength levels within one dataset
   - Largest resting-state clinical EEG dataset in our study
   - Need to verify: multiple recordings per subject? Channel montage?
2. **TUAB consideration**: Not resting-state (clinical routine with HV/photic). Dropped from comparison. Could use as "clinical EEG" reference if needed.

### Paper preparation
3. Generate publication-quality figures (signal strength spectrum, frozen vs FT comparison)
4. Write methods and results sections
5. Statistical tests (permutation tests for eta-squared, confidence intervals for BA)

### Optional extensions
6. TUAB/TUEP as non-resting-state reference point
7. Spectral-guided FiLM conditioning (proposed but not implemented)
8. Contrastive learning for subject-invariant features
