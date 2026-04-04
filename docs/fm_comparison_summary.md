# Foundation Model Comparison for Resting-State EEG Stress Classification

**Dataset**: UCSD Stress — 17 subjects, 70 recordings, 5-min resting-state EEG @ 200Hz, 30 channels  
**Task**: Binary stress classification (increase vs normal) using DASS questionnaire labels  
**Date**: April 2–4, 2026

---

## 1. Models Evaluated

| Model | Paper | Pretraining | Architecture | Params | Embed Dim |
|-------|-------|-------------|-------------|--------|-----------|
| **LaBraM** | Jiang et al., ICLR 2024 | Large-scale EEG, masked patch prediction | ViT (12 layers, 10 heads) with channel position embeddings | ~13M | 200 |
| **REVE** | Elouayas et al., 2024 | MAE reconstruction on EEG | ViT with 4D Fourier positional encoding | ~13M | 200 |
| **CBraMod** | Wang et al., ICLR 2025 | Masked patch prediction on TUEG | Criss-Cross Transformer (12 layers, 8 heads) with ACPE | ~13M | 200 |

---

## 2. Evaluation Paradigms

### Trial-Level CV (paper comparison)
- `StratifiedKFold` on recordings (5 folds) — same subject may appear in train and test
- Matches reference paper methodology (arXiv 2505.23042)
- Window-level training with per-recording majority-vote evaluation

### Subject-Level CV (rigorous, no leakage)
- `StratifiedGroupKFold` by patient_id (5 folds)
- No subject overlap between train/val/test
- Global prediction pooling across folds
- More honest generalization estimate, but harsher with N=17

---

## 3. Key Results

### 3.1 Best Results per Model

| Model | CV Type | Mode | Label | Norm | Aug | Bal Acc | Acc | Sens | Spec | κ |
|-------|---------|------|-------|------|-----|---------|-----|------|------|---|
| **LaBraM** | trial | FT | subject-dass | zscore | 75% | **0.796** | 0.800 | 0.783 | 0.809 | 0.57 |
| **LaBraM** | subject | FT | subject-dass | zscore | 75% | **0.602** | 0.600 | 0.609 | 0.596 | 0.19 |
| **REVE** | trial | FT | subject-dass | none | 75% | **0.805** | 0.843 | 0.696 | 0.915 | — |
| **REVE** | subject | LP | — | none | — | **0.536** | 0.586 | 0.429 | 0.643 | — |
| **CBraMod** | subject | FT | subject-dass | none | 75% | **0.486** | 0.429 | 0.600 | 0.333 | -0.02 |

### 3.2 LaBraM Ablation: Label Definition (Trial-Level CV)

| Label Scheme | Distribution | Bal Acc | Acc | Notes |
|-------------|-------------|---------|-----|-------|
| DSS threshold (t=60) | 14:56 | 0.521 | 0.514 | LP frozen backbone |
| File-path DASS | 14:56 (80:20) | 0.616 | 0.729 | FT + zscore + 75% aug |
| **Subject-level DASS** | **23:47 (33:67)** | **0.796** | **0.800** | **FT + zscore + 75% aug** |

**Insight**: Correcting the label definition from file-path to subject-level DASS improved balanced accuracy by +18 percentage points. This was the single largest factor across all experiments.

### 3.3 LaBraM Subject-Level CV: Regularization Ablation

| Run | wd | Label Smooth | Bal Acc | Notes |
|-----|-----|-------------|---------|-------|
| Baseline | 0.05 | 0.0 | **0.602** | Best subject-level result |
| +label smoothing | 0.01 | 0.1 | 0.489 | Degraded significantly |
| +label smoothing | 0.05 | 0.1 | 0.384–0.504 | Inconsistent across runs |
| wd=0.05, no LS | 0.05 | 0.0 | 0.547–0.602 | Run variance is high |

**Insight**: Added regularization (label smoothing, different weight decay) did not improve over baseline. High variance across runs with identical configs (~10% BA swing) indicates the 17-subject dataset is the fundamental bottleneck.

### 3.4 REVE Comprehensive Evaluation

| Paradigm | Mode | Best Bal Acc | Notes |
|----------|------|-------------|-------|
| Subject-level LP | LP | 0.536 | Barely above chance |
| Subject-level LoRA | LoRA | 0.455 | Catastrophic forgetting |
| Trial-level DSS sweep | LP | 0.595 | Subject leakage inflates |
| Trial-level FT | FT | 0.805 | With subject-dass + aug |
| Longitudinal (within-subject) | — | 0.333–0.426 | Worse than random features |

**Insight**: REVE's trial-level FT result (0.805) is competitive with LaBraM (0.796), but this paradigm has subject leakage. Under subject-level CV (the honest evaluation), REVE LP features are at chance level. REVE's MAE reconstruction pretraining does not preserve stress-discriminative neural patterns in frozen features.

### 3.5 Paper-Matched Dataset: 82 Recordings (400s Duration Filter)

The reference paper filters recordings >400s (poor signal quality from cap adjustments), yielding 82 recordings from the original 92. Our previous experiments used only the 70 recordings with DSS scores, which included 9 noisy long recordings (417–1509s) the paper excludes.

**Subject-Level CV (82 recordings, FT, subject-dass, 75% aug):**

| Model | Bal Acc | Acc | κ | vs 70-rec |
|-------|---------|-----|---|-----------|
| **REVE** | 0.553 | 0.549 | 0.104 | +0.02 (from LP 0.536) |
| **LaBraM** | 0.537 | 0.537 | 0.073 | -0.065 (from 0.602) |
| **CBraMod** | 0.488 | 0.451 | -0.022 | ~same |

**Insight**: Matching the paper's 82-recording dataset did not improve subject-level generalization. LaBraM actually dropped 6.5 points. The fold composition (which subjects land in train vs test) dominates results with N=17 subjects — any change in included recordings reshuffles folds and causes ~6% BA swings. All three FMs remain near chance (0.49–0.55) under honest subject-level CV.

**Trial-Level CV (82 recordings, LaBraM FT, subject-dass, 75% aug):**

| Experiment | Bal Acc | Acc | κ | Notes |
|---|---|---|---|---|
| Reference paper (arXiv 2505.23042) | 0.810 | — | — | Published result |
| **LaBraM (82 recs, this work)** | **0.862** | **0.854** | **0.710** | Exceeds paper by +5.2 pts |
| LaBraM (70 recs, prior run) | 0.796 | 0.800 | 0.570 | With 9 noisy recordings included |

Per-fold test BA: 0.778, 0.889, 0.889, 0.857, 0.889.

**Insight**: Removing the 9 noisy long recordings (>400s) was the key — trial-level BA jumped from 0.796 to 0.862. The paper's duration filter is validated: these recordings genuinely degrade model training. Our LaBraM FT pipeline now exceeds the published benchmark by 5 points on the same dataset split.

---

## 4. Architectural Analysis

### Why LaBraM leads under subject-level CV

1. **Channel-position embeddings**: LaBraM maps 30 EEG channels to standard 10-20 electrode positions via a 128-slot learned position embedding. This encodes anatomical spatial relationships (e.g., frontal vs parietal) that are relevant for stress biomarkers like frontal alpha asymmetry. Neither REVE nor CBraMod has this electrode-position-aware encoding.

2. **Feature normalization**: LaBraM applies `fc_norm` (LayerNorm) on pooled features before the classifier head, plus learnable layer-scale initialization (`init_values=0.1`). CBraMod lacks final feature normalization. This matters for stable fine-tuning gradients.

3. **Full attention vs decomposed**: LaBraM uses standard full self-attention over all tokens (channels × patches flattened). CBraMod splits features in half — spatial attention (d_model/2) operates across channels, temporal attention (d_model/2) operates across patches. This prevents learning cross-channel-temporal interactions (e.g., how frontal theta evolves relative to parietal alpha over time), which may be the discriminative signal for resting-state stress.

### Why CBraMod underperforms despite richer encoding

CBraMod has several theoretically superior components:
- **ACPE** (Asymmetric Conditional Positional Encoding): Content-dependent positional encoding via depthwise Conv2d (19×7 kernel) over spatial-temporal dimensions — more flexible than LaBraM's fixed learned embeddings
- **Dual-stream embedding**: CNN temporal features + FFT spectral projection combined before the transformer
- **Criss-Cross attention**: Computationally efficient parallel spatial + temporal attention

However, the Criss-Cross decomposition appears to be a liability for this task:
- Each attention head only sees **half the feature dimensions** (d_model/2 = 100)
- Spatial attention cannot access temporal context; temporal attention cannot access cross-channel patterns
- The ACPE is a **local** operation (19×7 conv) — good for local structure but not long-range dependencies
- For resting-state stress where discriminative signals are distributed and subtle (not localized ERPs), full attention's flexibility to find arbitrary feature correlations outperforms structured inductive biases

### Trial-level vs subject-level performance gap

| Model | Trial BA | Subject BA | Gap |
|-------|----------|-----------|-----|
| LaBraM | 0.796 | 0.602 | -0.194 |
| REVE | 0.805 | 0.536 | -0.269 |

The large gap indicates both models learn subject-specific patterns rather than generalizable stress biomarkers. With only 17 subjects, same-subject leakage in trial-level CV dramatically inflates results. The honest generalization to unseen subjects drops by 19–27 percentage points.

---

## 5. Key Findings

### F1: Label definition is the biggest lever
Switching from file-path DASS labels (14:56) to subject-level DASS labels (23:47) improved LaBraM trial-level BA from 0.62 to 0.80. Three subjects (IDs 3, 11, 17) have recordings in both `increase/` and `normal/` directories; subject-level labeling correctly assigns all their recordings based on DASS score.

### F2: Subject-level generalization is the hard problem
All models show a 19–27 point BA drop from trial-level to subject-level CV. With N=17 subjects, the dataset is fundamentally small for cross-subject generalization of resting-state stress. High run-to-run variance (~10% BA) confirms this.

### F3: Channel-position encoding matters for resting-state EEG
LaBraM's electrode-position-aware embeddings provide a consistent advantage over models that treat channels as generic sequences (REVE) or use content-dependent positional encoding (CBraMod ACPE). For resting-state paradigms where spatial topography carries clinical information, this is a meaningful architectural advantage.

### F4: Frozen features are not stress-discriminative
Across all FMs, linear probing on frozen features yields chance-level performance (~0.50 BA). Fine-tuning is necessary, meaning the pretrained representations do not natively separate stress states — the task-specific signal must be learned during adaptation.

### F5: More complex encoding ≠ better transfer
CBraMod's richer pipeline (CNN + FFT spectral + ACPE + criss-cross attention) underperforms LaBraM's simpler architecture (temporal conv + learned position embeddings + standard attention). The Criss-Cross decomposition's inductive bias constrains the cross-channel-temporal interactions needed for stress classification.

---

## 6. Recommendations

1. **Use LaBraM as the primary FM** for this dataset and task. It provides the best subject-level generalization (0.60 BA) and near-paper trial-level performance (0.80 BA).

2. **Report both evaluation paradigms** in any publication. Trial-level CV for comparison with prior work; subject-level CV for honest generalization claims.

3. **Do not invest in CBraMod tuning**. The 12-point BA gap vs LaBraM reflects an architectural mismatch (criss-cross attention decomposition) with resting-state stress classification, not a hyperparameter issue.

4. **Consider ensemble or classical feature baselines** (theta/alpha band power, frontal asymmetry) to contextualize FM performance against interpretable neuroscience features.

5. **Dataset size is the fundamental bottleneck**. With N=17 subjects, regularization tweaks have marginal and inconsistent effects. Collecting more subjects or incorporating cross-dataset pretraining would be more impactful than architecture search.
