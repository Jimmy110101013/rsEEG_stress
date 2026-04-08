# CLAUDE.md (Master Architecture Blueprint)
## 0. Default settings
Use timm_eeg for conda env and install things inside it.

## 1. Project Overview

Resting-state EEG stress classification using EEG Foundation Models. 18 subjects, 82 recordings (after 400s filter), binary classification (normal vs increase).

### Key Finding: Subject Identity Dominance (corrected 2026-04-08)
Frozen LaBraM allocates only **7.2% of its 200-d representation variance
to the stress label** (pooled $SS_{\text{label}} / SS_{\text{total}}$),
with fine-tuning producing **no measurable change** (7.23% → 7.24%).
Cross-dataset comparison shows fine-tuning rewrites the representation
cleanly only on ADFTD (2.8% → 7.7%, +2.76×) and not on Stress or
TDBRAIN. Canonical numbers: `paper/figures/variance_analysis.json`.
Methodology: `docs/eta_squared_pipeline_explanation.md`.

### Current Results (subject-level CV, 70-rec)
| Model | Subject-Level BA | Trial-Level BA | Status |
|-------|-----------------|----------------|--------|
| LaBraM FT | **0.656** | 0.862 | Primary |
| Classical RF | 0.666 | — | Matches FM |
| CBraMod | 0.488 | 0.712 | Ruled out |
| REVE | 0.553 | 0.770 | Ruled out |

### Paper Strategy: Diagnosis + Solution
1. **Expose** trial-level vs subject-level inflation gap (30+ points)
2. **Diagnose** why FMs fail (subject identity dominance, variance decomposition)
3. **Propose** subject-adversarial training + neurophysiology-guided modulation
4. **Demonstrate** improvement on subject-level CV

---

## 2. Architecture

### Current Pipeline
```
EEG (.set) → StressEEGDataset (epoch + cache) → FM Backbone → Global Pool → Classifier
```

### Proposed: Subject-Adversarial + Spectral-Guided Framework
```
                     EEG Input
                    /         \
            FM Backbone    Spectral Extractor (theta/alpha/beta)
                |                    |
        [GRL → Subject Classifier]  |   ← adversarial: remove subject identity
                |                    |
            FM Features  ←── FiLM ──┘   ← expert knowledge: modulate features
                |
          Stress Classifier
```

**Loss**: `L = L_stress + lambda_adv * L_subject_adversarial + lambda_neuro * L_spectral_prediction`

### FM Extractors (`baseline/`)
All output unified `(B, embed_dim)`: LaBraM (200), CBraMod (200), REVE (512).
Factory: `create_extractor(name)` — never pass embed_dim explicitly.

### Dataset
- CSV: `data/comprehensive_labels_stress.csv` with Duration column
- Filter: `--max-duration 400` (matches reference paper)
- Labels: `subject-dass` mode (all recordings from increase patients → increase)
- Cache: `data/cache/*.pt` — preprocessed (M, 30, 2000) epoch tensors

### Evaluation
- **Subject-level CV**: StratifiedGroupKFold (5-fold, no subject leakage) — primary metric
- **Trial-level CV**: StratifiedKFold (5-fold, subject leakage) — paper comparison only

---

## 3. Key Files
| File | Purpose |
|------|---------|
| `train_ft.py` | Subject-level CV fine-tuning (main evaluation) |
| `train_trial.py` | Trial-level CV (paper comparison) |
| `train_lp.py` | Linear probing baseline |
| `pipeline/dataset.py` | Dataset, WindowDataset, RecordingGroupSampler |
| `src/model.py` | DecoupledStressModel (extract_pooled + classify) |
| `src/loss.py` | Focal loss, ranking loss |
| `notebooks/FM_Representation_Diagnosis.ipynb` | Diagnostic analysis (t-SNE, variance, effect sizes) |
| `docs/related_work.md` | Living literature review |

---

## 4. Implementation Priorities

### Next: Subject-Adversarial Training
- Add GRL (gradient reversal layer) after FM backbone
- Subject discriminator: 2-3 FC layers predicting subject ID
- Lambda scheduling: start 0, ramp to 0.05-0.1 over training
- Goal: reduce subject eta-squared while maintaining/improving stress classification

### Future: Spectral-Guided Modulation
- Compute theta/alpha/beta power per epoch
- FiLM conditioning OR cross-attention to modulate FM features
- Auxiliary loss: FM features should predict spectral features (neurophysiology regularization)