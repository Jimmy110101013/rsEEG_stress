# Related Work (Living Document)

*Last updated: 2026-04-05. This document is incrementally updated as we discover more relevant literature.*

---

## 1. EEG Foundation Models

### 1.1 LaBraM (Jiang et al., ICLR 2024)
- **Architecture**: ViT with channel-position embeddings (10-20 system), VQ tokenizer
- **Pretraining**: ~2,000 hrs mixed EEG, masked patch prediction
- **Downstream use**: Predominantly task-based/event-related EEG. Only 2 known resting-state applications: stress (our reference paper) and Alzheimer's (LEAD comparison)
- **Ref**: arXiv:2405.18765

### 1.2 CBraMod (Wang et al., ICLR 2025)
- **Architecture**: Criss-Cross Transformer (parallel spatial + temporal attention on split feature halves), ACPE, CNN + FFT spectral patch embedding
- **Pretraining**: TUEG data, masked patch modeling
- **Benchmark standing**: Top performer in EEG-FM-Bench on stress/emotion tasks — but likely evaluated with trial-level CV
- **Our finding**: 48.8% BA under subject-level CV (chance level), 71.2% trial-level
- **Ref**: arXiv:2412.07236

### 1.3 REVE (Elouayas et al., 2024)
- **Architecture**: ViT with 4D Fourier positional encoding, MAE reconstruction pretraining
- **Our finding**: Frozen features are anti-informative for stress (worse than random). FT achieves ~80% trial-level but ~55% subject-level
- **Ref**: elouayas/reve_eeg

### 1.4 Other Notable FMs
- **EEGPT**: Top performer alongside CBraMod in EEG-FM-Bench
- **BIOT, BENDR, CSBrain**: Included in benchmarks but not evaluated in our work
- **NeuroLM (ICLR 2025)**: Universal multi-task FM, demonstrates multi-task learning benefits
- **EEG-DINO (MICCAI 2025)**: Hierarchical self-distillation, DINO-v2 inspired
- **ALFEE**: 25,000 hrs pretraining, channel + temporal encoder with task-specific token dictionary

---

## 2. EEG FM Benchmarks

### 2.1 EEG-FM-Bench (Xu et al., 2025)
- **Scope**: 7 EEG-FMs, 14 datasets, 10 paradigms
- **Critical findings**:
  1. Frozen backbone = near chance across ALL models (linear probing fails universally)
  2. Full fine-tuning is necessary; pretrained weights useful only as initialization
  3. CBraMod and EEGPT are top performers with FT
  4. Multi-task FT helps in data-scarce settings
- **Relevance**: Validates our LP failure results. However, their stress evaluation may use trial-level splits
- **Ref**: arXiv:2508.17742

### 2.2 AdaBrain-Bench (2025)
- **Scope**: BIOT, EEGPT, LaBraM, CBraMod across multiple tasks
- **Critical finding**: *"Emotion recognition and motor imagery yielded suboptimal performance"* cross-subject due to *"highly individual neural responses"*
- **Numbers**: SEED emotion — LaBraM 55.78%, CBraMod 51.11% BA (3-class); EEGMAT workload — LaBraM 85.83%, CBraMod 88.89% BA (binary, task-evoked, easier)
- **Relevance**: Confirms FMs fail on affect tasks but succeed on motor/cognitive load tasks
- **Ref**: arXiv:2507.09882

### 2.3 Brain4FMs (Feb 2026)
- **Scope**: 15 models (BENDR, BIOT, LaBraM, CBraMod, EEGPT, REVE, BrainWave, NeuroGPT, BrainOmni, etc.)
- **DEAP emotion**: Most models ~0.50 AUROC (random), 0.22-0.42 accuracy
- **Key quote**: *"current BFMs capture motor-related neural patterns more reliably than higher-level affective or communicative semantics"*
- **Quote**: *"Communication and affective computing remain challenging for cross-subject generalization due to strong non-stationarity and large cross-subject/inter-session variability"*
- **Finding**: Supervised baseline MSCARNet outperformed most BFMs on emotion tasks
- **Relevance**: Strongest independent confirmation that FM failure on affect/stress is universal, not specific to our dataset
- **Ref**: arXiv:2602.11558

### 2.4 EEG Foundation Models: Critical Review (2025)
- Most FMs use masked sequence reconstruction + transformers
- LP << FT gap raises questions about representation quality
- Evaluations are heterogeneous, making cross-paper comparison difficult
- **Ref**: arXiv:2507.11783

---

## 3. Subject Leakage / Evaluation Protocol

### 3.1 "Data Leakage in Deep Learning Studies of Translational EEG" (Frontiers Neuroscience, 2024)
- **Core finding**: Majority of published DNN-EEG studies use segment-based CV and dramatically overestimate real-world performance
- **Mechanism**: When segments from one subject appear in both train and test, models learn subject identity rather than pathology
- **Task**: Alzheimer's, epilepsy classification
- **Relevance**: DEFINITIVE reference for our trial-level vs subject-level argument
- **Ref**: Frontiers in Neuroscience, 2024, DOI: 10.3389/fnins.2024.1373515

### 3.2 "The Role of Data Partitioning on EEG-Based Deep Learning Models" (Computers in Biology and Medicine, 2025)
- **Method**: Trained 100,000+ models comparing sample-based vs subject-based CV
- **Finding**: Sample-based CV consistently inflates performance. Nested-LNSO provides most realistic estimates
- **Architectures tested**: ShallowConvNet, EEGNet, DeepConvNet, TResNet
- **Relevance**: Strongest empirical evidence to date for our evaluation methodology
- **Ref**: Computers in Biology and Medicine, 2025

---

## 4. Cross-Subject Generalization Methods

### 4.1 Domain Adaptation / Generalization
- **SCLDGN (IEEE TBME, 2025)**: Supervised contrastive learning + deep correlation alignment + domain-agnostic mixup for cross-subject motor decoding. Key: learning domain-invariant AND class-relevant features simultaneously
- **ADFR (BMC Bioinformatics, 2024)**: Adaptive deep feature representation learning for cross-subject EEG
- **Domain-Generalized DL for Emotion (PMC, 2025)**: Evaluated 12 approaches (4 DG techniques x 3 architectures). No single DG technique dominates; effectiveness depends on architecture
- **Cross-Subject Contrastive Learning (Scientific Reports, 2025)**: Region-aware contrastive learning for emotion recognition, focuses on brain areas implicated in emotional processing

### 4.2 LEAD: Subject-Regularized Training (Wang et al., 2025)
- **Task**: Alzheimer's detection on resting-state EEG
- **Architecture**: Gated Temporal-Spatial Transformer (3.3M params, smaller than LaBraM)
  - Parallel temporal attention (within-channel) + spatial attention (across-channel)
  - Learnable sigmoid gate to fuse: `E = G * E^T + (1-G) * E^S`
  - 3D channel embedding from 10-20 coordinates (similar to LaBraM)
  - Multi-sampling segmentation (200/100/50 Hz) as data augmentation
- **Subject-regularized training** (3 components):
  1. **Subject-level CE loss**: Average predictions across all segments from same subject, compute CE on the averaged prediction. Forces consistent per-subject predictions. Combined with sample-level CE: `L = α·L_sam + β·L_sub` (α=β=0.5)
  2. **Index group-shuffling**: Custom batching to guarantee ≥G samples per subject per batch (G=8 for FT), making subject-level loss meaningful
  3. **Multi-sampling segmentation**: Segment at multiple rates for multi-scale augmentation
- **Evaluation**: Monte Carlo CV with subject-independent splits (8:1:1), 5 seeds. NOT LOSO for main results
- **vs LaBraM**: LEAD (1,186 hrs pretraining) beats LaBraM (2,000 hrs) — attributed to domain-focused pretraining + subject regularization
- **Key numbers**: Subject-level F1 on ADFTD: LEAD 93.95% vs LaBraM 93.77% (close), but AUROC: LEAD 100% vs LaBraM 75% (big gap)
- **Handcrafted features**: Only as baseline (32 features → linear classifier, performs poorly)
- **Relevance**: Their subject-level CE loss is directly applicable to our problem. Our global pooling (averaging epoch features) achieves a similar structural effect. Their gated temporal-spatial attention is analogous to CBraMod's criss-cross but with learnable fusion instead of concatenation
- **Ref**: arXiv:2502.01678

### 4.3 EEG-GraphAdapter (2024)
- PEFT with GNN adapter on frozen FM backbone
- Only 2.4% trainable params, up to 16.1% F1 improvement
- Adds spatial graph representations to temporal-only backbones

---

## 5. Resting-State EEG Stress Classification

### 5.1 Reference Paper: "From Theory to Application" (Lin et al., 2025)
- **Dataset**: Same UCSD stress dataset (18 subjects, 82 recordings after 400s filter)
- **Method**: Fine-tuned LaBraM, 5s windows, trial-level CV
- **Result**: 81% balanced accuracy (our reproduction: 86.2% with same protocol)
- **Limitation**: Trial-level CV (subject leakage), not discussed in paper
- **Ref**: arXiv:2505.23042

### 5.2 Brain2Vec (2025)
- CNN-LSTM-Attention for EEG stress on DEAP dataset
- 81.25% accuracy but AUC only 0.68 (class imbalance issues)
- Likely trial-level evaluation
- **Ref**: arXiv:2506.11179

### 5.3 Spiking Neural Networks for Stress (Scientific Reports, 2025)
- Reports 98.75% accuracy with 10-fold CV
- Almost certainly inflated by subject leakage — illustrates the problem exactly
- **Ref**: Nature Scientific Reports, 2025

### 5.4 Review: Mental Stress by Deep Learning (Neural Computing & Applications, 2024)
- Comprehensive review of CNN/LSTM approaches for EEG stress
- Reported accuracies: 85–99%+, but most use within-subject or trial-level splits
- Notes lack of standardized evaluation as a field-wide gap
- **Ref**: Springer, 2024

---

## 6. Neurophysiological Stress Biomarkers & Hybrid Approaches

### 6.1 Theta/Alpha Ratio (TAR) for Stress (Scientific Reports, 2026)
- **Finding**: TAR outperforms absolute band power as single most discriminative EEG feature for cognitive stress
- **Frontal asymmetry** (Fp1-Fp2, F3-F4) also informative
- Feature fusion (EEG+ECG) surpasses single modality
- Cross-subject evaluation used (gender-stratified)
- **Relevance**: Directly supports our hybrid FM + TAR approach. Also validates the theta band finding from our Cross_Stress_Subject_Analysis.ipynb (d=0.8, though not significant at N=17)
- **Ref**: Nature Scientific Reports, 2026, DOI: 10.1038/s41598-026-38356-3

### 6.2 Multiband Dynamic Attention Network (2025)
- Multifrequency decomposition + frequency-domain attention + cross-band interaction
- Key insight: Standard DL fails to extract frequency band-specific features and misses inter-band interactions
- Explicitly modeling cross-band (theta-alpha) interactions improves stress classification
- **Relevance**: Supports the idea that neurophysiology-aware feature engineering adds value over pure end-to-end learning

### 6.3 Gap: FM + Handcrafted Feature Fusion
- **No paper found** that combines EEG-FM features (LaBraM/CBraMod) with classical band-power or connectivity features
- LEAD uses handcrafted features only as a weak baseline, not fused
- This is a potential novel contribution for our work

---

## 7. Public EEG Stress Datasets

### 7.1 Resting-State Stress Datasets

| Dataset | N | Protocol | EEG Setup | Labels | Public | Ref |
|---------|---|----------|-----------|--------|--------|-----|
| **UCSD Classroom** (ours) | 18 subj, 82 rec | 5-min eyes-open rest, longitudinal | 32ch, 1000Hz | DASS + DSS | No | arXiv:2505.23042 |
| **Al-Shargie 2020** | 33 | 3-min eyes-closed rest | 5ch Emotiv Insight, 128Hz | PSS-10 + expert interview | **No** | Sensors 2020, PMC7180785 |
| **EDPMSC** | 28 (13M/15F) | 3-min eyes-open rest, pre/post activity | 4ch Muse, 256Hz | PSS-10 (≥20 = stressed) | Yes | PMID:31283515 |

**Gap**: No public resting-state EEG dataset with DASS-style chronic stress labeling exists besides ours. Al-Shargie (closest, PSS + rest) is not public.

### 7.2 Task-Evoked Stress Datasets (Potential Auxiliary Data)

| Dataset | N | Protocol | EEG Setup | Labels | Public | Ref |
|---------|---|----------|-----------|--------|--------|-----|
| **SAM 40** | 40 (14F/26M, mean 21.5y) | Stroop, arithmetic, mirror image + relaxation, 25s epochs ×3 | 32ch Emotiv Flex | Task vs relaxation | Yes (CC BY 4.0) | Figshare, ScienceDirect |
| **EEGMAT** | 36 (university students, Ukraine) | 180s rest + 60s mental serial subtraction | 19ch 10-20, 500Hz | Rest vs arithmetic | Yes (PhysioNet, ODC) | PhysioNet |
| **STEW** | 48 | 2.5-min rest + 2.5-min SIMKAP multitask | 14ch Emotiv EPOC, 128Hz | Self-rated workload (1-9) | Yes (IEEE DataPort) | IEEE DataPort |
| **DASPS** (anxiety) | 23 (10M/13F, mean 30y) | Anxiety induction via exposure recall | 14ch Emotiv EPOC | Anxiety levels (psych assessment) | Yes (IEEE DataPort) | DOI:10.21227/barx-we60 |

**FM benchmark results on these datasets**:
- EEGMAT: CBraMod 71.94%, LaBraM 85.83% BA (subject-independent, EEG-FM-Bench/AdaBrain-Bench) — but task-evoked binary is fundamentally easier
- SAM 40: No FM evaluation found
- STEW: Limited FM results

**Potential use**: EEGMAT and SAM 40 could serve as auxiliary data for multi-dataset training or as independent validation that FM failure extends to task-evoked stress paradigms under subject-level CV.

### 7.3 Related Affect/Emotion Datasets (Not Stress)

- **SEED** (15 subj, 62ch): Film-clip emotion, 3/4 class. LaBraM 55.78% BA cross-subject (AdaBrain-Bench)
- **DEAP** (32 subj, 32ch): Music video emotion, valence/arousal. Most FMs ~0.50 AUROC (Brain4FMs)
- **Note**: WESAD (15 subj) has NO EEG — physiological only (ECG, EDA, EMG)

---

## 8. Neurophysiological Basis for Classification Ceiling

### 8.1 Neural Efficiency Hypothesis
- **Theory** (Haier 1988; Neubauer & Fink 2009): Higher intelligence/education → less neural activation for equivalent tasks
- **EEG evidence**: High cognitive-reserve individuals show lower spectral power in theta/delta, higher individual alpha peak frequency, greater coherence stability across conditions
- **Implication for our dataset**: Taiwanese graduate students = high-education, high-CR population. Predicts less neural differentiation between stress states — the very signal we're trying to classify is attenuated at the source
- **Ref**: Neubauer & Fink, *Neuroscience & Biobehavioral Reviews* 33(7), 2009; Fleck et al., *Frontiers in Aging Neuroscience* 12, 2020

### 8.2 Allostatic Regulation in Young Adults
- **Theory** (McEwen 2010): Chronic stress accumulates allostatic load; young adults have lower load — regulatory systems (HPA axis, ANS) return to baseline more effectively
- **Cortisol habituation**: Repeated/chronic stress → HPA axis downregulates response → neural signatures adapt to "new normal" indistinguishable from pre-stress baseline
- **Resilience = stability**: Lower resting-state brain network flexibility correlates with higher psychological resilience in young adults (PMC6989886) — stable patterns are the opposite of what stress classification needs
- **Ref**: McEwen, *Annals of the New York Academy of Sciences* 1186, 2010

### 8.3 Frontal Alpha Asymmetry: Trait vs State
- **Finding**: In young adults, alpha asymmetry reflects enduring dispositions (trait-like) rather than transient stress states
- **Implication**: Our strongest classical feature (right-hemisphere alpha) captures between-subject personality differences, not within-subject stress variation — explaining why it helps subject-level classification (~66% BA) but cannot push much further
- **Ref**: MDPI Symmetry 14(8), 2022

### 8.4 LOSO Accuracy Drop in Student Populations
- Studies achieving 93% with k-fold drop to **74% with LOSO** on same student EEG data
- Saeed et al. (2020): 85.2% on bachelor students (18-23y) with DASS + expert labeling, using standard k-fold — not LOSO
- Literature consensus: Chronic stress classification with LOSO consistently yields 55-75% BA
- **Ref**: Saeed et al., *Sensors* 20(7), 2020; Springer LNCS 2021 (LOSO necessity)

### 8.5 Our Hypothesis (Untested)
Young graduate students' brains may maintain stable resting-state EEG patterns even under chronic academic stress due to: (1) high cognitive reserve from years of education, (2) efficient allostatic regulation typical of young adults, (3) cortisol habituation from prolonged stress exposure. This would make the resting-state stress classification task inherently harder for this population than for older or less-educated cohorts, explaining the ~65% BA ceiling observed across all methods (FM, classical ML, adversarial training).

---

## 9. Summary: Our Positioning

| Aspect | Field Status | Our Contribution |
|--------|-------------|-----------------|
| FM on resting-state stress | Only 1 paper (our reference, 90% BA with leakage) | First multi-FM comparison (3 models) with rigorous subject-level CV |
| Subject-level evaluation | Known problem, rarely applied to FMs | Expose 30+ point inflation gap; quantify subject identity encoding (34-37x eta-squared) |
| FM + classical features | Not done | RF matches FM with 0 GPU; potential hybrid approach |
| Cross-subject stress with FMs | Unexplored | 3 FM benchmarks now confirm failure; we explain *why* via variance decomposition |
| Young-brain stress ceiling | Hypothesized in neuro lit, not tested on EEG classification | Converging evidence: neural efficiency + allostatic regulation explains ~65% ceiling |
| Public stress EEG benchmarks | No resting-state DASS dataset exists | Our dataset is unique; EEGMAT/SAM40 available for auxiliary validation |

### Key Papers to Cite
1. Data leakage paper (Frontiers 2024) — methodological backing
2. 100k models partitioning paper (2025) — empirical backing
3. EEG-FM-Bench (2025) — FM benchmark context
4. AdaBrain-Bench (2025) — FM benchmark context (emotion/stress failure)
5. Brain4FMs (2026) — strongest confirmation of FM failure on affect (15 models, near-random)
6. LEAD (2025) — subject-regularized training technique
7. TAR stress paper (2026) — neurophysiological feature justification
8. Reference paper (Lin et al., 2025) — dataset and baseline
9. Neubauer & Fink (2009) — neural efficiency hypothesis
10. McEwen (2010) — allostatic load framework
11. Saeed et al. (2020) — DASS-based EEG stress, closest comparison
