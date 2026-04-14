# Related Work (Living Document)

*Last updated: 2026-04-08. Major reframe after EEG-FM-Bench (Xiong et al. 2025) closest-prior pressure test: TDBRAIN fold-drift dilution is now the headline novelty; Stress and ADFTD are the framing endpoints. See §2.1 and §9.*

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

### 2.1 EEG-FM-Bench (Xiong et al., 2025) — *closest methodological prior*
- **Scope**: 7 EEG-FMs (BENDR, BIOT, LaBraM, EEGPT, CBraMod, CSBrain, REVE) + 2 time-series FMs, 14 datasets, 10 paradigms, 3 FT strategies (frozen / LoRA / full-parameter), 2 task setups (single- / multi-task), 3 classifier heads.
- **End-task findings**:
  1. Frozen backbone shows a *severe generalization gap*; full-parameter FT is the upper bound (§4.4, Fig. 9).
  2. Multi-task FT acts as a regularizer in data-scarce settings, often beating single-task FT.
  3. Compact, EEG-specific architectures (EEGPT, CBraMod) outperform much larger models — scaling laws break in low-SNR EEG.
- **Diagnostic findings (most relevant to our paper, §4.3 + Fig. 7)**:
  - **Gradient-mass analysis**: in pretrained-FT settings, gradient norms concentrate on the **Temporal Embedding** (input projection); Attention/MLP/Norm receive small gradients. They interpret this as *"pre-training stabilizes the Transformer backbone… fine-tuning primarily adapts the Temporal Embedding to bridge raw signals and latent space."* Framed as a feature, not a failure.
  - **CKA / RSA panels**: track *Scratch-vs-Pretrained* training trajectories — multi-task FT pulls scratch toward pretrained ("attractor"). They are *not* tracking how a pretrained representation evolves during FT on a single small clinical task.
- **What they explicitly do not do** (this is the gap our paper fills):
  1. No label-aware variance decomposition (CKA/RSA are unsupervised geometric measures).
  2. No per-dataset analysis — diagnostic results are multi-task averaged across 14 datasets, hiding any per-dataset failure modes (e.g. our ADFTD injection vs TDBRAIN erosion contrast).
  3. No per-fold representation tracking — fold-model drift on subject-saturated datasets is invisible to their pipeline because pooled OOF features would average it out.
  4. No claim that "FT does nothing on small data" — they argue full-parameter FT is the upper bound for accuracy.
- **How we position against EEG-FM-Bench**: their gradient-mass observation is the *input-side*, qualitative, multi-task-averaged version of our *output-side*, quantitative, dataset-stratified, label-aware measurement. The two are mechanistically consistent on Stress (small gradients on backbone → unchanged pooled label fraction 7.23%→7.24%), but only our metric makes the three-mode taxonomy visible — in particular, the TDBRAIN fold-drift dilution (3.0%→1.5%) cannot be seen with their CKA/RSA on multi-task averaged features.
- **Ref**: arXiv:2508.17742 (Feb 2026 v2)

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
- **Evaluation**: Subject-independent train/val/test 6:2:2, 5 seeds (41–45) on fixed splits. NOT LOSO and NOT k-fold.
- **vs LaBraM**: LEAD (1,186 hrs pretraining) beats LaBraM (2,000 hrs) — attributed to domain-focused pretraining + subject regularization
- **Verified key numbers (Table 2, single-dataset training, ADFTD, 53,215 samples / 65 subjects)**:
  - LaBraM sample-level: Acc 55.07, F1 71.03
  - LaBraM subject-level: **Acc 57.14, F1 72.73** ← anchor for our ADFTD comparison
  - (Earlier note in this doc claiming LaBraM F1 = 93.77% / AUROC 75% was incorrect — verified 2026-04-07 against LEAD arXiv v1 Table 2.)
- **Our LaBraM-FT on same dataset**: Subject Acc 0.7538, F1 0.7541, BA 0.7521 (5-fold StratifiedGroupKFold) — slightly **above** LEAD's reported vanilla LaBraM, confirming our FT recipe is properly trained, not undertrained.
- **TDBRAIN role in LEAD**: TDBRAIN (911 subjects) is used ONLY for self-supervised pre-training of LEAD, NOT as a downstream evaluation dataset. LEAD's 5 downstream datasets are ADFTD, BrainLat, CNBPM, Cognision-ERP, Cognision-rsEEG. **No published LaBraM-on-TDBRAIN-MDD-classification benchmark exists** in LEAD, EEG-FM-Bench (2508.17742), or AdaBrain-Bench (2507.09882, confirmed 2026-04-07).
- **Handcrafted features**: Only as baseline (32 features → linear classifier, performs poorly)
- **Relevance**: Their subject-level CE loss is directly applicable to our problem. Our global pooling (averaging epoch features) achieves a similar structural effect. Their gated temporal-spatial attention is analogous to CBraMod's criss-cross but with learnable fusion instead of concatenation
- **Ref**: arXiv:2502.01678

### 4.3 EEG-GraphAdapter (2024)
- PEFT with GNN adapter on frozen FM backbone
- Only 2.4% trainable params, up to 16.1% F1 improvement
- Adds spatial graph representations to temporal-only backbones

### 4.4 Kumar et al., "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution" (ICLR 2022 Oral)
- **Theoretical claim**: Full fine-tuning provably distorts pretrained features in the OOD direction relative to a linear-probe-then-finetune (LP-FT) two-step procedure. The mechanism: a randomly-initialized head pulls the backbone toward features that fit the head's noise during the first epochs of FT, before the head has converged.
- **Why it matters for our paper**: Provides the theoretical anchor for the *projection-vs-rewriting* distinction we measure with the pooled label fraction. The Stress regime (head learns a projection through an unchanged backbone) is the *opposite* failure mode from Kumar's distortion — the backbone *resists* the head's pull because gradients are too small (consistent with EEG-FM-Bench §4.3). The TDBRAIN regime (per-fold representations drift apart, diluting the global label signal) is closer to Kumar's distortion mechanism, manifested across CV folds rather than across in-distribution / OOD splits.
- **Practical implication**: LP-FT is the canonical *fix* to try on Stress as a sanity check that our diagnostic is actionable.
- **Ref**: arXiv:2202.10054

### 4.5 Huang, "Using Cluster Bootstrapping to Analyze Nested Data with a Few Clusters" (Educational and Psychological Measurement, 2018)
- **Methodological claim**: For nested data (observations within clusters), the standard percentile bootstrap underestimates SE because it treats correlated observations as independent. The fix is to resample *clusters* with replacement and include all observations from each resampled cluster.
- **Why it matters for our paper**: Justifies our cluster bootstrap (resampling subjects, not recordings) for the pooled label fraction confidence intervals. Without this, an n=195 ADFTD bootstrap would treat 195 recordings as independent when the effective n is ~65 subjects, producing anti-conservative CIs that a stats reviewer would catch immediately.
- **Ref**: Huang 2018, *Educational and Psychological Measurement* 78(2), 297–318

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

The headline novelty is the **three modes of FT behavior** on resting-state and task-evoked EEG — label-signal injection / mild injection / active erosion — measured with two complementary metrics:
1. **Representation-level**: pooled label fraction $SS_\text{label}/SS_\text{total}$ on the 200-d encoder output (paired matched-subsample resampling 100 draws/rung + subject-level label-permutation null).
2. **Behavioral-level**: multi-seed subject BA via Frozen LP (LogisticRegression on cached features) vs canonical-recipe FT.

The taxonomy across four datasets (updated 2026-04-10 after Stress reclassification):

| Mode | Dataset (n_rec / n_subj) | Label type | Representation Δ (frozen → FT) | Behavioral Δ (frozen LP → FT BA) | Representation behavior |
|---|---|---|---|---|---|
| **Injection** | ADFTD (195 / 65) | between-subject, strong EEG biomarker (theta/delta increase in AD) | 2.79% → 7.70% (**+5.22 to +5.68 pp N-invariant**) | 0.669 → 0.752 (**+8.3 pp**) | Clean label-signal injection at both metrics. The canonical success case. |
| **Mild injection** | EEGMAT (72 / 36) | within-subject, rest vs arithmetic (classical cognitive-load biomarker) | 5.35% → 5.82% (≈ 0, crossed design) | 0.671 → 0.736 (**+6.5 pp**) | Representation unchanged but BA improves — FT re-shapes head/projection without rewriting backbone. |
| **Silent erosion** | TDBRAIN (734 / 359) | between-subject, MDD (weak EEG biomarker, contested lit) | 2.97% → 1.47% (**−1.06 to −1.58 pp N-invariant**, opposite side of permutation null) | 0.679 → 0.681 (≈ 0) | Representation label-fraction drops 50% while BA is unchanged — classifier compensates in a linear projection the frozen representation already supplied. |
| **Behavioral erosion** | Stress (70 / 17) | between-subject, DASS trait class (no accepted EEG biomarker) | **stale** — needs re-run under `--label dass` | 0.605 ± 0.030 → 0.443 ± 0.068 (**−16.2 pp**) | Frozen LP beats FT by 16 pp. Real FT is statistically indistinguishable from null (FT on shuffled labels: 0.497 ± 0.081, p(null ≥ real)=0.70). Full evidence: `results/studies/exp03_stress_erosion/`. |

**The headline correction (2026-04-08 evening, after the matched-subsample experiment):** the previous "dataset-size-dependent" framing of this taxonomy was wrong. At literally matched N=17 with nearly identical frozen baselines (ADFTD 6.50%, Stress 7.23%), ADFTD gains +5 pp from FT and Stress gains 0 pp. The three modes are properties of the *label biology*, not the training set size. The canonical "ADFTD ×2.76 rewrite" ratio is also N-inflated (the pooled-fraction denominator shrinks at small N) — at Stress-comparable N=17 the true ratio is 1.87×, and the **N-invariant additive +5 pp** is the honest effect size. Use the additive framing as the headline; the ratio framing should appear only at full N for literature comparison.

The TDBRAIN active-erosion finding is then the strongest leg of the stool: under permutation null, TDBRAIN's observed Δ doesn't merely fail to be positive — it sits on the *opposite* side of zero from where the null sits. FT is doing real damage to a label signal the frozen FM already carried.

Reviewer-prior coverage: EEG-FM-Bench's CKA/RSA pipeline averages over multi-task and over folds, so neither the projection-only behavior on Stress + EEGMAT nor the active erosion on TDBRAIN is visible to it. Their gradient-mass observation ("backbone remains stable on pretrained-FT") is mechanistically consistent with our no-op result on Stress and EEGMAT but never quantified, label-aware, or stratified per-dataset, and they cannot see the ADFTD injection or the TDBRAIN erosion at all because their CKA pipeline is unsupervised and dataset-averaged.

| Aspect | Field Status | Our Contribution |
|--------|-------------|-----------------|
| **FT representation evolution diagnostics** | EEG-FM-Bench tracks gradient mass (input-side) and CKA/RSA (unsupervised, multi-task averaged) | Pooled label fraction $SS_\text{label}/SS_\text{total}$ — output-side, label-aware, dataset-stratified, per-fold-aware. Sensitive to fold-drift dilution that CKA/RSA average out. |
| **N-invariance of FT effect direction** | Field default: assume more data → cleaner adaptation (monotonic). | Three N-invariant modes: injection (ADFTD +5 pp stable across N), no-op (Stress/EEGMAT Δ ≈ 0), erosion (TDBRAIN −1.5 pp stable across N, on opposite side of permutation null). Verified by paired matched subsampling and label-permutation null. Inverts the scaling-laws expectation: TDBRAIN gets *worse* with FT regardless of how much data you give it. |
| **FM on resting-state stress** | Only 1 paper (our reference, 90% BA with leakage) | First multi-FM comparison with rigorous subject-level CV (Stress is one of three case studies, not the headline) |
| **Subject-level CV rigor for FMs** | Known to matter (Brookshire 2024); rarely applied to FMs | 20+ point inflation gap quantified across 3 FMs; subject-level cluster bootstrap and subject-level PERMANOVA used as the inference layer |
| **Statistical rigor of representation analysis** | Naive percentile bootstrap on recording-level features is the field default | Cluster bootstrap over subjects (Huang 2018) + per-fold degeneracy guard for n_subjects_per_class < 2 |
| **FM + classical features on resting-state clinical EEG** | Not directly compared | RF on hand-crafted band power matches FT LaBraM on ADFTD (0.753 vs 0.752); Stress comparison is **stale** under subject-dass (RF 0.666, FT 0.656) — both need re-run under per-rec DASS. |
| **Young-brain stress ceiling explanation** | Hypothesized in neuro lit, not tested on EEG classification | Converging evidence: neural efficiency + allostatic regulation explains the ~65% Stress ceiling at the *signal* level, complementing the *representation* level diagnosis |

### Key Papers to Cite (in order of centrality to the new framing)

**Closest priors / what we measure differently than:**
1. **Xiong et al. 2025 — EEG-FM-Bench (arXiv:2508.17742)** — closest methodological prior. Their gradient-mass and CKA/RSA analyses are the input-side, unsupervised, multi-task-averaged counterparts to our output-side, label-aware, dataset-stratified pooled label fraction. We position against §4.3 Fig. 7 explicitly.
2. **Kumar et al. 2022 — LP-FT (arXiv:2202.10054)** — theoretical anchor for the projection-vs-rewriting distinction. The Stress regime is the opposite failure mode from Kumar's distortion (backbone resists head pull); the TDBRAIN regime is closer to Kumar's distortion across folds.

**Statistical methodology:**
3. **Brookshire et al. 2024 — Data leakage in translational EEG (Frontiers Neuroscience)** — the definitive case for subject-level holdout and subject-level resampling.
4. **Huang 2018 — Cluster bootstrap for nested data (Educational and Psychological Measurement)** — justifies cluster bootstrap over subjects in our CIs.
5. **Anderson 2001 — PERMANOVA** — non-parametric robustness check on the parametric mixed-effects results.
6. **100k models partitioning paper (2025)** — empirical backing for subject-level CV necessity.

**Field context (FM benchmarks confirming failure on affect/clinical):**
7. **AdaBrain-Bench (arXiv:2507.09882)** — emotion/stress failure across FMs.
8. **Brain4FMs (arXiv:2602.11558)** — 15-model confirmation that FM failure on affect is universal.
9. **Critical Review of EEG FMs (arXiv:2507.11783)** — frames the LP << FT gap as a representation-quality concern.

**Datasets and reference baselines:**
10. **Lin et al. 2025 (arXiv:2505.23042)** — UCSD stress dataset and the trial-level baseline we replicate then re-evaluate at subject level.
11. **LEAD (arXiv:2502.01678)** — ADFTD baseline LaBraM number we anchor against (Table 2).
12. **Wang et al. 2025 (LaBraM, arXiv:2405.18765)** — backbone architecture.

**Neurophysiological ceiling (signal-level explanation, complement to representation-level diagnosis):**
13. Neubauer & Fink 2009 — neural efficiency hypothesis.
14. McEwen 2010 — allostatic load framework.
15. Saeed et al. 2020 — DASS-based EEG stress, closest comparison.
