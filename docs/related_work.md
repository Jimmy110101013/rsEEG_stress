# Related Work (Living Document)

*Last updated: 2026-04-19.*

**Current defensibility status after HP audit (2026-04-19)** — see `findings.md` top banner for full inventory.
- **HP-safe** (pipeline independent of FT runs): F-A (variance decomposition, 18/18 subject > label), F-NEURO (spectral anchors on frozen reps), F-HHSA (raw EEG), F-E (classical RF), Frozen LP numbers.
- **Pending sanity reproduction** (all depend on HP-contaminated exp_newdata FT): F-C (FT direction flip), F-D.2-4 (Stress FT, architecture ceiling, permutation null), ρ(subject_id_BA, ΔBA) correlation, mean ΔBA between/within.
- **Crowded niches** (no longer novel — literature scooped): "FT underperforms under subject-level CV" (ICLR 2026 *Are EEG FMs Worth It?*, Dingkun *EEG-FM-Benchmark*, Xiong *EEG-FM-Bench*); reconstruction-objective critique (Laya, STELAR, EEGDM, NeuroRVQ); subject identity as strong FM signal (*Subject-Aware Contrastive*, TS4H NeurIPS 2025 — same empirical fact, opposite framing).

**Open niche** (post-sanity-reproduction): quantitative mechanism linking subject-axis strength to FT behavior (requires HP-clean FT rerun to verify).

**Framework precedents**: EEG-Bench (Scherer, arXiv:2512.08959, NeurIPS 2025 — §2.5), NeurIPS 2025 EEG Foundation Challenge (§10), bioRxiv 2025.11 sample-size psychiatry (§2A), Massive-SFT / Robustness-Tradeoff (§4.6–§4.7).

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

### 2.5 EEG-Bench (Scherer et al., NeurIPS 2025)
- **Scope**: 14 datasets × 11 clinical tasks (epilepsy, schizophrenia, Parkinson, OCD, TBI) under strict cross-subject evaluation.
- **Headline finding**: *"Foundation models achieve strong performance in certain settings, but simpler models often remain competitive, particularly under clinical distribution shifts."*
- **Relevance**: Fourth independent clinical-EEG benchmark (alongside EEG-FM-Bench, AdaBrain-Bench, Brain4FMs) converging on "FMs are not universally superior on clinical cross-subject tasks." Does not preempt F-A (variance decomposition), F-C (direction-flipping FT), or F-D (Stress power-floor); strengthens F-B.
- **Ref**: arXiv:2512.08959

### 2.7 "Are EEG Foundation Models Worth It?" (ICLR 2026, OpenReview)
- **Scope**: 6-axis evaluation framework (accuracy, robustness, scaling, transfer, efficiency, task diversity); benchmarks multiple open-source FMs against compact NN decoders and classical (non-neural) baselines under strict LOSO zero-shot on diverse BCI tasks. Introduces own ST-EEGFormer (ViT + MAE, 8M segments) as control.
- **Headline finding**: *"Linear-probed foundation models generally underperform across all protocols."* FT-FMs fail to beat compact NN or classical decoders on data-scarce BCI; no scaling law observed across neural decoders.
- **Relevance**: Direct prior art for our "FT underperforms under subject-level CV" observation. They provide the *failure-surface* documentation; they do NOT attribute it to subject-identity dominance or quantify ρ(subject-decodability, ΔBA). Our mechanism claim remains distinct.
- **Ref**: OpenReview forum id 5Xwm8e6vbh.

### 2.8 EEG-FM-Benchmark (Dingkun et al., 2026)
- **Scope**: 12 open-source FMs × 13 datasets × 9 paradigms, cross-subject LOSO vs within-subject few-shot comparison, full-FT vs head-only FT vs trained-from-scratch compact baselines.
- **Status**: Diagnostic only; no proposed mechanism or solution. Distinct from EEG-FM-Bench arXiv:2508.17742 (Xiong et al., §2.1) — uses the same protocol scaffolding but a different dataset / model mix.
- **Relevance**: Second independent benchmark confirming FT ≯ compact baselines under LOSO. Further tightens the "failure surface is already published" reality check.
- **Ref**: GitHub `Dingkun0817/EEG-FM-Benchmark` (associated paper in preparation as of April 2026).

### 2.9 Recent EEG FM Reviews / Surveys
- **Kwon et al. 2026** — *Eur J Neurosci*, DOI 10.1111/ejn.70376. Review of EEG FMs toward unified representations; useful field-survey cite.
- **"Foundation Models for EEG Decoding"** (PubMed 41145005) — recent review synthesis.
- **"Inter- and Intra-Subject Variability in EEG: A Systematic Survey"** (arXiv:2602.01019, Feb 2026) — systematic review framing for the inter-subject variability problem our F-A quantifies.

---

## 2A. Statistical Power & Small-N Reliability

### 2A.1 Sample-Size Reliability in Psychiatric EEG (bioRxiv 2025)
- **Title**: "Sample Size Critically Shapes the Reliability of EEG Case-Control Findings in Psychiatry" (bioRxiv 2025.11.10.687610, PMID 41292863, posted 2025-11-12).
- **Scope**: Multisite N = 2,874 resting-state EEG across ADHD/ASD/anxiety/learning disorders.
- **Key quote**: *"Results from small samples were unstable, with inflated and highly variable effect sizes across iterations. Larger samples produced consistent findings, converging on uniformly small but robust effects."*
- **Relevance**: Direct citation for our §8 power-floor framing — supports F-D.1 (FT null-indistinguishable on N=70 / 14-positive) and F-B G-F08 (cuDNN ±5–10 pp swings on small Stress cohort).

### 2A.2 Robust EEG Brain-Behavior Associations Require Large N (bioRxiv 2026)
- **Title**: "Robust EEG Brain-Behavior Associations Emerge Only at Large Sample Sizes" (bioRxiv 2026.02.06.704323).
- **One-liner**: Same large-N-required conclusion from the brain-behavior correlation angle; independent precedent for the §8 power-floor argument.

---

## 3. Subject Leakage / Evaluation Protocol

### 3.1 "Data Leakage in Deep Learning Studies of Translational EEG" (Frontiers Neuroscience, 2024)
- **Core finding**: Majority of published DNN-EEG studies use segment-based CV and dramatically overestimate real-world performance
- **Mechanism**: When segments from one subject appear in both train and test, models learn subject identity rather than pathology
- **Task**: Alzheimer's, epilepsy classification
- **Relevance**: DEFINITIVE reference for our trial-level vs subject-level argument
- **Ref**: Frontiers in Neuroscience, 2024, DOI: 10.3389/fnins.2024.1373515

### 3.1a "Impact of Trial-wise and Test Data Leakage on EEG-Based Emotion Classification" (CEUR-WS Vol-4115 paper7, 2024)
- **One-liner**: Quantifies inflation from trial-wise leakage specifically in affect classification.
- **Relevance**: Emotion-specific precedent for our Wang 2025 subject-level reanalysis — strengthens the §3 leakage case beyond Brookshire's clinical framing.
- **Ref**: CEUR-WS Vol-4115 paper 7, 2024.

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
- **Task**: 3-class HC vs AD vs FTD, 88 subjects / 167,083 samples. ADFTD is never reported as binary in this paper (binary AD/HC appears only for ADFSU, ADSZ, APAVA).
- **Evaluation**: Subject-independent 8:1:1 train/val/test (Monte Carlo CV), 5 seeds (41–45). NOT LOSO, NOT k-fold.
- **FT hyperparameters** (§3.1, unified across all 16 baselines including LaBraM/CBraMod): AdamW + CosineAnnealingLR, **lr=1e-4** (pre-training uses 2e-4), batch=512, up to 200 epochs with patience=15 on best sample-level F1.
- **LaBraM checkpoint**: `labram-base.pth` from `935963004/LaBraM`. ~5.82M params. Paper adds Conv1D channel-mapping layer; other architecture default.
- **LaBraM on ADFTD — reproduction targets (v4 Table 2 + Appendix G.1 Table 8)**:

  | Level | Acc | F1 | AUROC | AUPRC |
  |---|---|---|---|---|
  | Sample-level | 77.00±3.76 | 75.64±4.68 | 91.22±2.72 | 84.75±4.41 |
  | Subject-level | 92.00±7.48 | 91.14±8.64 | 93.77±6.16 | 88.78±10.19 |

- **CBraMod on ADFTD (v4 Table 2)**: sample F1 68.33±4.53, AUROC 86.95±2.89; subject F1 82.21±6.30, AUROC 87.10±3.77.
- **LEAD own result**: sample F1 81.01±5.02 / AUROC 94.03±1.33; subject F1 93.95±4.95 / AUROC 95.36±3.85.
- **vs LaBraM**: LEAD (1,186 hrs pretraining) beats LaBraM (2,000 hrs) — attributed to domain-focused pretraining + subject regularization.
- **TDBRAIN role**: used ONLY for LEAD self-supervised pre-training, NOT as downstream eval. LEAD's 5 downstream datasets are ADFTD, BrainLat, CNBPM, Cognision-ERP, Cognision-rsEEG. No published LaBraM-on-TDBRAIN-MDD benchmark in LEAD, EEG-FM-Bench, or AdaBrain-Bench.
- **Relevance to us**: LEAD's subject-level CE loss and majority-vote aggregation define the community protocol. Their ADFTD numbers are our sanity reproduction target (`results/studies/sanity_lead_adftd/`, run 2026-04-19+).
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

### 4.4a Multi-dataset Joint Pre-training of Emotional EEG (arXiv:2510.22197, NeurIPS 2025)
- **Group**: NCClab-SUSTech.
- **Method**: Cross-dataset covariance alignment loss for affective EEG pretraining.
- **Result**: +4.57% AUROC few-shot and +11.92% zero-shot over SOTA large EEG FMs on emotion tasks.
- **Relevance**: Acknowledges inter-subject variability as the dominant FM failure mode on affect and proposes an alignment-based rescue; complementary to our diagnostic framing. Our paper should note this as an active research direction our F-D Stress-ceiling analysis does not rule out.
- **Ref**: arXiv:2510.22197.

### 4.4b Massive Supervised Fine-tuning Experiments (arXiv:2506.14681)
- **Scope**: 757 fine-tuned models × 10 base architectures × 10 datasets (LLM domain).
- **Key quote**: *"The inductive biases of the base model outweigh the specific SFT corpus in determining the final representation."*
- **Relevance**: LLM cross-modality precedent for F-C (FT outcome is governed by the pretrained architecture more than by the downstream data). Independently supports our model × dataset interaction finding rather than scooping it.
- **Ref**: arXiv:2506.14681.

### 4.4c Li et al., "On the Robustness Tradeoff in Fine-Tuning" (ICCV 2025)
- **One-liner**: Full fine-tuning systematically trades robustness against accuracy in vision models.
- **Relevance**: Vision-domain analogue of our TDBRAIN active-erosion finding — FT can damage pretrained features even when downstream accuracy looks flat. Supports the generality of the erosion mode we document.

### 4.6 Subject-Aware Contrastive Learning for EEG FMs (TS4H workshop, NeurIPS 2025)
- **Scope**: Patch-based LBM-style transformer pretrained with **intra-subject / inter-session contrastive positives** (no augmentation). Same-subject patches are forced to cluster in embedding space. Compared to LaBraM under LP and full FT; evaluated via alignment, uniformity, smooth-effective-rank, plus downstream task BA.
- **Framing (opposite of ours)**: **"Subject identity is signal, not confound."** They treat intra-subject regularity as the natural supervision signal for SSL and design pretraining to amplify it.
- **Why critique is still defensible despite their embrace**: their application target is within-subject BCI (calibrate-then-deploy). For **cross-subject clinical screening** (our ADFTD / MDD / Stress target population — test subjects are new), subject-conditioned representations are the wrong inductive bias. Our Type-C claim is specifically about **cross-subject clinical deployment**, where subject-axis dominance becomes leakage, not signal.
- **How to cite them**: corroborating evidence for the empirical fact (FM features are highly subject-decodable), paired with a principled objection on application domain. They validate our diagnostic; they do not pre-empt our prescription.
- **Ref**: OpenReview forum id MdgBATPjEu (TS4H workshop, NeurIPS 2025, Sep 23 2025).

### 4.5 Huang, "Using Cluster Bootstrapping to Analyze Nested Data with a Few Clusters" (Educational and Psychological Measurement, 2018)
- **Methodological claim**: For nested data (observations within clusters), the standard percentile bootstrap underestimates SE because it treats correlated observations as independent. The fix is to resample *clusters* with replacement and include all observations from each resampled cluster.
- **Why it matters for our paper**: Justifies our cluster bootstrap (resampling subjects, not recordings) for the pooled label fraction confidence intervals. Without this, an n=195 ADFTD bootstrap would treat 195 recordings as independent when the effective n is ~65 subjects, producing anti-conservative CIs that a stats reviewer would catch immediately.
- **Ref**: Huang 2018, *Educational and Psychological Measurement* 78(2), 297–318

---

## 5. Resting-State EEG Stress Classification

### 5.1 Reference Paper: "From Theory to Application" (Wang et al., 2025)
- **Dataset**: Komarov 2020 Stress dataset (resting-state EEG, graduate students, longitudinal). Their report uses 18 subjects.
- **Method**: Fine-tuned LaBraM, 5s windows, trial-level CV
- **Result**: 90.47% balanced accuracy. Senior author (Jung) is shared with the dataset paper (Komarov et al. 2020) — same lab.
- **Limitation**: Trial-level CV (subject leakage), not discussed in paper. This is the inflated baseline we re-evaluate under subject-level CV.
- **Ref**: Wang, Zhang, Chen, Truong, Jung. arXiv:2505.23042 (2025-05-29).

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

### 5.5 DeepAttNet / Ear-EEG Stress (Frontiers Human Neuroscience, 2025)
- **Scope**: Subject-independent stress decoding on ear-EEG with a cross-attention architecture.
- **Relevance**: Current competitor to our stress pipeline on a different (ear-EEG) modality, under subject-independent evaluation. Useful comparator for §5 positioning.
- **Ref**: Frontiers in Human Neuroscience, 2025, DOI: 10.3389/fnhum.2025.1685087.

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

> **⚠️ Superseded by Type C critique pivot (2026-04-18) + HP audit (2026-04-19).** The previous §9 content built a "three modes of FT behavior" taxonomy (ADFTD injection / TDBRAIN erosion / EEGMAT projection / Stress no-op) with hardcoded Δ label-fraction numbers (2.79→7.70%, 2.97→1.47%, 5.35→5.82%, etc.). **All of those numbers came from exp_newdata FT runs that are HP-contaminated** — see audit banner at the top of `findings.md`. Do not cite Δ label-fraction direction flips from this section until reproduction under per-FM official HP. Historical content trimmed 2026-04-19; retrieve from git if reviewer asks for the pre-audit framing.

### What remains defensible regardless of HP audit outcome

1. **Frozen representations are subject-dominated across 6 datasets × 3 FMs** (F-A; RSA 12/12, variance decomposition) — this does not depend on any FT.
2. **Wang 90.47% Stress BA inflates under subject-level CV** — Frozen LP 0.605 and classical XGBoost 0.553 are our reproductions; these do not depend on FT HP.
3. **Our evaluation is stricter than community** — subject-level 5-fold CV (vs most papers' predefined subject split × seeds) and recording-level pooled classification at LP (vs community per-window LP). Both direction flags ourselves as conservative, not optimistic.
4. **Positioning vs EEG-FM-Bench**: their CKA/RSA is input-side, multi-task averaged, unsupervised; ours is output-side, label-aware, per-dataset (F-A). This contribution holds even if FT numbers change.

### What depends on HP-clean FT reruns (currently stale)

- Any FT-vs-Frozen ΔBA statement across our 6 datasets (formerly the paper's Type A / Type C headline evidence).
- The three-modes FT direction taxonomy (injection / projection / erosion).
- ρ(subject_id_BA, ΔBA) correlation — both the +0.50 between-arm and underpowered within-arm claims.
- "Architecture-independent Stress ceiling" (F-D.3) at 0.52–0.58 across FMs.

See `findings.md` top-of-file audit for the full HP-contaminated vs HP-safe claim inventory.

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
10. **Komarov, Ko, Jung 2020 (IEEE TNSRE 28(4):795)** — Stress dataset paper. Longitudinal resting-state EEG + DASS-21 + DSS on Taiwan graduate students.
11. **Wang, Zhang, Chen, Truong, Jung 2025 (arXiv:2505.23042)** — LaBraM FT on Komarov's Stress dataset at trial-level CV, reports 90.47% BA. This is the inflated baseline we re-evaluate at subject level.
12. **LEAD (arXiv:2502.01678)** — ADFTD baseline LaBraM number we anchor against (Table 2).
13. **Jiang et al. 2024 (LaBraM, ICLR; arXiv:2405.18765)** — backbone architecture.

**Neurophysiological ceiling (signal-level explanation, complement to representation-level diagnosis):**
13. Neubauer & Fink 2009 — neural efficiency hypothesis.
14. McEwen 2010 — allostatic load framework.
15. Saeed et al. 2020 — DASS-based EEG stress, closest comparison.

---

## 10. Field-level Framing: Cross-Subject Clinical Decoding as the Open Problem

### 10.1 EEG Foundation Challenge 2025 (Truong et al., NeurIPS 2025 competition)
- **Title**: "EEG Foundation Challenge: From Cross-Task to Cross-Subject EEG Decoding."
- **Scope**: NeurIPS 2025 competition track explicitly framed around cross-subject generalization and psychopathology prediction (CBCL p-factor, internalizing, externalizing, attention).
- **Relevance**: Strongest field-level marker that cross-subject + clinical psychopathology decoding is *the* open problem for EEG FMs. Provides external validation of the F-A (subject dominance) and F-D (Stress power-floor) framings our paper advances.
- **Ref**: arXiv:2506.19141.

---

## 11. Pretraining-Objective Redesign (Alternatives to Masked Reconstruction)

### 11.1 Laya — LeJEPA for EEG (arXiv, 2026)
- **Diagnostic claim**: Reconstruction-based SSL biases representations toward **high-variance artifacts** rather than task-relevant neural structure.
- **Method**: JEPA (Joint-Embedding Predictive Architecture) — abandons pixel/waveform reconstruction; predicts future features in latent space.
- **Result**: Beats reconstruction baselines under LP at 10% pretraining data on EEG-Bench clinical tasks.
- **Overlap with us**: *partial framing alignment* — "high-variance axis is a distractor" is adjacent to our "subject identity is the dominant variance axis". Laya does not name subject identity or quantify subject-level CV diagnostics; they are prescriptive, we are diagnostic.
- **Ref**: arXiv (verify ID before citing — agent reported 2603.16281 which requires double-check).

### 11.2 STELAR — Dual-Space Pretraining (OpenReview 2025/2026)
- **Method**: Three-part loss — visible-token alignment (representation space), masked-token alignment, linear masked reconstruction (signal space). Encoder-centric with lightweight auxiliary heads.
- **Result**: +5% LP over EEGPT-like baselines, ~50% fewer pretraining parameters.
- **Overlap with us**: none on subject-identity dominance. Pure objective engineering.
- **Ref**: OpenReview forum id WzVHEkp4cK.

### 11.3 NeuroRVQ — Multi-Scale Tokenization (arXiv:2510.13068)
- **Diagnostic claim**: Neural tokenizers fail to preserve high-frequency dynamics, limiting generative reconstruction fidelity; single-scale tokenization blurs oscillatory band structure.
- **Method**: Multi-scale feature extractor + hierarchical RVQ codebooks + phase/amplitude-aware loss.
- **Result**: +15% vs other large brain models on 5 BCI tasks.
- **Overlap with us**: none. Token-engineering paper.
- **Ref**: arXiv:2510.13068.

### 11.4 EEGDM — Diffusion-Based EEG SSL (arXiv:2508.14086, Aug 2025)
- **Method**: DDPM over state-space model backbone; latent-fusion transformer head. Abandons MAE entirely.
- **Datasets**: TUEV (interictal discharges) + CHB-MIT (seizures).
- **Overlap with us**: none. Clinical-event detection with a novel SSL objective; no subject-level CV diagnostic.
- **Ref**: arXiv:2508.14086.

### 11.5 Crowding Summary
The "replace reconstruction with something else" move is **crowded as of April 2026**. Laya/STELAR/NeuroRVQ/EEGDM/STELAR collectively saturate the prescription space. Our paper should **not** make a prescriptive objective-redesign claim; we stay diagnostic and cite these as independent motivations for why subject-axis leakage matters upstream.
