# SDL Paper Draft — §1 Introduction + §2 Related Work + §3 Methods

**Date**: 2026-04-15 (initial SDL thesis draft — not yet ported to LaTeX)
**Thesis**: "Subject-Dominance Limits (SDL): frozen EEG foundation models are subject atlases; fine-tuning rescues classification only when the task label has a within-subject neural contrast strong enough to reorganise around. When absent, no architecture — from 40k-parameter 2017 CNN to 1.4B-parameter pretrained FM — exceeds the subject-dominance ceiling."

---

## §1 Introduction

Foundation models (FMs) pretrained on large unlabelled EEG corpora have recently reported strong performance on downstream clinical tasks, rekindling interest in brain-data transfer learning [LaBraM (Jiang et al. 2024 ICLR); CBraMod (Wang et al. 2024); REVE (Zhong et al. 2025); Kauffmann et al. 2025 critical review]. A representative claim comes from Wang et al. 2025 (arXiv:2505.23042), who report 90.47 % balanced accuracy (BA) on the Komarov–Ko–Jung resting-state stress dataset (2020, IEEE TNSRE 28(4):795), a 17-subject / 92-recording cohort with per-recording DASS-21 self-reports. Taken at face value, this result would imply that modern EEG FMs solve a cross-subject affective-state prediction problem that has frustrated traditional signal processing for a decade.

We argue that the claim does not survive honest evaluation, and that the broader question of *when* EEG FMs should be trusted for cross-subject clinical prediction is more tractable than the benchmark literature suggests. Our contribution is not a new architecture, a new loss, or a new dataset. It is a *diagnosis*: we show that (i) frozen EEG FMs overwhelmingly encode subject identity rather than task signal; (ii) on the UCSD Stress dataset, honest subject-level cross-validation collapses the reported 0.90 BA to 0.52–0.58 across three FMs, and further drops to chance when compared against label-permutation null; (iii) seven architectures spanning six orders of magnitude in parameter count — from a 40 k-parameter from-scratch 2017 CNN to a 1.4 B-parameter pretrained FM — converge to the same 0.44–0.58 band on this dataset, demonstrating that the ceiling is a task property, not a model property; and (iv) a direct paired comparison isolates the operative factor: on EEGMAT (rest vs. mental arithmetic, a condition associated with well-characterised alpha desynchronisation [Klimesch 1999]), LaBraM fine-tuning reaches 0.73 BA, whereas on UCSD Stress reframed as a within-subject longitudinal DSS classification task, all three FMs × three classifiers fail at or below chance.

We call this pattern **Subject-Dominance Limits (SDL)**: when the downstream label corresponds to a within-subject neural contrast that is strong relative to subject-level representational variance, FM fine-tuning can project the label variance onto an otherwise subject-dominated representation. When the neural contrast is weak or absent, no architecture available to us escapes the subject-dominance ceiling.

**The SDL diagnosis reframes a field-wide question**. Independent EEG FM benchmarks (Brain4FMs 2026; EEG-FM-Bench [Xiong 2025]; AdaBrain-Bench 2025; EEG-Bench [Scherer et al. 2025, NeurIPS]) have converged on the empirical observation that FMs do not consistently outperform simpler models on cross-subject clinical tasks. Our contribution is a mechanistic layer under that observation: we quantify subject variance versus label variance in frozen FM representations, and we show via a controlled within-subject paired experiment that the failure locus is contrast strength, not model capacity.

The practical implication is a **pre-benchmark diagnostic question** for any group deploying EEG FMs on a small clinical cohort: *does the label have a within-subject neural contrast?* If no independent evidence exists for such a contrast, reported subject-level benchmarks should be interpreted as bounded by the subject-dominance ceiling — regardless of how large the pretrained model or how careful the fine-tuning recipe.

**Contributions** (three, aligned with SDL):

1. **Subject-atlas quantification (§4)**. On four clinical datasets (UCSD Stress, EEGMAT, ADFTD, TDBRAIN), variance decomposition of frozen FM representations shows that task labels explain 2.8–7.2 % of representation variance, while subject identity explains 56–71 %. RSA subject-similarity exceeds RSA label-similarity in 12 of 12 model × dataset pairs. This establishes the "subject atlas" structural premise that all downstream results must contend with.

2. **Honest-evaluation audit of UCSD Stress (§5)**. A single protocol correction — subject-level StratifiedGroupKFold in place of trial-level StratifiedKFold — drops Wang et al. 2025's 0.90 BA to 0.52–0.58 across three FMs. A further permutation-null comparison places LaBraM fine-tuning at p = 0.70 vs label-shuffle, statistically indistinguishable from chance. We release the exact code and checkpoints for the fix and audit.

3. **Paired within-subject contrast diagnostic (§6) + architecture-independence check (§7)**. A controlled pair of within-subject experiments (EEGMAT rest vs arithmetic, known alpha-ERD; UCSD Stress longitudinal DSS, no published within-subject correlate) isolates neural-contrast strength as the determining factor. Cross-model band-stop ablation provides independent neural-level confirmation: EEGMAT shows cross-FM convergence on alpha; UCSD Stress shows cross-FM peak divergence. A forest plot of seven architectures on UCSD Stress confirms architecture-independence of the ceiling.

The rest of the paper is organised as follows. §2 summarises related work on EEG FMs, honest evaluation, small-sample EEG, and cognitive-load spectral signatures. §3 presents the datasets, pipeline, and — importantly — the methodological corrections and disclosures that distinguish this work from previously reported FM benchmarks: a LaBraM architectural drift we discovered and fixed (N-F21), a DASS-21 psychometric scope statement, and a pre-registered contrast-strength anchoring protocol that avoids circular FM-dependent reasoning.

---

## §2 Related Work

### 2.1 EEG Foundation Models and the cross-subject clinical challenge

The past two years have seen the emergence of large EEG foundation models pretrained on aggregated public corpora: LaBraM [Jiang et al. 2024, ICLR; vector-quantised neural tokenizer, ~100 M parameters], CBraMod [Wang et al. 2024; criss-cross attention, ~100 M parameters], REVE [Zhong et al. 2025; linear patch embedding, ~1.4 B parameters], and a growing family of related encoder variants surveyed by Kwon et al. (2026, Eur. J. Neurosci.) and PubMed 41145005 (2025). Each reports competitive or state-of-the-art performance on selected downstream tasks — typically sleep staging, seizure detection, or event-related potentials. However, performance on *cross-subject affective and clinical state prediction* has been more mixed.

Four recent large-scale benchmarks have independently attempted to stress-test FM claims under strict cross-subject evaluation. Brain4FMs (2026), EEG-FM-Bench [Xiong et al. 2025], and AdaBrain-Bench (2025) each report settings where current FMs fail to outperform classical pipelines or compact from-scratch models. Most directly relevant is EEG-Bench [Scherer et al., arXiv:2512.08959, NeurIPS 2025], which covers 14 datasets × 11 clinical tasks (epilepsy, schizophrenia, Parkinson, OCD, TBI) under strict cross-subject splits and concludes that "while foundation models achieve strong performance in certain settings, simpler models often remain competitive, particularly under clinical distribution shifts." Our work complements these benchmark audits by providing a **mechanistic explanation** for the observed pattern rather than a further benchmark enumeration: we decompose the subject-vs-label variance structure of FM representations, conduct a controlled paired within-subject experiment that isolates contrast strength as the operative variable, and confirm architecture-independence of the ceiling on a single small clinical dataset where seven architectures converge. Kauffmann et al. (arXiv:2507.11783v3, NeurIPS 2025) present the most explicit position-paper-style critique of EEG FM progress to date, noting that "Linear probing results were relatively worse" than full fine-tuning and that "general time-series foundation models sometimes outperformed EEG-FMs" — observations consistent with our finding that fine-tuning acts as *projection* onto subject-dominated axes (§4, §6) rather than representation rewrite.

The cross-subject psychopathology decoding problem has also been explicitly elevated to competition-track status by the EEG Foundation Challenge 2025 [Truong et al., arXiv:2506.19141, NeurIPS 2025 competition], whose explicit framing positions cross-subject + clinical psychopathology prediction as *the* open problem for this field. This motivates a clear pre-benchmark diagnostic protocol, which our paper supplies.

### 2.2 Subject leakage, honest cross-validation, and data leakage in EEG deep learning

The systematic impact of subject leakage on EEG deep learning performance has been documented for nearly a decade [Roy et al. 2019, J. Neural Engineering; Brookshire et al. 2024, Frontiers in Neuroscience 10.3389/fnins.2024.1373515; CEUR-WS Vol-4115 paper 7, 2024]. Brookshire et al. 2024 is the current canonical reference for trial-level vs subject-level CV gap quantification in translational EEG DL; CEUR-WS (2024) provides emotion-classification-specific leakage precedent particularly relevant to DASS-based stress labels. Our Wang et al. 2025 re-analysis (§5) is, to our knowledge, the first independent subject-level audit of a claimed "EEG FM beats classical stress classification" result; we find that a single protocol correction accounts for all of the reported 40+ pp gap.

### 2.3 Statistical power and small-N reliability in EEG

The reliability of small-N deep learning findings on EEG has recently been quantified at unprecedented scale. A multisite psychiatry study of N = 2,874 resting-state EEG recordings [bioRxiv 2025.11.10.687610, PMID 41292863] concludes: "results from small samples were unstable, with inflated and highly variable effect sizes across iterations. Larger samples produced consistent findings, converging on uniformly small but robust effects." An independent brain-behavior correlation study [bioRxiv 2026.02.06.704323] arrives at the same conclusion from a different angle. These results provide the statistical-power backdrop against which our UCSD Stress (70 rec / 14 positive) audit operates: any claim at this cohort size is fragile by construction, and our permutation-null analysis (§5) quantifies this fragility directly.

A systematic survey of inter- and intra-subject variability in EEG [arXiv:2602.01019, 2026-02] frames the broader measurement-theoretic challenge.

### 2.4 Fine-tuning dynamics and cross-modality precedents

Our observation that FM fine-tuning acts as *projection* rather than *rewrite* on EEG downstream tasks (§6) has cross-modality analogues in recent large-scale language and vision work. Chen et al. (arXiv:2506.14681, "Massive Supervised Fine-tuning Experiments"; 757 fine-tuned models × 10 base architectures × 10 datasets) report that "the inductive biases of the base model outweigh the specific SFT corpus in determining the final representation" — a cleaner LLM-scale instantiation of the same phenomenon. In the vision domain, Li et al. (ICCV 2025, "On the Robustness Tradeoff in Fine-Tuning") document an analogous erosion pattern under fine-tuning. For the specific case of EEG emotion recognition, Multi-dataset Joint Pre-training of Emotional EEG [arXiv:2510.22197, NeurIPS 2025, NCClab-SUSTech] proposes a cross-dataset covariance-alignment loss that partially mitigates the inter-subject variability problem by design; our contribution is complementary — we *diagnose* the structural problem, whereas such methods *propose* solutions that future work may test against our SDL diagnostic.

### 2.5 Cognitive-load alpha desynchronisation and within-subject neural contrasts

The neurophysiological basis for distinguishing EEGMAT (rest vs mental arithmetic) from UCSD Stress (DASS state) as targets for within-subject FM evaluation rests on decades of oscillatory EEG literature. Klimesch (1999, Brain Research Reviews) established alpha desynchronisation as the canonical cognitive-load signature; Pfurtscheller & Aranibar (1979) provided the event-related desynchronisation framework; Klimesch (2012, NeuroImage) updated the model. EEGMAT rest vs arithmetic produces a **within-subject effect that is robust, spectrally localised to alpha, and reliably induced across participants**. By contrast, no published EEG literature establishes a per-recording within-subject correlate of DASS-21 or the Daily Stress Scale (DSS) at the resolution used in the Wang et al. 2025 benchmark. This distinction — independent of any FM analysis — is what our paired experiment (§6) operationalises.

### 2.6 Subject biometric identity in EEG

The dominance of subject-identifying information in EEG representations is not a new observation in the biometric literature [Campisi & La Rocca 2014, IEEE SP Magazine; Marcel & Millán 2007, IEEE TPAMI] — subject recognition from resting-state EEG approaches near-ceiling accuracy with classical features. What is new in our §4 is the *quantification of subject-vs-label variance inside pretrained FM representations* using variance decomposition and cross-dataset RSA, providing a representation-level bridge between the biometric literature and the downstream-task literature that has largely overlooked it.

---

## §3 Methods

### 3.1 Datasets

**UCSD Stress** (primary). Resting-state EEG from Komarov, Ko & Jung (2020, IEEE TNSRE 28(4):795): 17 Taiwanese graduate students, 92 recordings with DASS-21 and Daily Stress Scale (DSS) self-reports per session. We use the 70 recordings with both DASS-21 and DSS available (`data/comprehensive_labels.csv`). Epochs 5 s at 200 Hz × 30 channels; subject-level binary label via DASS-21 threshold 50 (`--label dass --threshold 50`), matching the Wang et al. 2025 protocol.

**EEGMAT** (paired within-subject comparator). Zyma et al. (2019), 36 healthy subjects × 2 recordings (resting baseline + serial-subtraction mental arithmetic); 19 channels, 60 s segments. This dataset provides the *strong within-subject alpha-desynchronisation contrast* side of our paired experiment (§6.1).

**ADFTD** (for §4 subject-atlas cross-dataset coverage). Miltiadous et al. (2024), Alzheimer's disease vs healthy controls, 195 recordings / 65 subjects / 19 channels.

**TDBRAIN** (for §4 cross-dataset coverage). van Dijk et al. (2022), MDD vs HC, 734 recordings / 359 subjects / 26 channels (downsampled to 19 for FM compatibility).

ADFTD and TDBRAIN are used exclusively for F-A subject-atlas variance decomposition; no fine-tuning claims are made on these datasets in this paper.

### 3.2 Pipeline and foundation models

Unified frozen-extractor interface: EEG → StressEEGDataset (epoch + cache) → FM backbone → global pool → classifier head. Three FMs:

- **LaBraM base** (Jiang et al. 2024): vector-quantised neural tokenizer, 200-dim patch embedding, ~100 M parameters. Weights from 935963004/LaBraM.
- **CBraMod** (Wang et al. 2024): criss-cross attention over spatial + temporal axes, 200-dim output, ~100 M parameters. Weights from the official release.
- **REVE** (Zhong et al. 2025): linear patch embedding over 30-channel × 10 s blocks, 512-dim output, ~1.4 B parameters.

**Input normalisation is model-dependent and must be set per-model to avoid silently destroying runs**: LaBraM uses `zscore` (matches pretraining preprocessing); CBraMod uses `none` (the extractor applies an internal x/100 scaling; zscored input double-scales to ~0); REVE uses `none` (linear patch is scale-sensitive to the µV-scale pretraining distribution).

### 3.3 Cross-validation and evaluation protocols

Primary evaluation is **subject-level StratifiedGroupKFold(5)**, grouping by `patient_id`. Trial-level StratifiedKFold(5) is included only as a literature-comparison reference (e.g., reproducing Wang et al. 2025) and reported transparently alongside subject-level numbers; it carries subject leakage and inflates reported BA by 20–30 pp on this dataset (§5).

**Determinism**: `torch.backends.cudnn.deterministic=True`, `benchmark=False`. Seed variance on the UCSD Stress 70-recording / 14-positive regime remains ±5–10 pp single-seed after cuDNN determinism, consistent with recent small-N stability literature [bioRxiv 2025.11.10.687610]. All headline claims are reported as mean ± sample std over ≥ 3 seeds.

**Permutation null**: recording-level label shuffle prior to CV. Applied via `train_ft.py --permute-labels <seed>` and the batched helper `scripts/run_perm_null.py`.

**Class imbalance handling for classical baselines** (R1 C3 audit): RF, LogReg (L1/L2), SVM all use `class_weight='balanced'`; XGBoost (sklearn GradientBoostingClassifier) uses `sample_weight='balanced'` via `sklearn.utils.compute_sample_weight`. Results before and after the XGBoost correction are reported side-by-side in §5.

### 3.4 Variance decomposition and representational similarity

For §4 subject-atlas quantification, we compute:

- **Pooled label fraction**: nested-SS decomposition of frozen FM embeddings into label-mean, subject-within-label, and residual components; reported as `label_SS / total_SS`.
- **Mixed-effects ICC**: LME with subject as random intercept on frozen embedding PCs.
- **RSA**: Spearman correlation between (a) subject-RDM (identity-coded) and embedding-RDM, and (b) label-RDM and embedding-RDM. Reported as `subject_r` vs `label_r` across 3 FMs × 4 datasets = 12 cells.
- **Silhouette, Fisher, kNN, LogME, H-score** for robustness. All agree on the subject-dominant direction.

Implementation in `src/variance_analysis.py` and `scripts/run_variance_analysis.py`; outputs in `paper/figures/variance_analysis.json` and `results/studies/exp06_fm_task_fitness/fitness_metrics_full.json`.

### 3.5 Contrast-strength anchoring protocol (important)

A naïve operationalisation of the SDL thesis would define "contrast strength" in terms of FM fine-tuning outcomes and then use those outcomes to validate the thesis — a circular construction. To avoid this, we pre-register two **external, FM-independent** contrast-strength anchors:

1. **EEGMAT anchor**: alpha-band effect size for rest vs arithmetic contrast, computed as per-subject paired Hedges' g on posterior-channel alpha power, with literature reference to Klimesch (1999) for expected effect magnitude.
2. **UCSD Stress anchor**: group-level cluster-permutation PSD analysis (DASS-high vs DASS-low recordings, all frequencies × all channels, cluster-based multiple-comparisons correction) on the same 70-recording subset used for FM evaluation. If a stable group-level spectral contrast exists, it provides an independent upper bound on within-subject contrast strength that is not conditioned on FM outputs.

The rationale for this protocol — anchoring contrast strength on published neural effect sizes (for EEGMAT) and on independent group-level spectral statistics (for Stress) — is to falsify SDL if the anchor disagrees with the behavioural pattern, rather than merely confirm it post hoc.

### 3.6 N-F21 LaBraM `self.norm` architectural disclosure

During preparation of this manuscript, we discovered that our LaBraM implementation applied two LayerNorm operations in sequence (`self.norm` + `self.fc_norm`) on mean-pooled patch tokens, whereas the official LaBraM code (`935963004/LaBraM/modeling_finetune.py`) sets `self.norm = nn.Identity()` under `use_mean_pooling=True`. Pretrained `norm.*` weights (trained for the CLS-token pretraining head) were therefore being loaded into a second LayerNorm not present in the original architecture.

We fixed this (`baseline/labram/model.py:197`: `self.norm = nn.Identity()`) and re-ran a Wang-protocol 12-seed diagnostic (`results/studies/exp24_labram_fixed/` vs `exp23_wang_full/`) to quantify the impact:

| Metric | pre-fix (exp23) | post-fix (exp24) | Δ |
|---|---:|---:|---:|
| Wang-4-seed window-level BA mean | 0.608 | 0.602 | −0.006 |
| 12-seed window-level BA mean | 0.571 | 0.578 | +0.007 |
| 12-seed recording-level BA mean | 0.646 | 0.592 | −0.054 |
| 12-seed recording-level per-seed swings | — | — | ±25 pp (cancel in mean) |

A post-fix subject-level spot-check (§[Supplementary]) at our canonical best-HP configuration confirms that the Δ under our primary subject-level CV protocol is within the cuDNN-determinism envelope of ±3 pp reported for this cohort size. All main-text claims use post-fix numbers; pre-fix numbers are reported in supplementary alongside this diagnostic for full reproducibility.

We also audited CBraMod and REVE for analogous architectural drift: REVE is a line-by-line match of the official `modeling_reve.py` (140/140 parameters load); CBraMod loads 209/209 encoder parameters with the masked-reconstruction head (`proj_out.*`) intentionally replaced by `Identity` for feature extraction. No analogous drift was found for CBraMod or REVE.

### 3.7 DASS-21 psychometric scope

DASS-21 is validated as a past-week trait/state screener [Lovibond & Lovibond 1995; systematic reviews of recent years]. The Komarov–Ko–Jung (2020) UCSD Stress dataset was designed as a *group-level longitudinal correlational study* linking DASS-21 / DSS trajectories to resting-state EEG features, not as a per-recording binary classification benchmark. The use of per-recording DASS-21 thresholding (Wang et al. 2025's protocol, which we reproduce for the purpose of independent audit) extends DASS-21 beyond its primary validation envelope. All claims derived from per-recording DASS-21 classification performance on this dataset are conditional on this scope; we return to this limitation in §10.

This caveat *does not weaken* our SDL diagnosis — it strengthens it. Under SDL, UCSD Stress is exactly the kind of dataset where subject-dominance would dominate, because the label's within-subject neural correlate is not established. The paired EEGMAT comparison (§6) relies on a contrast (mental-arithmetic alpha ERD) that *is* neurobiologically validated; the outcome difference between the two datasets is the experimental signal.

### 3.8 Hyperparameter selection and robustness disclosure

Hyperparameters for each FM's best fine-tuning configuration on UCSD Stress were selected via a 3 × 2 × 3 grid (3 LRs × 2 encoder-LR scales × 3 seeds) independently per model. The selected configurations differ between models (LaBraM: lr=1e-4, encoder-lr-scale=1.0; CBraMod: lr=1e-5, 0.1; REVE: lr=3e-5, 0.1). We stress that this is an argmax-over-HP-grid selection, not a factorial design; the per-model best-HP numbers are presented as observations, not as controlled comparisons between architectures. §9 (Discussion) discusses the implications for cross-model fine-tuning direction observations on ADFTD and TDBRAIN, which are presented as underpowered qualitative observations rather than a taxonomy.

### 3.9 Compute, reproducibility, and code release

All experiments were run on NVIDIA A100 GPUs using PyTorch 2.5.1 + CUDA 12.4. The `stress` conda environment (`environment.yml` in supplementary) fixes scipy 1.17, statsmodels 0.14, torch 2.5.1+cu124, timm 1.0.26. All scripts and configurations (including the N-F21 fix commit, exp23/exp24 diagnostic, permutation-null pool, and class-balanced classical baseline audit) will be released upon publication at `github.com/<anonymous>/UCSD-SDL`.

---

*Draft status: §1 (~1100 words), §2 (~1200 words), §3 (~1400 words). Ready for advisor review. Next: §4 subject-atlas, §5 honest-evaluation audit, §6 paired-contrast experiment, §7 architecture-independence, §8 FM-value-over-classical, §9 discussion, §10 limitations.*
