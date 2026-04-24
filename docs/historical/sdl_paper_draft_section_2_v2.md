# SDL Paper Draft — §2 Related Work (v2, 2026-04-20)

**Revisions from v1**: (a) 20+ verified citation additions from 2026-04-19 literature survey (`docs/paper_strategy_v3_minimum_viable.md` §9 anchor list); (b) new subsection on aperiodic-as-fingerprint lineage (Demuru 2020, Lanzone 2023, Gao 2017, Donoghue 2020) supporting §6.5 FOOOF causal ablation; (c) updated EEG-FM critique lineage with Yang ICLR 2026, EEG-FM-Bench, arXiv:2601.17883, Brain4FMs, AdaBrain-Bench. All citations URL-verified on 2026-04-19.

---

## §2 Related Work

### 2.1 EEG foundation models and their clinical evaluation

The past two years have seen emergence of large EEG foundation models pretrained on aggregated public corpora. **LaBraM** [Jiang et al. 2024, ICLR; arXiv:2405.18765] uses a vector-quantised neural tokenizer with a 12-layer transformer, pretrained on ~2,500 hours of EEG across 20 datasets (5.8M to 369M parameters). **CBraMod** [Wang et al. 2024/2025, ICLR 2025; arXiv:2412.07236] applies criss-cross attention for channel × time modelling. **REVE** [Zhong et al. 2025] scales to ~1.4 billion parameters with linear patch embedding. **EEGPT** [Wang et al., NeurIPS 2024; OpenReview `lvS2b8CjG5`] introduces masked reconstruction with 10M parameters and chooses linear probing as its headline metric. **BIOT** [Yang et al., NeurIPS 2023; arXiv:2305.10351], **Brant** [NeurIPS 2023], and **NeuroLM** [Jiang & Wang, ICLR 2025; arXiv:2409.00101] span the rest of the landscape. Each reports competitive or state-of-the-art results on selected downstream tasks.

**LEAD** [Wang et al. 2025/2026, arXiv:2502.01678] is the dedicated Alzheimer's EEG foundation model most directly relevant to our ADFTD and Stress evaluations. LEAD reports LaBraM 91.1 % subject F1 on "ADFTD" under 8:1:1 subject-independent split with 5 seeds. The ADFTD cohort used by LEAD is a merged corpus of ds004504 (resting-state) and ds006036 (photic-stimulation); the RS-only number reported in LEAD's own Table 3 is 55.49 ± 15.08 %. Our ADFTD evaluation uses ds004504 (RS-only, the clinically deployable paradigm), which places our numbers in the comparable range to LEAD's RS-only benchmark, not its merged-paradigm headline.

### 2.2 Published critiques and benchmarks of EEG foundation models

Recent benchmarks have converged on a consistent finding: EEG FMs do not uniformly outperform simpler baselines under honest subject-disjoint evaluation.

**"Are EEG Foundation Models Worth It?"** [Yang, Sun, Li, Van Hulle, ICLR 2026; OpenReview `5Xwm8e6vbh`] benchmarks 7 EEG FMs across 6 evaluation protocols and reports that FT-FMs do not consistently beat compact models or classical decoders in data-scarce settings, linear probing is "consistently weak", and no clear scaling law holds.

**EEG-FM-Bench** [Xu et al. 2025/2026; arXiv:2508.17742] evaluates 7 FMs plus 2 general time-series models across 14 datasets × 10 paradigms and confirms gradient conflict between masked reconstruction and downstream classification objectives, as well as scaling laws that deviate from typical transformer trends (compact architectures with task-relevant inductive biases often outperform larger FMs).

**EEG Foundation Models: Progresses, Benchmarking, Open Problems** [arXiv:2601.17883, Jan 2026] surveys 12 FMs × 13 datasets × 9 paradigms under leave-one-subject-out (LOSO) and few-shot conditions, concluding that larger FMs do not consistently outperform specialist scratch models and that linear probing is often insufficient.

**Brain4FMs** [arXiv:2602.11558, Feb 2026] benchmarks 15 brain foundation models across 18 datasets and reports affect and communication tasks failing cross-subject, with supervised baselines exceeding most BFMs on emotion recognition.

**AdaBrain-Bench** [arXiv:2507.09882 v2 Aug 2025] reports EEGPT under Cross-Subject Transfer on BCI-IV-2a: LP 47.89 % versus FT 25.81 % — a 22 pp degradation under full fine-tuning, framed as large-model overfitting on cross-subject distribution shift.

**EEG Foundation Models: Critical Review** [arXiv:2507.11783 v3 Jul 2025 / NeurIPS 2025] presents the most explicit position-paper critique, noting that masked-reconstruction pretraining may encourage noise memorization and that contrastive, cross-modal, and event-aware objectives are more robust.

This prior critique lineage establishes that the FT-underperforms-LP pattern (and the broader weaker-than-claimed cross-subject performance of EEG FMs) is a published observation. Our contribution is orthogonal: we provide a **mechanistic explanation** for why the pattern occurs (§6.5 aperiodic-carried subject dominance) and a **deployment-oriented diagnostic** (§9 pre-flight checklist) for clinical researchers contemplating EEG-FM use on their own cohorts.

### 2.3 Subject leakage and honest cross-validation in EEG deep learning

The impact of subject leakage on EEG-ML accuracy inflation has been documented for nearly a decade. **Kostas et al. 2021** [BENDR, Frontiers in Human Neuroscience; arXiv:2101.12037] established leave-one-subject-out and leave-multi-subjects-out as standard protocols in EEG SSL; their Fig 3 / Table 2 report linear classification *outperforming* full fine-tuning in 4 of 5 datasets under LOSO — the original FT<LP observation that our §8 replicates at scale.

**Banville, Chehab, Hyvärinen, Engemann, Gramfort 2021** [J. Neural Eng. 18(4) 046020] formalise subject-stratified evaluation in clinical EEG SSL. **Gemein et al. 2020** [NeuroImage] benchmark TUH abnormal detection under subject-disjoint splits, establishing the reference baseline. Most directly, **Brookshire et al. 2024** [Frontiers in Neuroscience, DOI 10.3389/fnins.2024.1373515] survey 63 post-2018 translational EEG DL studies (AD, PD, ADHD, depression, schizophrenia, seizure) and show that segment-level holdout systematically inflates accuracy by 15–30 pp versus subject-level holdout. Brookshire 2024 is the canonical current reference for segment-vs-subject CV inflation in clinical EEG deep learning.

Our Wang et al. 2025 audit (§5) fits directly into this lineage: a single protocol correction (subject-level `StratifiedGroupKFold(5)` replacing trial-level `StratifiedKFold(5)`) accounts for ~20 pp of the reported 40 pp gap. The residual gap we leave explicitly as unresolved, attributed to unspecified Wang pipeline design choices we cannot audit without full training logs.

### 2.4 Variance decomposition and representation diagnostics

**Gibson, Lobaugh, Joordens, McIntosh 2022** [NeuroImage 252; pii/S105381192200163X] partition EEG signal variance into across-subject and across-block components at the raw-signal level, finding across-subject variance dwarfs across-block (task) variance. This is the nearest published precedent to our §4 η² nested variance decomposition, with a key distinction: Gibson 2022 operates on raw EEG features, while our §4 operates on frozen FM embeddings. Quantifying subject-vs-label variance structure *inside* pretrained FM representations using η² nested SS decomposition is, to our knowledge, previously unclaimed.

**Wagh, Varatharajah et al. 2022** [NeurIPS; dl.acm.org/doi/10.5555/3600270.3601807] report the closest-in-spirit diagnostic-framework paper for EEG-ML latent spaces under distribution shift; our §4 subject atlas applies analogous analysis specifically to EEG-FM representations on clinical rsEEG.

**EEG-FM-Bench** [Xu et al. 2025, arXiv:2508.17742] uses CKA, RSA, and subspace affinity between FM layers during FT to track representation evolution, reporting cross-subject balanced-accuracy drops of 15, 19, and 7 pp for LaBraM, CBraMod, and EEGPT respectively on SEED. Our §8 complements this finding at the representation-level (CKA / RSA against task labels and layer-to-layer, as Xu et al. do) with *balanced accuracy* at the downstream classification level on rsEEG clinical tasks.

**GC-VASE** [Mishra et al., arXiv:2501.16626] reports subject-identification decoding as a *desired* VAE property (89.81 % bAcc on ERP-Core, 70.85 % on SleepEDFx-20), embracing subject identity as signal rather than confound. This is an illuminating inversion of our SDL framing: the exact representation property that GC-VASE architects for is what §4 shows EEG FMs already have as an emergent property, with downstream cost to clinical discrimination.

### 2.5 Aperiodic 1/f spectral structure and EEG biometric fingerprinting

The physiological grounding for aperiodic 1/f structure comes from **Donoghue et al. 2020** [Nat Neurosci 24:1655-1665; nature.com/articles/s41593-020-00744-x], which introduces FOOOF as a method for parameterising neural power spectra into periodic (Gaussian peaks) and aperiodic (1/f) components. **Gao, Peterson, Voytek 2017** [NeuroImage; PMID 28676297] link the aperiodic exponent to cortical excitation-inhibition balance. **Waschke, Kloosterman, Obleser, Garrett 2021** [Neuron 109(5):751-766; cell.com/neuron] review the link between aperiodic variability and behavior.

The aperiodic-as-subject-fingerprint claim originates in **Demuru & Fraschini 2020** [arXiv:2001.09424 / PMID 32421651] for EEG and was replicated on MEG by **Lanzone et al. 2023** [NeuroImage; pii/S1053811923004111]. Both papers show that 1/f slope and offset alone identify subjects with near-ceiling accuracy and that the aperiodic component dominates subject-identifiability over periodic peaks. **Voytek et al. 2024** [Nat Commun; nature.com/articles/s41467-024-45922-8] provide biophysical grounding for why aperiodic EEG carries individual information.

The classical EEG biometric literature [Campisi & La Rocca 2014; Maiorana, Marcel, Ruiz-Blondet et al. 2015-2025] established that subject recognition from resting EEG approaches near-ceiling accuracy with classical features. What is new in our §6.5 is a causal demonstration on *pretrained FM embeddings* rather than correlational evidence on classical features, and the dissociation between aperiodic (subject-carrying) and periodic (subject-neutral) spectral content.

### 2.6 Subject-invariant representation learning in EEG

Methods attempting to remove subject identity from EEG representations — typically as a confound to enable better cross-subject task performance — include: adversarial domain-invariant training [**Özdenizci, Wang, Koike-Akino, Erdoğmuş 2020**, IEEE Access; PMC7971154], VAE + domain-adversarial topographic embedding [**Han, Etemad et al. 2021**, IEEE J-BHI; PMC7961341], mutual-information-based MLP subject disentanglement [**Zhang & Liu 2025**, arXiv:2501.08693], and per-subject low-rank adaptation [arXiv:2510.08059, 2025]. These methods treat subject identity as a nuisance variable to be removed. GC-VASE (§2.4) treats it as signal to be preserved.

Our framing is diagnostic rather than corrective: we quantify subject dominance without removing it, and argue that the *quantification itself* is a prerequisite for clinical deployment decisions. The pre-flight checklist in §9 places no prior preference on preserving or removing subject identity — it simply makes the trade-off visible to the clinical researcher.

### 2.7 Within-subject neural contrasts and task anchors

The neurophysiological basis for distinguishing EEGMAT (rest vs mental arithmetic) from DASS-based stress labels as targets for within-subject FM evaluation rests on decades of oscillatory EEG literature. **Klimesch 1999** [Brain Research Reviews] established alpha desynchronisation as the canonical cognitive-load signature. **Pfurtscheller & Aranibar 1979** provided the event-related desynchronisation framework. **Klimesch 2012** [NeuroImage] updated the oscillatory model. For Alzheimer's EEG, **Babiloni et al. 2020** [Clin Neurophysiol] and **Ieracitano et al. 2021** document alpha slowing as a robust within-recording signature.

By contrast, no peer-reviewed EEG literature establishes a per-recording within-subject neural correlate of DASS-21 or Daily Stress Scale at the temporal resolution used in per-recording classification benchmarks. This absence — rather than the presence of a published anchor — is what our §6 paired experiment operationalises and what §9 Check 1 codifies.

### 2.8 Small-sample reliability in EEG deep learning

A multisite study of N = 2,874 rsEEG recordings [bioRxiv 2025.11.10.687610] concludes that "results from small samples were unstable, with inflated and highly variable effect sizes across iterations. Larger samples produced consistent findings, converging on uniformly small but robust effects." An independent brain-behavior correlation study [bioRxiv 2026.02.06.704323] reaches the same conclusion from a different angle. Our UCSD Stress cohort (70 recordings, 14 positive, 17 subjects) sits at the fragile end of this spectrum; the §5.3 permutation-null result (p = 0.70) quantifies the fragility.

### 2.9 Positioning

Against this literature, this paper's contribution is:
- **Not** a new EEG-FM architecture or pretraining objective;
- **Not** another independent benchmark of existing FMs;
- **Rather**: a diagnostic protocol (§5–§6.5), a causal mechanism (§6.5 FOOOF ablation), and a clinical deployment checklist (§9) that integrate existing observations into a directly usable tool for a clinical researcher or benchmark reviewer.

The closest precedents are Brookshire et al. 2024 (segment-leakage warning, extended to FMs here) and Yang et al. ICLR 2026 ("Are EEG FMs Worth It?" — which answers "often not"; we extend the "why" to a specific mechanism and give a "when"-to-skip-them rule).

---

*Draft status: §2 v2 ~1,900 words. All citations URL-verified against the anchor list in `paper_strategy_v3_minimum_viable.md` §9.*
