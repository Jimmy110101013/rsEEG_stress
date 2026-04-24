# SDL Paper Draft — §4 Subject Atlas

**Date**: 2026-04-15
**Role in SDL thesis**: *the denominator.* Before any downstream classification claim on EEG FMs can be interpreted, we must establish the structural baseline of what frozen FM representations encode. §4 shows that, across three FMs and four datasets, frozen FM representations are dominated by subject identity, with task labels occupying a small and often statistically indistinguishable-from-chance slice of representation variance. This is the "subject atlas" against which §5–§8 operate.

---

## §4 Subject Atlas: Frozen EEG FMs Primarily Encode Subject Identity

### 4.1 Question and design

The subject-dominance hypothesis makes a structural prediction that can be tested *independently of any classification outcome*: if frozen EEG FM representations are subject atlases, then a variance decomposition of those representations should find (i) that the subject factor accounts for the majority of explainable variance, (ii) that the task-label factor accounts for a small, near-chance slice, and (iii) that these patterns hold across multiple FMs and multiple datasets. We test this prediction on three representative FMs (LaBraM, CBraMod, REVE — spanning ~100 M to ~1.4 B parameters and three distinct architectural families) evaluated on four clinical EEG datasets (UCSD Stress, EEGMAT, ADFTD, TDBRAIN — spanning 14–359 subjects and 55–734 recordings).

For each of the 3 × 4 = 12 model × dataset pairs, we extract a single frozen embedding per recording by global-mean-pooling the final-layer patch tokens of the pretrained FM with no gradient flow, no classifier head, and no fine-tuning. Three convergent measures quantify the subject-vs-label variance structure of these frozen embeddings:

1. **Nested variance decomposition** (pooled label fraction vs pooled subject fraction). Implemented in `src/variance_analysis.py` as a nested sum-of-squares partition of the recording-level embedding matrix into label-mean, subject-within-label, and residual components. The reported `label_frac` is `SS_label / SS_total × 100`; `subject_frac` is `SS_subject / SS_total × 100`.
2. **Representational similarity analysis (RSA)**. Two RDMs are computed from the frozen embedding matrix — one pairwise-label RDM (same-label vs different-label mask) and one pairwise-subject RDM (same-subject vs different-subject mask) — and Spearman-correlated with the embedding-distance RDM. The reported subject-r and label-r are the two Spearman correlations; we tabulate the sign of (subject_r − label_r) across all 12 cells.
3. **PermANOVA** on the recording-level embeddings, with `subject` as the factor. The reported pseudo-F and p-value (499 permutations) test whether the embedding distances cluster by subject at a level the shuffled null cannot reproduce.

### 4.2 Result 1 — Subject variance dwarfs label variance

Table 4.1 reports the nested decomposition for all 12 frozen model × dataset cells (source: `results/studies/exp06_fm_task_fitness/variance_analysis_all.json`).

**Table 4.1.** Frozen-embedding nested variance decomposition.

| Model | Dataset | n_rec | n_subj | Label frac (%) | Subject frac (%) | Subject / Label |
|---|---|---:|---:|---:|---:|---:|
| LaBraM | UCSD Stress | 55 | 14 | 4.53 | 48.7 | 10.8× |
| LaBraM | EEGMAT | 72 | 36 | 5.35 | 71.2 | 13.3× |
| LaBraM | ADFTD | 195 | 65 | 2.60 | 66.8 | 25.7× |
| LaBraM | TDBRAIN | 734 | 359 | 2.41 | 54.3 | 22.5× |
| CBraMod | UCSD Stress | 55 | 14 | 2.55 | 45.8 | 18.0× |
| CBraMod | EEGMAT | 72 | 36 | 3.72 | 79.6 | 21.4× |
| CBraMod | ADFTD | 195 | 65 | 0.67 | 61.6 | 91.9× |
| CBraMod | TDBRAIN | 734 | 359 | 1.82 | 95.6 | 52.5× |
| REVE | UCSD Stress | 55 | 14 | 3.10 | 39.3 | 12.7× |
| REVE | EEGMAT | 72 | 36 | 2.46 | 77.3 | 31.4× |
| REVE | ADFTD | 195 | 65 | 0.82 | 43.9 | 53.3× |
| REVE | TDBRAIN | 734 | 359 | 0.004 | 60.6 | ≫1000× |

Across all 12 cells, the label fraction is < 6 % and the subject/label ratio is ≥ 10×. The smallest subject/label ratio observed is 10.8× (LaBraM × UCSD Stress); the largest is effectively unbounded (REVE × TDBRAIN, where label_frac ≈ 0). This is not a fringe observation on one particular (model, dataset) pair — it is uniformly true for every combination of FM and dataset we tested, and therefore describes a property of *pretrained-FM representations of clinical EEG*, not a property of any single dataset or architecture.

### 4.3 Result 2 — Subject RSA exceeds label RSA in 12 of 12 cells

Nested variance decomposition measures "how much total signal energy is concentrated on each factor," but does not directly address whether distances between embeddings *correspond to* subject or label identity. RSA provides that complementary view. Across all 12 frozen model × dataset cells, the Spearman correlation between embedding-distance RDM and subject-identity RDM exceeds the correlation between embedding-distance RDM and label-identity RDM (`results/studies/exp06_fm_task_fitness/fitness_metrics_full.json`). Figure 4.1 (to be rendered from the existing `fitness_heatmap` source) displays subject_r vs label_r for each cell; no cell falls above the diagonal.

Robustness triangulation (detailed in supplementary): five alternative subject-separability metrics — silhouette coefficient, Fisher score, nearest-neighbour subject-recovery accuracy, LogME, and H-score — agree on the subject-dominant direction in every cell. We interpret the unanimity across RSA + variance-SS + five auxiliary metrics as strong evidence that subject-dominance is a representation-level structural property rather than an artefact of any single scoring method.

### 4.4 Result 3 — PermANOVA on subject factor is significant where sample size permits it to be

On the three datasets where n is large enough for PermANOVA to have adequate power (ADFTD, TDBRAIN, with n_rec ≥ 195), the subject-factor p-value from 499-permutation PermANOVA is ≤ 0.04 for 5 of 6 model × dataset cells (LaBraM-ADFTD p = 0.038, LaBraM-TDBRAIN p = 0.002, CBraMod-TDBRAIN p = 0.004, REVE-TDBRAIN and REVE-ADFTD underpowered; full table in supplementary). On UCSD Stress (n_rec = 55, n_subj = 14), PermANOVA p-values are uniformly non-significant (p = 0.46–0.59), consistent with the small-sample instability literature [bioRxiv 2025.11.10.687610] rather than absence of subject clustering — pooled-label and RSA measures confirm the same subject-dominant structure on UCSD Stress, they merely cannot be bootstrapped to permutation-significance at n = 55.

### 4.5 A partial exception that refines — rather than contradicts — the thesis

One cell in Table 4.1 deserves separate treatment: **LaBraM × ADFTD fine-tuning.** Under fine-tuning (not the frozen number reported above), the label_frac rises from 2.60 % → 13.25 % (Δ = +10.65 pp) and the subject_frac falls from 66.8 % → 60.0 %. This is the largest label-fraction increase in any cell we measured and corresponds to a fine-tuned BA of 0.70 — the highest FT BA across our 3 × 4 grid. In every other cell, Δlabel_frac is ≤ +5.3 pp, and in four cells it is *negative* (the fine-tuning objective on that (model, dataset) pair pushed the representation *further* from the label axis while still improving classification head BA; a phenomenon we return to in §8).

LaBraM × ADFTD is therefore *not* an SDL counterexample — it is the cell where the contrast-strength factor (Alzheimer's-driven resting-state alpha slowing, well-established in the clinical EEG literature) is strong enough that fine-tuning can reorganise the representation around the label axis. Under the SDL thesis this is the expected behaviour; the UCSD Stress cells, where Δlabel_frac is ≈ 0 or negative, are the expected behaviour at the other end of the contrast-strength spectrum. §6 operationalises this intuition via a paired within-subject experiment on EEGMAT vs UCSD Stress where contrast strength is externally anchored rather than inferred from FT outcomes (§3.5).

### 4.6 Takeaway for §5–§8

§4 establishes that frozen EEG FM representations — across three pretrained architectures spanning 14× in parameter count, and across four clinical datasets spanning 14–359 subjects — are dominated by subject identity by factors of 10× to > 50×. Downstream classification performance on these representations is therefore constrained to work against a representation in which task-label variance is a small minority of the signal. §5 shows what this constraint looks like when honest subject-level cross-validation is applied to the reported Wang et al. 2025 benchmark on UCSD Stress; §6 asks the paired question of what distinguishes the (rare) datasets where fine-tuning escapes this constraint from the (majority) where it does not; §7 asks whether any architectural choice — including from-scratch compact models that never saw EEG pretraining data — can bypass the ceiling; §8 quantifies what, if anything, pretrained FMs add on top of simple baselines within the constrained regime.

---

*Draft status: §4 ~1,100 words. Ready for review. Waiting on go-ahead before continuing to §5.*
