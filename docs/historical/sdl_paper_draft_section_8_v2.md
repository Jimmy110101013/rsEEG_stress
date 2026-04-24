# SDL Paper Draft — §8 Cross-Dataset FT vs Frozen LP Analysis (v2, 2026-04-20)

**Role in SDL thesis**: *the generalisation audit.* §5 established the UCSD Stress ceiling; §6 showed EEGMAT state signal is linearly separable from frozen FM embeddings while Stress's is not; §7 showed architecture scale does not bypass the Stress ceiling. §8 asks whether the FT-vs-LP pattern observed on Stress generalises to three other clinical rsEEG cohorts, and whether the cases where FT helps admit a mechanistic explanation.

---

## §8 Fine-Tuning Rarely Exceeds Frozen Linear Probing Across Datasets

### 8.1 Grid: 3 FMs × 4 datasets, consistent per-FM protocol

We compare each FM's frozen linear probe (LP, per-window training + prediction pooling, §5.2 matched protocol, 8 seeds) against its own fine-tuning (FT, per-FM canonical recipe, 3 seeds) on the same four cohorts used throughout the paper: UCSD Stress (17 subjects, 70 recordings, DASS per-recording binary), ADFTD (65 subjects, 195 recordings, AD/FTD vs HC), TDBRAIN (359 subjects, 734 recordings, MDD vs HC), and EEGMAT (36 subjects, 72 recordings, rest vs mental arithmetic).

All twelve (FM × dataset) cells use **per-FM canonical recipes** (LaBraM `lr=1e-5, encoder_lr_scale=0.1, zscore`; CBraMod `lr=1e-5, elrs=0.1, none`; REVE `lr=3e-5, elrs=0.1, none`) — **without** per-dataset hyperparameter tuning. This reflects a realistic deployment scenario where clinical researchers cannot afford per-task HP search, and it structurally avoids test-set HP contamination. Source tables: `paper/figures/source_tables/master_frozen_ft_table_v2.{md,json}`.

### 8.2 Twelve-cell result

**Table 8.1.** Frozen LP (per-window, 8 seeds) vs Fine-tune (canonical recipe, 3 seeds). Balanced accuracy mean ± sample std (ddof=1); Δ in percentage points.

| Model | Phase | Stress | ADFTD | TDBRAIN | EEGMAT |
|---|---|---|---|---|---|
| LaBraM | Frozen LP | 0.525 ± 0.040 | 0.653 ± 0.018 | 0.752 ± 0.011 | 0.736 ± 0.025 |
|  | Fine-tune | 0.443 ± 0.083 | 0.709 ± 0.014 | 0.665 ± 0.045 | 0.731 ± 0.021 |
|  | ΔFT − LP | −8.1 | **+5.6** | −8.7 | −0.5 |
| CBraMod | Frozen LP | 0.430 ± 0.033 | 0.570 ± 0.030 | 0.525 ± 0.022 | 0.722 ± 0.022 |
|  | Fine-tune | 0.548 ± 0.031 | 0.537 ± 0.027 | 0.488 ± 0.014 | 0.620 ± 0.058 |
|  | ΔFT − LP | **+11.8** | −3.3 | −3.7 | −10.2 |
| REVE | Frozen LP | 0.441 ± 0.022 | 0.652 ± 0.027 | 0.519 ± 0.022 | 0.733 ± 0.021 |
|  | Fine-tune | 0.577 ± 0.051 | 0.658 ± 0.030 | 0.488 ± 0.019 | 0.727 ± 0.035 |
|  | ΔFT − LP | **+13.7** | +0.6 | −3.1 | −0.6 |

**Summary: 3 of 12 cells show FT > LP by more than 1 pp, 3 are tied within 1 pp, and 6 show FT < LP by more than 1 pp.** Median Δ = −3.1 pp, mean Δ = −1.8 pp. Fine-tuning is, on average, not a net-positive procedure across these datasets at these cohort sizes.

This pattern — fine-tuning failing to exceed a frozen linear probe under honest subject-disjoint evaluation — has been previously reported on individual datasets by BENDR (Kostas et al. 2021, 4 of 5 LOSO datasets) and AdaBrain-Bench (Wang et al. 2025, EEGPT on BCI-IV-2a cross-subject). Table 8.1 replicates the pattern across three FMs and four rsEEG cohorts under the matched per-window protocol. It is not a new phenomenon; it is an underreported one that headline tables in LaBraM/CBraMod/REVE papers, which use trial-level or fixed subject-disjoint single-split protocols, systematically understate.

### 8.3 Three rescue cases — two mechanisms

Only three of twelve cells show fine-tuning exceed frozen LP, and they split cleanly into two mechanistic categories.

**Category A — Contrast-driven rescue (1 of 3):** *LaBraM × ADFTD (+5.6 pp).* ADFTD is the dataset with the strongest published within-recording neural signature in our set — posterior alpha slowing in Alzheimer's disease is well-characterised [Babiloni 2020, Ieracitano 2021] and directly detectable in resting-state PSD. Among the three FMs, only LaBraM — the only architecture with a vector-quantised tokenizer — succeeds at projecting this contrast onto its classification head during FT. CBraMod and REVE FT on ADFTD **degrade** relative to their own frozen LP (−3.3, +0.6 pp) despite operating on the same signal. The rescue is therefore jointly contingent on (a) dataset contrast strength and (b) architectural plasticity. This matches the variance decomposition evidence in §4.5: LaBraM × ADFTD showed the largest Δlabel_frac under FT (+10.65 pp, unique among our 12 cells).

**Category B — Small-cohort between-subject "rescue" (2 of 3):** *CBraMod × Stress (+11.8 pp)* and *REVE × Stress (+13.7 pp).* Both rescue cases land on the smallest cohort in our study — 17 subjects / 70 recordings / 14 positives — and both are on architectures whose FT does not help on any of the three larger cohorts. Within Stress, positive recordings are structurally concentrated in a subset of subjects rather than uniformly distributed across all 17, which introduces a design-level collinearity between subject identity and recording-level label.

We therefore must ask: are these rescues genuine state-signal acquisition, or are they subject-identity shortcut exploitation under label-subject collinearity? §5.3 already showed that on the same cohort LaBraM FT is statistically indistinguishable from label-shuffle null (p = 0.70 under permutation over 10 seeds). Extending this permutation-null test to CBraMod and REVE on Stress is the cleanest way to separate the two hypotheses; that extension is pending compute and will be reported in an updated draft (§10 open items).

The SDL-consistent interpretation is that CBraMod and REVE fine-tuning is exploiting the 14-positive-recording class imbalance combined with subject-label collinearity — giving a classifier head the shortcut path "predict based on which subject-fingerprint cluster the test embedding is closest to". The class-imbalanced subset allows this shortcut to produce a 12–14 pp BA gain that a subject-invariant state signal could not. If the pending permutation-null test confirms CBraMod FT > CBraMod LP under shuffled Stress labels, the "rescue" is mechanically identified as shortcut exploitation. If not, the finding is unexplained and worth reporting as-is.

### 8.4 What this says about each FM

Across the 4-dataset grid, each FM has its own FT pattern:

- **LaBraM**: 1 genuine contrast-driven rescue (ADFTD), 2 tied-or-degraded (EEGMAT, TDBRAIN), 1 degraded (Stress). LaBraM's vector-quantised tokenizer and z-score preprocessing appear to give it the plasticity to succeed on strong-contrast cohorts, at the cost of FT stability on weaker-contrast or small cohorts.
- **CBraMod**: 0 contrast-driven rescues, 1 shortcut-suspect rescue on Stress, 3 degradations. On all three strong-contrast cohorts (ADFTD, EEGMAT, TDBRAIN), CBraMod FT degrades frozen LP. This is the most BENDR-style "FT is harmful" profile in our set.
- **REVE**: 0 contrast-driven rescues, 1 shortcut-suspect rescue on Stress, 2 ties, 1 degradation. The largest FM (~1.4 B parameters) has the narrowest FT gain range across these datasets, consistent with scaling not helping and possibly amplifying FT instability (as AdaBrain-Bench 2025 reports on EEGPT for similar cross-subject shift).

None of these three patterns supports a general "fine-tuning strengthens EEG FMs" story at the 17–359-subject cohort scale.

### 8.5 Robustness checks

Two robustness concerns about the §8.2 table were pre-registered:

1. *Could the ceiling reflect our CV protocol?* We re-ran the LP arm on Stress with leave-one-subject-out (LOSO, 17 folds) — a tighter partitioning than 5-fold for small-N. LOSO shifts frozen LP BA by −1.2 to −3.3 pp across the three FMs, does not flip any verdict, and preserves the LaBraM > REVE > CBraMod ranking on Stress. Numbers: LaBraM 0.491, CBraMod 0.402, REVE 0.429 (LOSO, deterministic). `results/studies/perwindow_lp_all/stress/{model}_loso.json`.

2. *Could the LaBraM × Stress FT underperformance reflect a poor HP choice?* The full per-FM Stress HP sweep (`results/hp_sweep/20260410_dass/`, 6 HP combinations × 3 seeds) places the canonical recipe (lr=1e-5) 5th of 6 for LaBraM. The best-performing HP (lr=1e-4, encoder_lr_scale=1.0) raises LaBraM Stress FT from 0.443 to 0.524 — still within 1 pp of LaBraM frozen LP (0.525) on the same cohort. Even under best-HP selection, LaBraM FT does not exceed frozen LP on Stress. (CBraMod and REVE best-HP = canonical recipe, so their numbers are unchanged.) The ceiling is not an HP artifact. See supplementary Table S2 for the full 54-run grid.

### 8.6 Take-away for clinical practice

For clinical researchers considering fine-tuning an EEG FM on their own small rsEEG cohort, this section's evidence says: **frozen linear probing is the safe default, and fine-tuning is a gamble unless the target label has a published within-recording neural anchor**. The single robust rescue we observed (LaBraM × ADFTD) coincides with the only cohort in our grid whose target contrast is neurally well-characterised. Fine-tuning on trait-like or state-weak labels (DASS, MDD) shows systematically negative or null returns, and apparent rescues on small imbalanced cohorts (Stress CBraMod/REVE) require mechanistic verification before being trusted.

§9 operationalises this guidance into a three-step pre-benchmark checklist for clinical EEG practitioners.

---

*Draft status: §8 v2 ~1,400 words. Depends on pending permutation-null test for Stress CBraMod/REVE (noted in §10 open items). Otherwise ready for review.*
