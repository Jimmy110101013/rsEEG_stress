# SDL Paper Draft — §8 What FM Pretraining Contributes Inside the Ceiling

**Date**: 2026-04-15
**Role in SDL thesis**: *within-ceiling FM value quantification.* §5 established the ceiling, §6 identified contrast strength as its governor, §7 ruled out architecture and scale as rescue levers. §8 answers the remaining symmetric question that reviewers will reasonably ask: granted that FM fine-tuning does not rescue contrast-bounded tasks, does pretraining nevertheless contribute *anything* above classical baselines? The answer is: a small, consistent, representation-level bump — not a free-lunch advantage, and not the 15–40 pp gains suggested by evaluation-inflated benchmarks.

---

## §8 Two-Regime FM Value: Rescue Under Contrast, Bump Without

### 8.1 The accounting

The cleanest way to audit what pretraining contributes is to hold the classifier head constant across representations. We compare four representation families on UCSD Stress under the same honest subject-level protocol: (i) hand-crafted classical band-power features passed to a class-balanced XGBoost; (ii) frozen linear-probe heads trained on pretrained FM embeddings with no gradient flow to the backbone; (iii) end-to-end fine-tuned FMs; and (iv) from-scratch convolutional baselines for reference (§7).

**Table 8.1.** Representation comparison on UCSD Stress, subject-level CV, class-balanced where applicable.

| Representation | Best method in family | Subject-level BA |
|---|---|---:|
| Classical band-power | XGBoost (class-balanced) | 0.553 ± 0.081 |
| Frozen FM | LaBraM frozen linear probe (8-seed) | 0.605 ± 0.032 |
| Fine-tuned FM | REVE FT (best model × best HP, 3-seed) | 0.577 ± 0.051 |
| From-scratch EEG CNN | ShallowConvNet (3-seed) | 0.557 ± 0.031 |

Two facts stand out. First, frozen FM linear probes beat the best class-balanced classical baseline by ~5 pp, sit above from-scratch EEG CNNs by ~5 pp, and — within the single-model seed std — are not reliably exceeded by end-to-end FT on this cohort. Second, the spread across the four families is ~5 pp; no representation advantage approaches the 15–40 pp gap reported under trial-level CV in the benchmark literature.

### 8.2 The 5 pp is consistent with, but does not prove, a small representation-quality bump

The 5 pp frozen-FM-over-classical advantage is small enough to be worth interpreting cautiously. In the 70-recording cohort with 14 positives, a shift of that magnitude corresponds to correctly classifying one to two additional recordings per fold on average, and the per-fold variance admits multi-pp swings driven by which single subject enters the validation partition. We therefore refrain from strong claims about the 5 pp — we report it as an observation, consistent with the hypothesis that pretrained FM embeddings carry a modestly richer representation of the 30-channel spectral structure than a hand-crafted band-power feature vector, and consistent with the observation from the cross-model neural band-consensus analysis (§6.4) that FMs on Stress do capture some spatially-structured signal (posterior-weighted, §F-NEURO on classical alpha loadings in §F-E) that simple band-power features capture more coarsely.

Crucially, **the 5 pp bump lives entirely inside the 0.44–0.58 architecture-independent band identified in §7**. FM pretraining does not move the classification out of the ceiling regime; it places the classifier slightly higher within it. This is the within-ceiling version of the SDL story.

### 8.3 The anchored-contrast regime: FM pretraining provides real rescue

The same accounting performed on datasets where the within-subject contrast is neurally anchored produces a qualitatively different picture. Table 8.2 summarises the representation comparison on EEGMAT (Klimesch alpha-ERD anchor) and on ADFTD (clinical alpha-slowing anchor), both under subject-level CV.

**Table 8.2.** Representation comparison across contrast-anchored datasets (BA, 3-seed mean).

| Dataset | Best classical | Best frozen FM | Best FT FM | Anchored-over-classical gain |
|---|---:|---:|---:|---:|
| EEGMAT (rest vs arithmetic) | ~0.60 (literature) | ~0.68 (frozen LP) | 0.73 (LaBraM FT) | ~+13 pp |
| ADFTD (AD vs HC) | ~0.60 (literature) | ~0.65 (frozen LP) | 0.70 (LaBraM FT) | ~+10 pp |

On both anchored-contrast datasets, FT delivers an additional rescue step above frozen LP. On EEGMAT this step is specifically produced by projection of the label axis onto the subject-dominated representation rather than by representation rewrite (§4.5, §6.2). On ADFTD, variance decomposition shows the largest label-fraction increase under FT in our 3 × 4 grid (frozen 2.60 % → FT 13.25 %, Δ = +10.65 pp) — the only cell in which the fine-tuning objective substantively reorganises the representation around the label axis. Contrast anchoring thus both enables and is measurable within the representation.

### 8.4 Two-regime summary

Combining §5–§8, the empirical picture of what EEG FM pretraining contributes divides cleanly into two regimes. In the **anchored-contrast regime** (EEGMAT, ADFTD), frozen FM representations already outperform classical baselines by ~5–8 pp, and fine-tuning adds a further ~5–10 pp rescue on top of that — a double benefit, of which only the fine-tuning component is contrast-dependent. In the **contrast-bounded regime** (UCSD Stress per-recording DASS, UCSD Stress longitudinal DSS, and likely any clinical label for which no within-subject neural correlate has been independently established), frozen FM representations still contribute a ~5 pp bump above classical baselines, but fine-tuning does not add further rescue and in some (model, dataset) cells actively erodes the frozen representation (for example, LaBraM FT on Stress drops from frozen LP 0.605 to FT 0.524, a ~8 pp erosion; §F-C.2).

The two-regime picture aligns with what concurrent large-scale EEG FM benchmark audits have reported at the aggregate level (EEG-Bench, Kauffmann et al., EEG-FM-Bench, AdaBrain-Bench) — namely that FM fine-tuning sometimes matches and sometimes fails to match compact baselines under strict cross-subject evaluation. §6 supplies the mechanistic layer under this aggregate pattern: the determining factor is the presence or absence of a within-subject neural anchor for the target label. §7 supplies the complementary architectural robustness check: in the contrast-bounded regime, the anchor's absence cannot be compensated by architectural choice.

### 8.5 Implications for practitioners

The practical consequence of §8 for a group considering deploying an EEG FM on a small clinical cohort is straightforward and pre-benchmark: **quantify the within-subject contrast anchor for your target label before committing to an FM fine-tuning plan**. If the label has a robust literature-anchored within-subject EEG correlate, FM pretraining will likely provide a meaningful representation bump over classical baselines, and fine-tuning will likely provide an additional rescue gain — comparable to what §8.3 reports for EEGMAT or ADFTD. If no such anchor exists in the published literature, frozen FM embeddings may still provide a small representation advantage over classical features, but reported fine-tuned benchmarks on that cohort should be interpreted as bounded by the subject-dominance ceiling and treated with the evaluation discipline detailed in §5.

§9 discusses how this two-regime diagnostic relates to broader concurrent critiques of EEG FM benchmarks and positions the SDL thesis within the surrounding benchmark-audit and critical-review literature.

---

*Draft status: §8 ~1,150 words. Ready for review.*
