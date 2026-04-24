# Paper Strategy v3 — Minimum Viable Diagnostic Path

**Created**: 2026-04-19
**Status**: Planning document. Supersedes prior SDL critique framing (v2) for the immediate publication target.
**Scope**: Three-experiment minimum viable path to a diagnostic critique paper on rsEEG EEG-FM subject-identity dominance.
**Target venue**: TNSRE / IEEE TBME short-to-regular, or eLife Methods. Timeline ~3 months.

---

## 0. Why v3 exists (today's literature-review pivot)

Four parallel literature-review agents (2026-04-19) established:

1. **"FM underperforms on subject-level CV" is no longer a novel finding.** Three 2026 papers already report this:
   - *Are EEG Foundation Models Worth It?* (Yang et al., ICLR 2026). OpenReview `5Xwm8e6vbh`.
   - arXiv:2601.17883 (Jan 2026) — 12 FMs × 13 datasets × 9 paradigms under LOSO + few-shot, "larger ≠ better".
   - Brain4FMs (arXiv:2602.11558, Feb 2026) — 15 BFMs × 18 datasets, cross-subject collapse on affect/communication.
2. **Cross-subject drop has been measured via CKA** on FM layer pairs: LaBraM −15%, CBraMod −19%, EEGPT −7% on SEED (EEG-FM-Bench, arXiv:2508.17742).
3. **Aperiodic 1/f as subject fingerprint is ESTABLISHED**: Demuru 2020 (arXiv:2001.09424), Lanzone 2023 ScienceDirect S1053811923004111.
4. **Gibson 2022 NeuroImage** (pii/S105381192200163X) performed variance partitioning on raw EEG finding across-subject variance dwarfs across-block variance — closest prior art, but on raw signal, NOT on FM embeddings.
5. **LEAD ADFTD = ds004504 (RS) + ds006036 (PS) merged** (confirmed by user from PDF §3/Appendix B). LEAD's headline 91.14% LaBraM subject F1 is on the merged corpus; RS-only reference is Table 3 = 55.49% ± 15.08%.

**Strategic consequence**: SDL must be repositioned from a *failure-discovery* paper to a *mechanism/diagnostic-tool* paper. The contribution is not "FMs fail", it is "we can diagnose *why* and measure it with ground-truth-validated tools".

## 1. Open gaps (what the three experiments claim)

| Gap | Prior art boundary | This paper's claim |
|---|---|---|
| No η² nested-SS decomposition on FM embeddings | Gibson 2022 did it on raw EEG; EEG-FM-Bench does CKA between layers, not vs subject_id | First nested-variance diagnostic of subject vs label dominance on EEG-FM embeddings |
| No ground-truth validation of any EEG variance-decomposition metric | All prior work uses real data only — unfalsifiable | First synthetic/semi-synthetic validation of a variance-decomposition metric on FM outputs |
| No causal test of aperiodic → subject dominance | Demuru 2020 is correlational | First aperiodic-ablation / aperiodic-swap intervention on FM embeddings |
| No Finn-style identifiability on EEG FM embeddings for clinical rsEEG | Finn 2015 on fMRI-FC; Amico & Goñi 2018 on EEG-FC but not FM | First Finn-style identifiability rate for EEG FMs on test-retest rsEEG |
| No correlation between subject-decodability and FT-collapse | Unclaimed | (Nice-to-have supplementary; core identifiability vs state-accuracy correlation is H3 of experiment δ) |

## 2. HP-contamination boundary (clarifies what can be reused)

From 2026-04-19 HP re-lockdown (per-FM official recipes + bs=128 + SWA):

| Result class | Affected by HP change? | Reason |
|---|---|---|
| Frozen feature extraction | No | Only depends on per-FM input norm (already locked) |
| Linear probing on frozen | No | Standard L2 sklearn, HP-insensitive at this scale |
| Variance decomposition / η² | No | No model training involved |
| Subject-ID probe | No | sklearn logreg |
| CKA / MI / RSA | No | Statistical metric |
| FOOOF / HHSA feature computation | No | Signal processing |
| FT-based downstream BA / F1 | **Yes — all prior FT results are potentially invalid** | HP only locked 2026-04-19 |
| exp_30 SDL pipeline | Depends on composition | Frozen parts survive; FT parts need rerun |

**Consequence for v3**: All three proposed experiments (α, β, δ) are **HP-contamination-immune** because they operate on frozen features, statistical metrics, and signal processing. This is the path with least wasted prior work.

---

## 3. Experiment α — Ground-Truth Validation of the Separability Metric

### Motivation
Before using a variance-decomposition metric to critique FMs, we must validate it against a known ground truth. Every prior variance-partitioning paper (Gibson 2022, Wagh 2022, etc.) uses real data only — the metric cannot be falsified on its own terms. A reviewer will ask: "How do you know your metric is measuring what you claim?"

### Hypotheses

**H1 — Metric validity (primary)**
On semi-synthetic EEG with controlled ground-truth state/subject variance ratio, the separability metric computed on frozen FM embeddings recovers the ratio monotonically.
- **Test**: 5 ratio regimes {0.1, 0.3, 1.0, 3.0, 10.0} of state-variance : subject-variance
- **Success**: Spearman r > 0.8 between ground-truth ratio and measured separability
- **Sanity baseline**: same metric on FOOOF features must have r > 0.95 (since signal is FOOOF-parameterized)

**H2 — FM inductive bias suppresses state (secondary, high-impact)**
Even at ground-truth ratio = 1.0 (equal state and subject variance), FM embeddings report separability < 0.5, indicating systematic compression of state signal.
- **Test**: paired comparison of FM separability vs ideal 0.5 at ratio=1.0
- **Success**: significant negative bias across ≥2 FMs

**H3 — Architectural differences (tertiary)**
Different FMs (LaBraM / CBraMod / EEGPT) show differing suppression curves, pointing to architecture-specific biases.
- **Output**: one figure panel per FM showing separability vs ground-truth ratio

### Design — semi-synthetic, not purely synthetic

**Why semi-synthetic**: Pure synthetic EEG may be too out-of-distribution for FMs; they may output noise. Semi-synthetic uses real subject variability + injected controlled state signal, keeping stimulus-domain realism.

**Protocol**:
1. Base corpus: LEMON (N≈200, 10 min eyes-closed rsEEG) or HBN healthy controls — high subject diversity, minimal task state
2. Segment each recording into 10 × 1-minute epochs
3. Randomly assign 5 epochs as "state 0", 5 as "state 1"
4. For state 1, multiply alpha-band (8–12 Hz) amplitude by factor k. For state 0, untouched
5. k ∈ {1.05, 1.1, 1.2, 1.5, 2.0} → calibrated to produce ratios {0.1, 0.3, 1.0, 3.0, 10.0}
   - Calibration: estimate per-subject alpha variance from base corpus, set k so injected state variance matches each target ratio
6. Extract frozen embeddings from LaBraM, CBraMod, EEGPT
7. Compute separability metric via nested ANOVA (subject, state nested in subject)
8. Baselines: (a) FOOOF parameter vector, (b) raw Welch PSD, (c) random projection

**Output**: Figure 1 — one panel per FM + baselines, x=ground-truth ratio, y=measured separability, diagonal line = ideal.

### Failure modes
- H1 fails on FOOOF baseline → metric formulation is broken; redesign required
- H1 holds on FOOOF but r < 0.5 on FMs → FMs scramble signal; repositions paper as "FMs are unreliable variance-decomposition substrates"
- H2 fails (no systematic bias) → core motivation for the paper weakens

### Time / resources
2–3 weeks. Frozen GPU pass only. LEMON access is open (OpenNeuro ds000221).

---

## 4. Experiment β — Aperiodic Ablation Causal Test

### Motivation
Demuru 2020 established correlation: aperiodic 1/f parameters allow subject identification. No intervention study has tested whether aperiodic removal causally destroys subject-decodability in FM embeddings, nor whether it liberates state signal. This is the mechanism-paper centerpiece.

### Hypotheses (four-link causal chain)

**H1 — Aperiodic ablation kills subject decodability (primary)**
After removing the FOOOF aperiodic fit from EEG and re-extracting FM features, subject-ID probe accuracy drops substantially.
- **Success**: ≥20 percentage-point drop in subject probe bAcc (e.g., 92% → 70%)

**H2 — Aperiodic ablation liberates state signal (secondary, "money shot")**
State-label probe accuracy increases or stays flat after aperiodic removal — demonstrating aperiodic was *masking* state signal.
- **Success**: no drop in state bAcc; ≥5-pp increase on at least one dataset

**H3 — Periodic ablation is not equivalent (control)**
Removing periodic peaks instead of aperiodic should NOT significantly reduce subject decodability.
- **Success**: subject-ID bAcc drops < 5 pp under periodic-only removal

**H4 — Swap intervention (strongest causal evidence)**
Grafting subject A's aperiodic component onto subject B's periodic + residual signal causes FM to predict "A" rather than "B".
- **Success**: FM subject-ID prediction follows aperiodic donor >80% of the time

### Design

**Signal processing pipeline** (must be validated carefully — easy to introduce artifacts):

Per 10-second epoch, per channel:
```
1. Welch PSD → FOOOF fit → parameters (offset b, slope χ, peaks [(f_i, A_i, BW_i)])
2. Aperiodic-removed:
     X(f) = FFT(x)
     aper_power(f) = 10^(b - χ · log10(f))
     X_flat(f) = X(f) / sqrt(aper_power(f))
     x_ablated = IFFT(X_flat)
3. Periodic-removed (control):
     Subtract gaussian peaks in PSD domain; reconstruct with preserved phase
4. Swap (H4):
     For pair (A, B): X_swap(f) = X_B(f) / sqrt(aper_B(f)) · sqrt(aper_A(f))
     → applies A's aperiodic envelope to B's content
```

**Dataset selection** (cover trait → state spectrum):
- ADFTD (ds004504, RS-only — keep as trait-leaning)
- EEGMAT (rest vs mental arithmetic — mid-spectrum state)
- SleepEDF (wake vs NREM — strong state)
- Optional: TDBRAIN or MODMA

**ML protocol**:
- 4 conditions × 3 FMs (LaBraM, CBraMod, EEGPT) × 3 datasets × 2 probes (subject-ID, state-label) = 72 runs
- Subject-ID probe: subject-internal CV (leave-one-session-out if available, else stratified k-fold on epochs)
- State-label probe: leave-subjects-out CV

**Output**: Figure 2 — grouped bars, ΔbAcc vs Original for (subject-ID, state-label) across conditions and datasets. Table 1 — Swap accuracy by (A-aperiodic, B-content) pairs.

### Failure modes
- H1 fails → Demuru 2020 doesn't replicate under this setup; check FOOOF fits, channel-wise aperiodic validity, and whether FM relies more on phase than power
- H2 fails (state bAcc also drops) → aperiodic change co-modulates with state (e.g., alpha-decrease actually = 1/f-steepening); paper narrative weakens but doesn't die
- H3 fails (periodic removal also kills subject ID) → subject info is distributed, not aperiodic-specific; need finer decomposition (per-band)
- H4 fails → causal chain not as clean; paper retreats to correlational level

### Known technical traps
- FOOOF fit unreliable at < 2 Hz and > 40 Hz — restrict fit range, report failure rate
- Some channels/recordings will fail FOOOF — report failure rate openly, do not silently drop
- Flattened signal is out-of-distribution for FM — report embedding norm / variance change, pre-empt "you just broke the FM" critique
- Random seed sensitivity in FOOOF — fix seeds, report ±

### Time / resources
3–4 weeks. Majority spent validating signal processing correctness. Frozen GPU only.

---

## 5. Experiment δ — Finn Identifiability on EEG FM Embeddings

### Motivation
Finn 2015 (Nat Neurosci, 3000+ citations) established connectome fingerprinting as the canonical individual-specificity metric on fMRI-FC (92.9% identification rate). No paper has applied this standard metric to EEG foundation model embeddings on test-retest clinical rsEEG. Reporting "EEG FM embeddings achieve ~95% Finn identifiability — higher than fMRI — which directly predicts their cross-subject clinical failure" is the abstract-friendly killer result.

### Hypotheses

**H1 — FM embeddings are strong brain fingerprints (primary)**
Test-retest identifiability rate on EEG FM embeddings ≥ 80%, plausibly ≥ Finn's 92.9% on fMRI-FC.
- **Success**: nearest-neighbor identification rate ≥ 80% across ≥2 datasets

**H2 — FM pretraining amplifies subject signature vs raw features (key comparison)**
FM embedding identifiability is significantly higher than FOOOF or raw PSD feature identifiability.
- **Success**: paired bootstrap, FM > FOOOF, p < 0.01

**H3 — Identifiability predicts state-classification failure (mechanism link)**
Across (FM × dataset) pairs, identifiability rate negatively correlates with published/measured state classification accuracy.
- **Success**: Spearman r < −0.5

### Design

**Public test-retest rsEEG datasets**:

| Dataset | Size with repeat sessions | Interval | Access |
|---|---|---|---|
| HBN (Healthy Brain Network) | ~500 multi-session | days–weeks | Apply through CMI, ~1 week |
| LEMON (MPI Leipzig) | ~60 follow-up (of 227) | days | OpenNeuro ds000221 direct |
| TDBRAIN | ~200 | ~year | Already locally available |
| MIPDB | multi-session | ~month | Apply |
| EEG-IIT test-retest | designed for reliability | short | Apply |

**Bootstrap plan**: start on TDBRAIN (zero access delay) to validate pipeline, then scale to HBN/LEMON.

**Metric computation**:
```
For subject s with session-1 embedding z_{s,1} and session-2 embedding z_{s,2}:
    I_self(s)    = corr(z_{s,1}, z_{s,2})
    I_others(s)  = mean over s' != s of corr(z_{s,1}, z_{s',2})
    DI           = mean_s [I_self(s) − I_others(s)]               (Amico & Goñi 2018)
    IR           = fraction of s where argmax_{s'} corr(z_{s,1}, z_{s',2}) == s
```

Direction-symmetric: also compute session-2→session-1 and average.

**Baselines (graduated)**:
- Random embedding (chance = 1/N)
- Raw PSD mean vector
- FOOOF aperiodic-only
- FOOOF periodic-only
- Older FMs (BIOT, BENDR) — expect lower identifiability
- Modern FMs (LaBraM, CBraMod, EEGPT, REVE) — expect highest

**Cross-condition extension (Finn's strongest finding)**:
If HBN provides eyes-open + eyes-closed rsEEG or other state conditions, report cross-condition identifiability. A high cross-condition rate means FM fingerprint is state-invariant — a trait-level signal — which *directly explains* state-classification failure.

**Output**: Figure 3 — identifiability rate across FMs and baselines (bars, per dataset). Figure 4 — scatter of identifiability rate vs state classification accuracy; annotated with H3 regression.

### Failure modes
- H1 fails (IR < 50%) → FM embeddings are session-noise-sensitive; repositions paper to explore "what does FM embedding actually encode"
- H2 fails (FM ≤ FOOOF) → paper claim weakens; FM merely replicates known aperiodic fingerprint
- H3 fails (no correlation) → mechanism link not established; paper retreats to H1+H2 observations only

### Technical traps
- Session-to-session impedance/montage differences → per-session z-score or rank-based correlation
- Session gap varies → report identifiability as function of gap
- HBN child population has developmental drift → age-matched subsample or explicit control

### Time / resources
2–3 weeks. Biggest time sink: HBN access (submit immediately). Frozen GPU only.

---

## 6. Storyline — why all three together

```
α  builds a trustworthy measurement tool       (methodological foundation)
      │
      ▼
β  uses it + causal intervention to find the pathogen  (mechanism)
      │
      ▼
δ  uses the canonical neuroscience metric to quantify  (clinical severity)
   how bad the disease is in real test-retest data
```

**Single-experiment weaknesses**:
- α alone → methods-journal paper, low impact
- β alone → mechanism paper, narrow scope
- δ alone → "Finn transplanted to EEG", observational

**Three together**: "We built a validated ruler (α), used it with causal intervention to identify the pathogen (β), then measured the severity in canonical terms (δ)." This is a coherent diagnostic paper.

## 7. Timeline

```
Week 1–2   Data access applications in flight (HBN, MIPDB, LEMON)
            Bootstrap Finn pipeline on TDBRAIN
Week 3–4   Experiment α (semi-synthetic + separability calibration)
Week 5–7   Experiment β (FOOOF ablation + swap)
Week 8–9   Experiment δ full run (once HBN access lands)
Week 10    Cross-experiment integration
            (β's aperiodic-removal effect → α validation /
             β's subject bAcc → δ identifiability correlation)
Week 11–12 Writing
```

**Critical path**: HBN access must be submitted Week 1 or δ delays the whole paper.

## 8. Ground-rules (from today's literature work)

- Every citation in the final draft must be re-verified from PDF or stable URL; do not cite from memory or earlier docs (reinforced feedback rule `feedback_verify_benchmark_numbers.md`).
- LEAD Table 3 RS-only 55.49% ± 15.08% is the correct reference point for ADFTD-RS-only results, not LEAD Table 2 91.14% (which is merged RS+PS).
- Do not reuse any prior FT results from before 2026-04-19 without rerunning under locked HP (per-FM recipe + bs=128 + SWA). Frozen/LP/variance/probe results from before 2026-04-19 survive only if input norm was correct per-FM.

## 9. Verified literature anchors for citations

Core citations for the paper (all URL-verified 2026-04-19):

**EEG FMs & critiques**
- LaBraM (Jiang et al., ICLR 2024): https://arxiv.org/abs/2405.18765
- CBraMod (Wang et al., ICLR 2025): https://arxiv.org/abs/2412.07236
- EEGPT (Wang et al., NeurIPS 2024): https://openreview.net/forum?id=lvS2b8CjG5
- LEAD (Wang et al., arXiv v4 Feb 2026): https://arxiv.org/abs/2502.01678
- Are EEG FMs Worth It? (Yang et al., ICLR 2026): https://openreview.net/forum?id=5Xwm8e6vbh
- EEG-FM-Bench (Xu et al., 2025/2026): https://arxiv.org/abs/2508.17742
- EEG FMs: Critical Review (2025): https://arxiv.org/abs/2507.11783
- EEG-FM Progresses & Benchmarking (Jan 2026): https://arxiv.org/abs/2601.17883
- Brain4FMs (Feb 2026): https://arxiv.org/abs/2602.11558

**Subject dominance & identifiability**
- Gibson et al. 2022 NeuroImage (variance partitioning, raw EEG): https://www.sciencedirect.com/science/article/pii/S105381192200163X
- Brookshire et al. 2024 Frontiers (data leakage in translational EEG): https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1373515/full
- Finn et al. 2015 Nat Neurosci (connectome fingerprint): https://pubmed.ncbi.nlm.nih.gov/26457551/
- Amico & Goñi 2018 Sci Rep (differential identifiability): https://www.nature.com/articles/s41598-018-25089-1
- GC-VASE (Mishra et al. 2025): https://arxiv.org/abs/2501.16626
- Özdenizci et al. 2020 (adversarial subject removal): https://pmc.ncbi.nlm.nih.gov/articles/PMC7971154/
- Zhang & Liu 2025 (MI-based subject disentanglement): https://arxiv.org/abs/2501.08693

**FOOOF & aperiodic fingerprint**
- Donoghue et al. 2020 Nat Neurosci (FOOOF): https://www.nature.com/articles/s41593-020-00744-x
- Gao et al. 2017 NeuroImage (E/I balance from field potentials): https://pubmed.ncbi.nlm.nih.gov/28676297/
- Waschke et al. 2021 Neuron (neural variability & behavior): https://www.cell.com/neuron/fulltext/S0896-6273(21)00045-3
- Demuru & Fraschini 2020 (aperiodic EEG fingerprint): https://arxiv.org/abs/2001.09424
- Lanzone et al. 2023 (MEG aperiodic fingerprint replication): https://www.sciencedirect.com/science/article/pii/S1053811923004111

## 10. Open action items

- [ ] Submit HBN data access application (Week 1 — blocking for δ)
- [ ] Submit MIPDB / LEMON applications in parallel
- [ ] Read LEAD v4 PDF §3 + Appendix B to extract exact quote on RS+PS merging for Related Work
- [ ] Audit `results/features_cache/` coverage matrix (FM × dataset) against HP-lockdown spec; document which cached features are still valid under locked per-FM norms
- [ ] Decide: extend to NeurIPS-scale path (add experiments γ — pretraining distribution intervention, ε — within-subject multi-state meta-dataset) or ship minimum viable to TNSRE first

## 11. What this document is NOT

- Not a decision. The user has flagged direction interest but the three-experiment commitment is pending explicit approval.
- Not exhaustive. Experiments γ and ε (pretraining intervention, multi-state dataset) from the full plan are deliberately excluded here; see discussion transcript 2026-04-19 for those.
- Not HP-decontaminated. FT-based findings from before 2026-04-19 remain in other docs unchanged; this plan is specifically chosen to avoid depending on them.
