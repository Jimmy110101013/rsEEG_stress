# SDL Paper Draft — §6 Representation-Level Evidence of Subject Dominance (v2, 2026-04-20)

**Supersedes**: `sdl_paper_draft_section_6.md` (2026-04-15).
**Key revisions**: Original §6 framed a "paired EEGMAT vs Stress contrast-strength test" pivoting on "FT rescue only in strong-contrast cases". Under the matched-protocol LP (§5.2), the EEGMAT "FT rescue" no longer holds as originally described — LaBraM FT = LP on EEGMAT (Δ = −0.5 pp). §6 v2 therefore reframes: the paired comparison becomes about **LP state separability** (what the frozen representation already contains), not FT gain (what fine-tuning adds). A new §6.5 reports the FOOOF-based causal ablation that identifies aperiodic 1/f as the primary substrate of subject dominance.

---

## §6 Representation-Level Evidence of Subject Dominance

### 6.1 Experimental design

We compare frozen foundation-model representations on two within-subject designs. Both use subject-level cross-validation and the same three FMs (LaBraM, CBraMod, REVE); both are evaluated under the matched per-window LP protocol introduced in §5.2 with 8 training seeds per cell. The two arms differ only in the neural-contrast strength of the target label:

- **EEGMAT** (Zyma et al. 2019, 36 healthy subjects × 2 recordings): rest vs serial-subtraction mental arithmetic. The canonical neural signature of this task is posterior alpha desynchronisation [Klimesch 1999; Pfurtscheller & Aranibar 1979], a robust within-subject contrast reliably induced across participants.

- **UCSD Stress** (DASS-thresholded per-recording label, 17 subjects × 4 recordings): no published per-recording within-subject EEG correlate of DASS-21 or DSS [documented in `docs/related_work.md`]. The cohort was designed for group-level correlational analysis, not per-recording within-subject classification.

Under the SDL thesis the two designs should produce opposite outcomes in the frozen LP arm: EEGMAT should show separable state signal in frozen FM features (because the contrast is neurally real); UCSD Stress should not (because the contrast is absent or very weak).

### 6.2 Frozen LP state separability diverges sharply between arms

**Table 6.1.** Matched-protocol frozen LP BA under subject-level CV (8 seeds).

| Model | EEGMAT (rest vs arithmetic) | UCSD Stress (DASS per-recording) | ΔBA |
|---|:---:|:---:|:---:|
| LaBraM | 0.736 ± 0.025 | 0.525 ± 0.040 | +0.211 |
| CBraMod | 0.722 ± 0.022 | 0.430 ± 0.033 | +0.293 |
| REVE | 0.733 ± 0.021 | 0.441 ± 0.022 | +0.293 |

EEGMAT produces frozen LP accuracies of 0.72–0.74 across all three FMs; UCSD Stress produces 0.43–0.53. The gap averages ~27 pp across architectures. Because the FM weights and extraction protocol are identical across the two arms, the gap cannot be explained by FM capacity, HP choice, or evaluation protocol. It reflects a property of the underlying *task contrast*: EEGMAT's alpha-desynchronisation signal is large enough to be linearly separable in a frozen FM embedding; the per-recording DASS-thresholded Stress label is not.

### 6.3 Fine-tuning does not rescue Stress and barely moves EEGMAT

**Table 6.2.** Fine-tuning BA vs frozen LP BA, both under matched protocols.

| Dataset | Model | Frozen LP (8-seed) | Fine-tune (3-seed) | ΔFT − LP |
|---|---|---:|---:|---:|
| EEGMAT | LaBraM | 0.736 ± 0.025 | 0.731 ± 0.021 | −0.005 |
| EEGMAT | CBraMod | 0.722 ± 0.022 | 0.620 ± 0.058 | −0.097 |
| EEGMAT | REVE | 0.733 ± 0.021 | 0.727 ± 0.035 | −0.009 |
| Stress | LaBraM | 0.525 ± 0.040 | 0.443 ± 0.083 | −0.063 |

In 4 of 4 matched EEGMAT + Stress LaBraM cells (7 of 10 across the broader 4-dataset comparison reported in §8), fine-tuning fails to exceed frozen LP. On EEGMAT LaBraM — where the published §6.2 narrative previously claimed a "23 pp FT rescue" — LaBraM FT (0.731) is statistically indistinguishable from LaBraM frozen LP (0.736) under the matched protocol. **The signal separable on EEGMAT is already present in the frozen representation; fine-tuning does not produce it, and in fact on average slightly degrades it.** This reframes §6's take-home: the EEGMAT-vs-Stress contrast is not about whether fine-tuning rescues, but about whether the frozen representation already contains linearly separable state information.

On Stress, fine-tuning degrades below frozen LP by 6.3 pp on LaBraM. Combined with the null-failure permutation test of §5.3, this argues that Stress per-recording DASS lacks a within-subject neural signal reachable by any of the procedures we can apply.

### 6.4 Cross-architecture band consensus at the representation level

A complementary representation-level diagnostic — independent of classifier accuracy — is whether the three FMs *agree on which neural band their frozen representation most depends on*. We use causal band-stop ablation: a Butterworth filter removes one frequency band at a time from the input EEG; the cosine distance between the resulting frozen embedding and the unablated baseline measures per-band dependence.

This analysis is **representation-level rather than independent of FMs** — it uses FM embeddings to compute band dependence, not raw EEG. It is therefore another facet of FM-based evidence, not a neural-ground-truth anchor; we present it here because multiple FMs converging on the same band (or diverging) is itself an interpretable signal. Source: `results/studies/exp14_channel_importance/`.

On **EEGMAT**, LaBraM, REVE, and CBraMod all show their largest cosine-distance response to alpha-band removal (REVE strongest; alpha-cosine-distance 0.150, 3× the beta response). The three architectures *converge* on alpha — the Klimesch-predicted band.

On **UCSD Stress**, LaBraM peaks on beta, REVE on broadband, and CBraMod shows no band preference. The three architectures *diverge* on which band their representation relies on.

This pattern mirrors §6.2's frozen LP findings by a separate path. Convergence on alpha in EEGMAT is what a shared neural contrast would produce; divergence across bands on Stress is what an absent shared contrast would produce. §6.5 gives the strongest single form of this evidence: a causal ablation distinguishing aperiodic 1/f from periodic oscillations.

### 6.5 Aperiodic 1/f carries subject identity; periodic oscillations do not (causal ablation)

The cleanest mechanistic claim we can make about subject dominance is whether it is carried by (a) the 1/f aperiodic background, a physiological trait property shaped by cortical excitation/inhibition balance and skull/cap geometry [Donoghue 2020 Nat Neurosci; Gao 2017; Demuru 2020], or (b) periodic oscillatory peaks (alpha, beta), which are state-modulated [Klimesch 1999]. Prior work has shown aperiodic parameters identify individuals [Demuru & Fraschini 2020; Lanzone 2023 — MEG replication], but this evidence is correlational. We provide a causal test: surgically remove each component from the EEG input and measure what the frozen FM embedding can still decode.

**Method.** For each recording we fit FOOOF (Donoghue 2020) on channel-averaged PSD across all epochs (recording-level aperiodic fit, which achieves higher R² than per-epoch fits and matches the trait-level interpretation of 1/f as a stable property). Aperiodic model parameters (offset b, slope χ) and periodic peaks are used to reconstruct four signal variants per channel-epoch: *original*, *aperiodic-removed* (divide by aperiodic amplitude envelope in frequency domain, preserving phase), *periodic-removed* (subtract Gaussian peak power contributions, preserving phase), and *both-removed*. Each variant is fed through the frozen FM to produce per-window features (sources: `scripts/analysis/fooof_ablation.py`, `scripts/features/extract_fooof_ablated.py`).

FOOOF fit quality was high on both datasets: UCSD Stress R² mean = 0.895 (fit_success 100%, 74% with R² ≥ 0.9); EEGMAT R² mean = 0.980 (fit_success 100%, 99.5% with R² ≥ 0.9). Per-channel aperiodic slopes (χ) fall in physiologically plausible ranges: Stress mean 0.79 [IQR 0.26–1.29], EEGMAT mean 1.72 [IQR 1.39–2.00].

**Subject-identification probe**: multi-class logistic regression on per-window features, session-held-out split (half of each subject's recordings trained, other half tested), 8 seeds, recording-level prediction pooling. Reported as multi-class balanced accuracy; chance = 1/N_subjects (1/36 ≈ 0.028 for EEGMAT, 1/17 ≈ 0.059 for Stress).

**State-label probe**: binary logistic regression on per-window features, subject-disjoint CV, 8 seeds, prediction pooling. Reported on EEGMAT only (rest vs arithmetic, chance = 0.5); UCSD Stress DASS state-label probe omitted because §5 and §6.2 establish it as unlearnable under this protocol.

**Table 6.3.** FOOOF ablation effect on frozen FM embedding probes. Δ values in parentheses, relative to *Original*.

| Dataset | Model | Probe | Original | Aperiodic-removed | Periodic-removed | Both-removed |
|---|---|---|---:|---:|---:|---:|
| EEGMAT | LaBraM | Subject-ID | 0.556 ± 0.030 | 0.528 (−2.8) | 0.556 (0.0) | 0.497 (−5.9) |
| EEGMAT | CBraMod | Subject-ID | 0.507 ± 0.046 | **0.365 (−14.2)** | 0.507 (0.0) | 0.358 (−14.9) |
| EEGMAT | REVE | Subject-ID | 0.465 ± 0.071 | **0.205 (−26.0)** | 0.458 (−0.7) | 0.198 (−26.7) |
| EEGMAT | LaBraM | State | 0.722 ± 0.017 | 0.715 (−0.7) | 0.707 (−1.5) | 0.705 (−1.7) |
| EEGMAT | CBraMod | State | 0.707 ± 0.023 | 0.696 (−1.1) | 0.705 (−0.2) | 0.700 (−0.7) |
| EEGMAT | REVE | State | 0.712 ± 0.029 | 0.693 (−1.9) | 0.705 (−0.7) | 0.689 (−2.3) |
| Stress | LaBraM | Subject-ID | 0.569 ± 0.047 | 0.544 (−2.5) | 0.567 (−0.2) | 0.550 (−1.9) |
| Stress | CBraMod | Subject-ID | 0.569 ± 0.056 | 0.483 (−8.6) | 0.569 (0.0) | 0.483 (−8.6) |
| Stress | REVE | Subject-ID | 0.565 ± 0.077 | 0.569 (+0.4) | 0.567 (−0.2) | 0.557 (−0.8) |

**Three findings.**

- **Aperiodic ablation destroys subject-identity signal.** On EEGMAT, removing the 1/f aperiodic background drops CBraMod subject-ID probe BA by 14.2 pp (2.8× chance → 1.3× chance) and REVE by 26.0 pp (17× chance → 7× chance). LaBraM shows a smaller but same-direction effect (−2.8 pp); LaBraM's relative resilience is consistent with its z-scored input convention and vector-quantised tokeniser normalising aperiodic slope variation out of the representation. On Stress, CBraMod shows an equivalent drop (−8.6 pp); LaBraM and REVE effects are smaller, consistent with Stress's 17-subject session-held-out probe being noisier than EEGMAT's 36-subject equivalent.

- **Periodic ablation leaves subject-identity untouched.** Across all three FMs on both datasets, removing Gaussian peak power components changes subject-ID probe BA by at most 0.7 pp in magnitude. The contrast with aperiodic ablation establishes the dissociation: **aperiodic 1/f structure carries subject identity in the frozen FM embedding; periodic oscillations do not**.

- **State signal survives both ablations.** EEGMAT state-label probe BA drops by at most 2.3 pp under any ablation. The state signal is spectrally robust — not narrowly localised to a peak frequency that would be destroyed by periodic subtraction, and not dependent on aperiodic slope information. Alpha-ERD in mental arithmetic appears to be carried by phase-locked broadband modulation rather than by modulation of the aperiodic fit.

### 6.6 Putting §6 together

Three convergent lines of representation-level evidence on matched cohort data:

1. **Frozen LP separability** (§6.2): state signal is linearly extractable from frozen FM embedding on EEGMAT (BA 0.72–0.74) but not on Stress (BA 0.43–0.53).

2. **Cross-architecture band consensus** (§6.4): three FMs converge on alpha for EEGMAT (shared neural target) and diverge on Stress (no shared target).

3. **FOOOF causal ablation** (§6.5): aperiodic 1/f structure causally carries subject-ID decodability; periodic peaks do not; state signal is robust to both ablations. This mechanism is consistent across FMs, with LaBraM architecturally less dependent on aperiodic than CBraMod and REVE.

Taken together, §6 argues that the difference between "FM works" (EEGMAT) and "FM fails" (Stress) is determined by the *presence of within-recording neural contrast strong enough to be linearly separable in the frozen embedding* — not by FM scale, not by fine-tuning, not by pretraining corpus. Subject dominance is simultaneously (a) carried by aperiodic 1/f structure, (b) robust to periodic ablation, and (c) present across all three FMs to differing architectural degrees.

§7 asks the complementary architecture-independence question: once the Stress ceiling is in force, does any architecture — including from-scratch compact CNNs — escape it? §8 then examines fine-tuning dynamics across a broader 4-dataset × 3-FM grid to characterise when fine-tuning actually helps.

---

*Draft status: §6 v2 ~1,800 words. Includes new §6.5 FOOOF ablation mechanism sub-section. Table 6.2 (FT vs LP) and Table 6.3 (FOOOF ablation) replace the previous Table 6.1 (longitudinal DSS, which is now superfluous given the FT-vs-LP finding on EEGMAT).*
