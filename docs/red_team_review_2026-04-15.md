# Red-Team Review Synthesis — 2026-04-15

**Scope**: Synthesis of 4 independent reviewer agents (R1 adversarial methodologist, R2 neuroscience, R3 ML architecture, R4 literature scout) attacking the in-progress paper. Full individual reviews at `docs/review_R{1,2,3,4}_*.md`.

**Bottom line**: As currently written, R1 and R3 independently predict **major revision leaning reject** at IEEE TNSRE. The paper has **real, novel contributions** (R4 confirms no scoop on F-A or F-C) but its **execution and framing** have multiple overlapping weaknesses that a competent reviewer will catch. Three paths forward exist (retreat, arm-up, venue downshift). Decision required before more writing.

---

## 1. Convergent cross-reviewer concerns (any one alone fatal)

### Convergent-1. N-F21 LaBraM bug must be disclosed and canonical experiments re-run
**Raised by**: R1 (C5), R3 (C5, required control #2)
- **Fact**: LaBraM had `self.norm` architectural drift; fix applied 2026-04-15 but canonical exp03–08, 11, 12, 18, frozen/ft caches NOT re-run.
- **Authors' rationale**: Wang 4-seed window-level BA moved <1pp.
- **Reviewer counter**: recording-level BA moved −5.4pp on 12-seed mean, per-seed swings ±25pp. 3-seed means in F-C.2 (which has std 0.010) live INSIDE that per-seed noise envelope. A research-integrity reviewer will treat this as concealment.
- **Required action**: (a) re-run F-C.1 LaBraM ADFTD + TDBRAIN and F-C.2 LaBraM Stress with fixed architecture at ≥ 3 seeds; (b) report exp24 (fixed) as canonical with exp23 (old) in supplementary; (c) acknowledge in Methods.
- **Cost**: ~80 GPU-hr. Blocking for submission.

### Convergent-2. F-C "direction flip" is HP-selection artifact without single-HP robustness check
**Raised by**: R1 (F1), R3 (flagship weakness, required control #1)
- **Fact**: LaBraM best FT at (lr=1e-4, elrs=1.0); CBraMod at (1e-5, 0.1); REVE at (3e-5, 0.1). Each model's "direction" is read off ITS OWN winning cell. LaBraM at CBraMod's winning recipe = 0.443 ± 0.083 (erosion 5pp deeper, variance 8× higher).
- **Reviewer counter**: This is exactly the argmax-over-HP pattern Dodge 2019 "Show Your Work" and Musgrave 2020 warn against — papers the authors themselves cite approvingly in paper_strategy.md §6.
- **Required action**: pick ONE HP (ideally CBraMod/REVE's recipe) and re-run all 3 FMs × 2 datasets × 3 seeds. If direction survives at ≥ 2/3 HP recipes, F-C.1 is real. Otherwise retreat to "at its best HP, LaBraM injects on ADFTD and erodes on TDBRAIN" — a weaker single-model observation.
- **Cost**: ~60 GPU-hr for minimal 3-HP × 3-FM × 2-dataset × 3-seed sweep. Blocking for F-C claim.

### Convergent-3. 3-seed protocol is statistically underpowered for sub-1pp claims
**Raised by**: R1 (F1, C4), R3 (C5, C6)
- **Fact**: F-C.1 cells show Δ with stds of 0.28–3.35pp, computed from 3 seeds. Naive t-interval on n=3 (t_{2,.975}=4.30) gives: **3/6 cells have CIs that cross zero** (LaBraM ADFTD, REVE TDBRAIN, CBraMod ADFTD). CBraMod×ADFTD has σ > |μ|. Cohen's d > 2.5 at n=3 is arithmetic noise.
- **Reviewer counter**: paired permutation test on 3 seeds has min exact p = 0.125. N-F15's "all models p=0.25" is the authors' own admission that claims cannot clear any threshold.
- **Required action**: escalate to ≥ 8 seeds (matching Frozen LP protocol) on the 4 F-C.1 cells the paper wishes to defend (LaBraM ADFTD + TDBRAIN, REVE ADFTD + TDBRAIN). Drop CBraMod from F-C.1 headline until σ < |μ|/2.
- **Cost**: ~40 GPU-hr for 5 additional seeds × 4 cells.

### Convergent-4. Stress cannot be cited as evidence FOR F-C if §8 declares it a power floor
**Raised by**: R1 (F2), R3 (C7)
- **Fact**: paper_strategy.md §2 Pillar C counts F-C.1 (ADFTD/TDBRAIN) + F-C.2 (Stress) + F-C.3 (EEGMAT) as 3 independent triangulations. §8 declares Stress a power floor (F-D.1 LaBraM FT p=0.70 vs null). A dataset cannot be both evidence *for* a taxonomy and evidence *that the taxonomy can't be measured on it*.
- **Required action**: remove F-C.2 Stress row from F-C's triangulation claim. Cite it only in F-D as the power-floor case. F-C's dataset count drops from 3 to 2 (ADFTD + TDBRAIN), with EEGMAT as supporting evidence on a different metric. This is self-consistent but means the multi-dataset story is thinner.

### Convergent-5. "Contrast-strength" thesis is circular / unfalsifiable as currently written
**Raised by**: R1 (F3), R2 (verdict)
- **Fact**: The thesis claims contrast strength governs FM rescue, but contrast strength is defined/measured *by* FM outcome. EEGMAT succeeds → labeled "strong contrast"; Stress fails → labeled "weak contrast". Two data points with the outcome as its own predictor.
- **Required action**: define contrast strength *independently of FM outputs*. Two viable instruments:
  - **R1's proposal**: per-subject paired Hedges g on classical band-power between conditions, averaged over subjects. A number you can compute from raw EEG without any FM.
  - **R2's proposal**: published neural effect sizes (alpha-ERD for EEGMAT from Klimesch 1999; cluster-permutation PSD on Stress). Anchor on known neural signatures rather than deriving from our FM runs.
- Then show: contrast-strength metric **predicts** Δ(FT − frozen) across ≥ 3 datasets. Two-point thesis is not publishable as mechanism.
- **Cost**: trivial computation on existing data, but needs ≥ 2 additional datasets with measurable contrast to test monotonicity. ADFTD (AD vs HC is a strong neural contrast) and TDBRAIN (MDD vs HC, weaker clinical contrast) would fit.

---

## 2. Reviewer-specific concerns

### R1 (methodology) — additional blocking items
- **F-B Axis 2 "chance" framing is dubious** (C3): classical BA < 0.50 on 56/14 imbalance may be regression-to-majority, not signal absence. Authors must compare to stratified-dummy baseline and re-run classical with class-balanced loss. If class-balanced RF recovers to 0.55, Axis 2 collapses and "FMs beat classical" narrative weakens.
- **Bootstrap CI with n=3 seeds is malformed** (C2): CI width 0.041 narrower than sample std 0.032 is only possible if resampling something other than seeds (e.g. recordings within fold). Must clarify what is bootstrapped.
- **Multi-test burden uncorrected** (C1): 6 direction tests → Bonferroni α=0.0083. Under BH-FDR q=0.10, authors retain ~2 cells (LaBraM-TDBRAIN, REVE-ADFTD) — **same direction**, not opposite.
- **StratifiedGroupKFold(5) on 17 subjects** (C2): seed × split separation not reported. Reader cannot distinguish model variance from split variance.

### R2 (neuroscience) — blocking items
- **F-E alpha lateralization misinterprets Davidson/Coan FAA** (suspect claim #1): Frontal alpha asymmetry is a **L−R asymmetry** index, not absolute right-hemisphere alpha. Higher right-hemisphere alpha actually means **less right-hemisphere activity** (alpha ≈ cortical idling; Pfurtscheller 1999). The current F-E wording inverts the neural interpretation. Fix: reframe as "posterior alpha dominance consistent with resting-state occipital alpha" (non-committal) OR properly compute FAA (F3 alpha − F4 alpha) and cite Davidson framework correctly.
- **DASS-21 is past-week trait, not state** (suspect claim #2): DASS-21 is a 1-week recall screener. Per-recording classification is neurobiologically unvalidated — there is no published evidence that within-session EEG tracks DASS score at the recording level. Must flag this as a scope limitation in §3 Methods and §10 Limitations.
- **DSS within-subject correlate unpublished** (suspect claim #3): F-D.2 longitudinal failure may be "label has no neural ground truth" rather than "FMs can't detect the contrast." Alternative interpretation weakens the paired comparison with EEGMAT.
- **"Cross-model band convergence" has no precedent** (suspect claim #4): not an established signal/noise diagnostic in EEG interpretability literature. CBraMod always peaking at delta (NEURO.2 / NEURO.3) is a cherry-pick red flag — it could equally mean CBraMod is dominated by low-frequency artifacts or DC offset. Must either find precedent or frame as novel methodological proposal and defend separately.
- **Missing TNSRE-expected analyses**: group-level PSD contrast, ERD quantification, FAA features, microstates, source localization. R2 says 3.5 person-days of experiments would close major neuroscience objections.

### R3 (architecture) — additional blocking items
- **Scale mismatch not controlled** (C1): REVE 1.4B vs LaBraM/CBraMod 100M (14×). "Architecture" is not the controlled variable.
- **Pretraining-corpus similarity un-measured** (C2): LaBraM=TUH+, CBraMod=TUEG, REVE=BrainBridge. F-C's "model × dataset" may be "pretraining-corpus × downstream-dataset similarity" — a less architectural claim. Control: compute CKA/RSA between each FM's frozen embedding on held-out pretraining samples vs downstream datasets.
- **Window choice changes effects 2.7–5×**: ADFTD +5pp at 10s vs +1pp at 5s (A-F04 vs F-C.1); REVE Stress +12.8pp at 10s vs +4.8pp at 5s (F-C.4). Original F-C.2 REVE +8.3pp was 10s-FT + 5s-frozen **mismatch** — methodological error, not caveat.
- **Step-count confound**: 10s halves training windows. REVE's window-dependent effect is jointly architecture × window × SGD-steps. Control: equal-step budget comparison.
- **ShallowConvNet at 40k params matching 1.4B REVE** (C7): F-D.3 evidence is *corrosive* to F-C, not just a complement. If a 2017 CNN from scratch = 1.4B pretrained FM on Stress, what is "FT direction" even reading? Probably noise in the task head plus optimizer trajectory.

### R4 (literature) — additions required
- **Must cite EEG-Bench** (arXiv:2512.08959, NeurIPS 2025) — 4th independent benchmark converging on "simpler models competitive with FMs on clinical EEG."
- **Must cite "Sample Size Critically Shapes..."** (bioRxiv 2025.11.10.687610) — direct anchor for §8 power-floor narrative.
- **Must cite EEG Foundation Challenge 2025** (arXiv:2506.19141) — field-level endorsement that cross-subject psychopathology is *the* open problem.
- **Must cite Massive SFT Experiments** (arXiv:2506.14681) — LLM cross-modality precedent for F-C.
- **No direct scoop**: F-A and F-C are novel. EEG-Bench tempers F-B/F-D (not refuting, converging).
- **Gap**: no prior literature links DASS-21 or DSS to trial-level EEG features. Authors' "weak contrast" thesis is therefore scientifically well-grounded *and* an empirical first — but cannot claim prior evidence.

---

## 3. What survives unchallenged

All 4 reviewers agree these claims are strong:

- **F-A subject dominance** (R4: novel; R1: not directly attacked; R3: "strong and independent"; R2: fully consistent with EEG biometric literature Campisi & La Rocca 2014, Marcel & Millán 2007)
- **F-B Axis 1 (trial-vs-subject CV inflation)** (R1, R3, R4: consistent with Brookshire 2024 / EEG-Bench / field consensus)
- **F-D.3 architecture-independent Stress ceiling** (R1, R3: consistent with §8 power-floor narrative; R4: EEG-Bench supports "simpler models remain competitive")
- **EEGMAT alpha-desynchronization interpretation** (R2: textbook Klimesch 1999 / Pfurtscheller)
- **Paper novelty at April 2026** (R4: no scoop on F-A or F-C; EEG Foundation Challenge 2025 legitimizes the problem framing)

---

## 4. Three decision paths

### Path A — **Retreat and defend** (2–3 weeks)

**Premise**: accept R1/R3 criticisms of F-C as a taxonomy; retreat to defensible subset.

**Revised paper claims**:
- F-A (unchanged)
- F-B (Axis 1 only; drop Axis 2 or reframe as "FMs beat classical by ~15pp on Stress, gap width unstable and depends on classical baseline implementation")
- F-C → downgraded from "taxonomy" to "case study": **"At their respective best HPs, LaBraM injects on ADFTD and erodes on TDBRAIN; REVE shows the opposite pattern."** Qualitative single-model observations, not a 3-FM taxonomy.
- F-D (unchanged as power-floor case study)
- F-NEURO reframed as "cross-model band agreement on EEGMAT, divergence on Stress" — descriptive, not diagnostic.
- F-E fixed per R2 (remove Davidson FAA misinterpretation)

**Required work** (mandatory):
1. Re-run LaBraM F-C.1 cells with N-F21 fix (1 week, 80 GPU-hr)
2. Disclose N-F21 in methods + supplementary
3. Fix F-E alpha wording
4. Add DASS-21 validity caveat to §3 Methods + §10 Limitations
5. Update related_work.md per R4 (agent running now)
6. Correct classical baseline chance-vs-signal framing (F-B Axis 2)

**Pros**: fastest to submission; avoids overclaiming; R4 confirms novelty of F-A + F-B + F-D even without F-C taxonomy.
**Cons**: paper's headline contribution weakens — loses the "mechanism" story. Likely OK for IEEE TNSRE but probably not strong enough for NeurIPS D&B.
**Venue fit**: IEEE TNSRE viable; J. Neural Engineering viable; NeurIPS D&B probably not.

### Path B — **Arm up to defend F-C taxonomy** (2–3 months)

**Premise**: F-C is worth the fight. Run the controls R1 and R3 require.

**Required work** (mandatory, roughly prioritized):
1. **Convergent-1**: N-F21 LaBraM re-run (1 week, 80 GPU-hr) — blocking regardless
2. **Convergent-2**: single-HP robustness sweep — re-run F-C.1 at 3 HP recipes × 3 FMs × 2 datasets × 3 seeds = 54 runs (~60 GPU-hr, 1 week)
3. **Convergent-3**: escalate to 8 seeds on the 4 F-C.1 cells authors wish to defend (~40 GPU-hr, 1 week)
4. **Convergent-5**: pre-register a contrast-strength metric (R1's Hedges-g per-subject band-power OR R2's published effect sizes) and compute across ≥ 4 datasets. Show monotonic relationship.
5. **200-perm null on Stress** at best-HP for all 3 FMs (~90 GPU-hr, 1 week)
6. **Pretraining-corpus similarity control** (R3 C2) — compute CKA/RSA between FM frozen embeddings on held-out pretraining proxy vs downstream. If similarity predicts direction, retreat to "corpus-domain interaction."
7. **Scale control** (R3 C1) — add a 4th model at LaBraM/CBraMod scale OR a REVE-small variant. Expensive and may not be available.
8. **Equal-step budget REVE** (R3 C4) — re-run REVE 10s FT with truncated step budget.
9. **Neuroscience anchoring** (R2) — cluster-permutation PSD on Stress, alpha-ERD on EEGMAT, FAA; 3.5 person-days.

**Pros**: paper becomes a strong, reviewer-proof TNSRE or NeurIPS D&B submission. Novelty holds.
**Cons**: 2–3 months. Some controls (scale match) may be impossible without retraining a FM.
**Risk**: ~40% probability that single-HP sweep kills F-C.1 direction flip, and paper retreats anyway at month 2. Must be prepared to accept retreat verdict.

### Path C — **Venue downshift** (1–2 weeks, lowest effort)

**Premise**: reframe the paper as a workshop/TMLR-style "lessons learned" piece. Drop all mechanism claims.

**Positioning**: "I Can't Believe It's Not Better" NeurIPS workshop, TMLR, or similar. Frame as: *an honest audit of EEG FM benchmarks on a small clinical dataset, showing (a) Wang 2025 inflation, (b) subject-identity dominance, (c) architecture-independent ceiling, (d) limits of within-subject reframing.*

**Pros**: lowest risk, fastest, preserves all empirical findings.
**Cons**: much lower impact. Probably not sufficient for a TNSRE-bound thesis chapter.

---

## 5. Non-FM baseline inclusion recommendation

User's question: do EEGNet / EEG-Conformer / ShallowConvNet results belong in the paper?

**Recommendation**: **yes, but only ShallowConvNet, and only in F-D.3.**

Rationale:
- ShallowConvNet at 0.557 ± 0.031 inside the FM range (0.524–0.577) is the *single strongest piece of evidence* for F-D's architecture-independent ceiling. Without it, F-D is "FMs converge among themselves" (circular with F-C saying FMs diverge). With it, F-D is "every architecture we tried lands in the same 5pp band" (external anchor).
- EEGNet (0.518 ± 0.097) has σ wider than the difference between any two FMs — adds noise, not signal. Keep as supplementary footnote.
- EEG-Conformer not yet run multi-seed per TODO.md; inclusion would require additional work and is not needed if ShallowConvNet already makes the point.

**But**: R3's C7 observation is important — ShallowConvNet matching FMs is *corrosive* to F-C, not just supporting for F-D. The paper must either:
- (a) acknowledge the tension: "On Stress, FM architectures converge with classical deep CNNs — consistent with the F-D power-floor interpretation. F-C's direction claims therefore cannot be made on Stress and are restricted to ADFTD/TDBRAIN."
- (b) drop F-C.2 Stress row entirely from F-C's triangulation (matches Convergent-4 above).

Either is defensible; (b) is cleaner.

---

## 6. Proposed experiment priority list

Independent of Path A/B/C choice, these experiments are ordered by **marginal-benefit-per-GPU-hour** for defending the paper:

| # | Experiment | Addresses | Cost | Blocking? |
|---|---|---|---|---|
| 1 | LaBraM N-F21 re-run (F-C.1 ADFTD + TDBRAIN, F-C.2 Stress) | Convergent-1 | 80 GPU-hr | **Yes for any path that cites LaBraM FT numbers** |
| 2 | Fix F-E alpha wording (no experiment, paper edit) | R2 #1 | 30 min | Yes (neuroscience misinterpretation is a rejection risk) |
| 3 | Add DASS-21 validity caveat (paper edit) | R2 #2 | 30 min | Yes |
| 4 | Classical baseline class-balanced re-run | R1 C3 | 1 GPU-hr | **Yes for F-B Axis 2** |
| 5 | Single-HP robustness sweep for F-C.1 | Convergent-2 | 60 GPU-hr | Yes for Path B; optional for Path A |
| 6 | 8-seed escalation on F-C.1 surviving cells | Convergent-3 | 40 GPU-hr | Yes for Path B |
| 7 | Contrast-strength metric + monotone scatter | Convergent-5 | ~2 GPU-hr (analysis) | Yes for Path B; recommended for Path A |
| 8 | Group-level PSD + alpha-ERD + FAA on 70-rec | R2 missing analyses | ~4 person-days, no GPU | Recommended for Path A + B |
| 9 | 200-perm null on Stress at best HP, 3 FMs | Convergent-3 (Stress floor) | 90 GPU-hr | Recommended for Path A + B |
| 10 | Pretraining-corpus similarity control (CKA/RSA) | R3 C2 | ~8 GPU-hr | Recommended for Path B |
| 11 | Equal-step REVE 10s FT | R3 C4 | ~2 GPU-hr | Optional |
| 12 | Scale-matched 4th FM | R3 C1 | expensive | Optional (Path B stretch) |

**Minimum viable set for Path A**: 1, 2, 3, 4, 7, 8 (≈82 GPU-hr + 5 person-days).
**Minimum viable set for Path B**: 1, 2, 3, 4, 5, 6, 7, 8, 9 (≈315 GPU-hr + 5 person-days).

---

## 7. Immediate next steps

**Pending user decision on Path A / B / C.**

Once decided:
1. Update `paper_strategy.md` §2 and §3 to match chosen path's revised claims
2. Update `findings.md` (revert F-NEURO / F-D.2 framing if Path A; add monotone-scatter section if Path B)
3. Launch experiment queue per §6 priority table
4. Draft §10 Limitations to front-run the DASS-21 validity and scope caveats

**Independent of path**: related_work.md update is running (background agent); will cite EEG-Bench, Sample Size Shapes, EEG Foundation Challenge, Massive SFT, others.

---

## Appendix — Individual reviewer verdicts

- **R1 (methodology)**: *Major revision, leaning reject.* Requires 10+ seeds, 200-perm null, pre-registered contrast-strength metric, full N-F21 disclosure.
- **R2 (neuroscience)**: *Contrast-strength thesis is defensible in principle but circular as written.* Requires independent ground-truth anchors (cluster-permutation PSD on Stress, alpha-ERD on EEGMAT, Ko 2020 replication). 3.5 person-days of experiments would close major objections.
- **R3 (architecture)**: *F-C as written is not defensible for TNSRE or NeurIPS D&B.* Required controls: single-HP robustness, N-F21 canonical re-run, pretraining-corpus similarity. Pillars A, B, D are strong and independent.
- **R4 (literature)**: *Novelty holds up at April 2026.* F-A and F-C not scooped. Must cite 5 new papers. EEG-Bench (NeurIPS 2025) is the single most important adjacent paper.
