# Reviewer R2 Report — EEG / Cognitive Neuroscience

**Venue**: IEEE TNSRE
**Manuscript**: "Beyond Accuracy: What EEG Foundation Models Encode, and Why Fine-Tuning Direction Depends on Model × Dataset Interactions"
**Reviewer role**: Neurobiological plausibility, established EEG literature on resting-state, stress, cognitive load, and subject-invariant vs subject-specific features. Not a machine-learning reviewer.

---

## Summary judgment

The manuscript's *ML* story (subject dominance, FT taxonomy, honest evaluation) is
supported by its statistical machinery. The *neuroscience* story is thin, post-hoc,
and in several places **inconsistent with established EEG stress/affect literature**.
The authors lean on the word "neurobiologically" without producing a single source-
level, ERP-level, microstate-level, or connectivity-level analysis. The "contrast
strength" thesis is defensible as a *signal-to-noise* framing but is not a
neurobiological claim — the manuscript should stop labelling it as such. Several
specific claims (right-hemisphere alpha dominance, DSS as a within-subject neural
correlate, cross-model band-convergence as a signal/noise diagnostic) need
substantial rewording or are unsupported by the literature.

Recommendation: **Major revisions.** Details below.

---

## 1. Neurobiologically SUSPECT claims

### 1.1 Right-hemisphere alpha dominance = stress (F-E, §5 / §6.4)

**Claim**: RF importance ranks alpha power at right-hemisphere electrodes
(T4, C4, CP4, P4) and right-frontal (Fp2, F8) as the most discriminative
classical features for DASS-binary labels. The paper frames this as
"consistent with neuroscience literature."

**Problem**: The claim as stated is *backwards relative to the dominant model*,
and the paper obscures it.

- The Davidson / Coan & Allen frontal alpha asymmetry (FAA) model predicts
  **greater right-frontal alpha power = LESS right-hemisphere cortical
  activity** (alpha is inversely related to cortical activity). Under FAA,
  high-stress/withdrawal individuals show **relatively reduced LEFT frontal
  activity** (i.e., higher left-frontal alpha), not greater right-frontal
  alpha power (Coan & Allen 2004, *Biol Psychol* 67:7-50; Reznik & Allen
  2018 PMC6449497). A feature-importance result that ranks **right Fp2/F8
  alpha power** above left does not automatically map onto the
  withdrawal-motivation story.
- "Right-hemisphere alpha dominance" is *not* the stress signature in the
  canonical literature. The stress signature is **asymmetry** (L−R or log
  ratio), and RF does not receive an asymmetry feature — it receives raw
  per-channel band power, so channel-level importance does not speak
  directly to FAA at all.
- Temporo-parietal right-hemisphere alpha (T4, C4, CP4, P4) is a much
  weaker prior in the stress literature than frontal asymmetry. It is
  more commonly interpreted as attention/arousal-related (posterior alpha
  as cortical idling; Klimesch 1999 *Brain Res Rev* 29:169-195) than as
  stress-specific. §8.3 of `related_work.md` already concedes alpha
  asymmetry is trait-like rather than state-like in young adults (MDPI
  Symmetry 14(8), 2022) — but this caveat does not appear in the main text
  alongside F-E.

**Action required**:
1. Remove "consistent with neuroscience literature" unless the authors
   explicitly compute a left-minus-right FAA feature and show it is
   the dominant discriminator.
2. If they keep the channel-level importance, reframe as "posterior
   right-hemisphere alpha power co-varies with DASS class" and explicitly
   note this is **not** a FAA finding, only a hemispheric-alpha-power
   finding. Cite Klimesch 1999 rather than Davidson/Coan.
3. Acknowledge §8.3's trait-vs-state caveat in the main text, not just
   related_work.

**Supporting refs**:
- Coan & Allen 2004. "Frontal EEG asymmetry as a moderator and mediator of emotion." *Biol Psychol* 67(1-2):7-50. DOI: 10.1016/j.biopsycho.2004.03.002
- Reznik & Allen 2018. "Frontal asymmetry as a mediator and moderator of emotion: An updated review." *Psychophysiology* 55(1). PMC6449497.
- Klimesch 1999. "EEG alpha and theta oscillations reflect cognitive and memory performance: a review." *Brain Res Rev* 29(2-3):169-195. DOI: 10.1016/S0165-0173(98)00056-3
- Nusslock et al. 2018 (PMC5704985) — frontal asymmetry and reward anticipation; trait nature of FAA in young samples.

---

### 1.2 DASS-21 as an EEG-correlated state measure (§3 methods, F-D.3)

**Claim**: The paper operates under the premise that per-recording DASS-21
scores carry a recording-level neural state signature that frozen FMs or FT
FMs should be able to classify.

**Problem**: DASS-21 is a self-report trait/past-week screening instrument
(Lovibond & Lovibond 1995, *Behav Res Ther* 33:335–343). Its validation
timeframe is "the past week," not the 5–10 s EEG windows being classified.
The assumption that per-recording DASS → per-window neural state is
**neurobiologically unvalidated**. The only prior that comes close is the
dataset's own paper:

- **Ko et al. 2020** (*Assoc. among Emotional State, Sleep Quality, and
  Resting-State EEG Spectra*, PMID 32070988) report — on the same cohort —
  "a significant increase in resting-state spectral power density across
  theta and low-alpha bands associated with increased levels of anxiety
  and stress." **This is a between-subjects correlation at the study level**,
  not a demonstration that per-recording DASS is decodable at the single-
  recording level. The manuscript cites this paper as dataset source but
  does not use its *effect sizes* or *directionality* to set expectations.

**Action required**:
1. Add a Methods-level statement that DASS-21 is a past-week self-report
   and that per-recording classification is an *approximation* with
   unknown neural latency/stability.
2. Cite Ko et al. 2020 directly and state what neural features they
   reported (theta + low-alpha increase with stress/anxiety). Explicitly
   compare to F-E (which favors mid/high-alpha posterior right channels —
   **not** the same pattern).
3. The discrepancy between Ko et al.'s reported theta+low-alpha stress
   signature and F-E's posterior right-hemisphere alpha signature is
   scientifically interesting and should be discussed, not hidden.

**Supporting refs**:
- Lovibond & Lovibond 1995. "The structure of negative emotional states: Comparison of the DASS with the Beck Depression and Anxiety Inventories." *Behav Res Ther* 33(3):335-343. DOI: 10.1016/0005-7967(94)00075-U
- Ko, Komarov, Hairston, Jung, Lin 2020. "Associations Among Emotional State, Sleep Quality, and Resting-State EEG Spectra: A Longitudinal Study in Graduate Students." *IEEE TNSRE* 28(4):795-804. PMID 32070988.

---

### 1.3 DSS (Daily Stress Scale) as a within-subject EEG correlate (F-D.3)

**Claim**: F-D.3 interprets the LOO within-subject DSS classification failure
(all 3 FMs, all 3 classifiers, κ < 0) as evidence that "frozen FM features do
not capture within-subject stress variation."

**Problem**: Before blaming the FMs, the paper must establish that DSS **has**
a within-subject neural correlate detectable at the single-recording level.
The literature base is very thin:

- DSS (Daily Stress Inventory / Daily Stress Scale; e.g., Brantley & Jones
  1989, *J Personality Assessment*) is a daily-event enumeration instrument;
  its published validation is against cortisol and self-report, not
  single-session resting EEG.
- No study I could find directly validates per-day DSS scores against
  same-day resting-state EEG features. Even the dataset paper (Ko et al.
  2020) reports **between-subjects** theta/low-alpha associations, not
  within-subjects across recordings.
- Within-subject stress EEG literature that *does* exist (e.g., acute
  stressor paradigms — TSST, Stroop) typically reports frontal theta
  increases and alpha decreases during **acute task stress**, not during
  resting eyes-open EEG at varying daily DSS levels.

A failure of FMs to classify DSS within subjects is therefore **under-
determined**: it could mean (a) FMs lack the feature, (b) DSS has no
single-recording neural correlate, (c) the correlate exists but is below
detection at n ≤ 7 recordings/subject. The paper jumps to (a).

**Action required**:
1. State explicitly that DSS is *not* a validated within-subject EEG
   correlate. Frame F-D.3 as "we find no evidence of classifiable within-
   subject DSS signal; this does not distinguish FM failure from absence
   of ground-truth neural signature."
2. This actually *strengthens* the §8 power-floor narrative: F-D.3
   becomes a ground-truth-instrument limitation, not a model limitation.
3. Keep F-D.3 in the appendix as stated; do not promote it to main text.

---

### 1.4 "Cross-model band consensus" as a signal/noise diagnostic (§7, proposed F-NEURO)

**Claim**: On EEGMAT, LaBraM and REVE both peak at alpha in band-stop
ablation → "real neural signal." On Stress, LaBraM peaks at beta, REVE at
alpha → divergence → "noise floor."

**Problem**: This is a **methodological novelty being presented as an
established diagnostic**, which a TNSRE reviewer should flag. There is no
precedent in the EEG interpretability literature that I can find for using
cross-architecture band-importance convergence as a signal/noise test. The
actual interpretability literature (Nakagome et al. 2023 *Sci Rep* 13, on
DL EEG explanation methods; Cui et al. 2023 Frontiers; Hartmann et al.)
evaluates single-method consistency against simulated ground truth, not
cross-architecture consistency.

Specific problems:
1. Architectures with different inductive biases (LaBraM's neural tokenizer
   vs REVE's linear patch embedding) can legitimately attend to different
   bands even with identical *underlying neural signal*. A linear patch
   embedder is spectrally flat by construction; a VQ tokenizer imposes a
   discrete codebook that may concentrate on whichever band the pretraining
   data weighted most. Divergence → noise is not the only inference.
2. On EEGMAT, the alpha convergence is *expected by design* — mental
   arithmetic produces classical posterior alpha desynchronization
   (Klimesch 1999; Pfurtscheller & Lopes da Silva 1999). Any model that
   decodes EEGMAT *at all* will find alpha. Convergence on EEGMAT
   therefore has no discriminative power as a diagnostic — it is a
   boundary-condition check, not a "real signal" witness.
3. CBraMod always peaking at delta (noted in §7 parenthetical) is a
   **giant red flag** for the diagnostic: CBraMod's delta peak on EEGMAT
   would normally indicate artifact or architecture bias. If CBraMod is
   excluded from the cross-model test on EEGMAT but the authors still
   claim convergence, the test is cherry-picked.

**Action required**:
1. Either cite a precedent paper for cross-architecture band-importance
   convergence as a signal/noise diagnostic, or present it as a *proposed*
   heuristic with explicit caveats and call it such.
2. Acknowledge that alpha convergence on EEGMAT is the *a priori expected*
   outcome (alpha desynchronization during mental arithmetic is textbook),
   not a blind test of the diagnostic.
3. CBraMod's delta fixation must be addressed head-on; it cannot be a
   parenthetical note. If CBraMod is on the "noise" side for EEGMAT too,
   the 2-of-3 convergence is weaker than the text claims.

**Supporting refs**:
- Klimesch 1999 (above).
- Pfurtscheller & Lopes da Silva 1999. "Event-related EEG/MEG synchronization and desynchronization: basic principles." *Clin Neurophysiol* 110(11):1842-1857.
- Cui et al. 2023. "Towards best practice of interpreting deep learning models for EEG-based brain computer interfaces." *Front Comput Neurosci* 17:1232925.
- Nakagome et al. 2023. "An empirical comparison of deep learning explainability approaches for EEG using simulated ground truth." *Sci Rep* 13:43871.

---

## 2. Neurobiologically WELL-GROUNDED claims

### 2.1 Subject identity dominates frozen FM features (F-A)

**Supported by**: Decades of EEG biometrics literature establish that
resting-state EEG contains individual-specific signatures sufficient for
100% identification at ~100 subjects (La Rocca et al. 2014, *IEEE TIFS*
9(11):2406-2418; Campisi & La Rocca 2014, *IEEE SPM* 31(5):51-62). The
finding that foundation models trained on ~2,000 hours of mixed EEG encode
subject identity strongly is **expected**, and the manuscript's 71% ICC /
12/12 RSA dominance is consistent with the biometric literature's estimate
that resting-state EEG is individually unique within ~30 s of recording.

This is solid. The paper could *strengthen* this claim by explicitly
citing Campisi & La Rocca 2014 and Marcel & Millán 2007 in §5 as
pre-existing neurobiological expectations, rather than presenting it as
a surprising ML finding.

**Refs to add**:
- Campisi & La Rocca 2014. "Brain waves for automatic biometric-based user recognition." *IEEE Trans Inf Forensics Security* 9(5):782-800.
- Marcel & Millán 2007. "Person authentication using brainwaves (EEG) and maximum a posteriori model adaptation." *IEEE TPAMI* 29(4):743-752.
- La Rocca et al. 2014. "Human brain distinctiveness based on EEG spectral coherence connectivity." *IEEE TBME* 61(9):2406-2412.
- Maiorana & Campisi 2018. "Longitudinal evaluation of EEG-based biometric recognition." *IEEE TIFS* 13(5):1123-1138.

### 2.2 EEGMAT rest-vs-task alpha convergence (§7)

The observation that LaBraM + REVE peak at alpha for EEGMAT classification
is the textbook Klimesch/Pfurtscheller result. This is ground truth and
correctly used as a *positive control*. My objection in 1.4 is that it
cannot double as a *diagnostic for noise elsewhere*; it is fine as a
sanity check on the interpretability pipeline itself.

### 2.3 Young-adult stress ceiling (§8.1–8.3 of related_work)

The neural-efficiency + allostatic-load explanations (Neubauer & Fink 2009;
McEwen 2010) are legitimate neurobiological framings for why this cohort
(Taiwan graduate students) may have attenuated within-subject resting-
state stress signatures. This content currently lives in related_work and
should be promoted to the Discussion as one plausible mechanism for the
power-floor observation.

---

## 3. "Contrast strength" thesis — is it defensible?

**Claim** (§6.4, §8): "Within-subject contrast strength (not just
design type) determines whether within-subject framing rescues FM."
EEGMAT has strong contrast (rest vs arithmetic → alpha desync), Stress
has weak contrast (day-to-day DASS variation), therefore FT works on
EEGMAT and fails on Stress.

**Verdict**: *Partially* defensible. Reframe required.

1. As a **signal-to-noise** argument it is fine. EEGMAT's
   rest-vs-arithmetic contrast has a decades-long literature of large
   effect sizes at the group level (alpha ERD Cohen's d typically
   0.8–1.5; theta Fm increase d ~0.6–1.0). Per-recording DASS variation
   in this cohort has no equivalent published effect size — Ko et al.
   2020 report only between-subject associations, not within-subject
   per-recording correlates.
2. As a **neurobiological** argument it is under-specified. "Contrast
   strength" needs to be operationalized. Without a per-dataset estimate
   of the ground-truth neural effect size (e.g., cluster-based permutation
   ERD on EEGMAT vs DASS split on Stress at group level), the thesis is
   circular: "FT fails on Stress because contrast is weak; we know
   contrast is weak because FT fails."
3. A proper neurobiological version would say: "EEGMAT's paradigm has a
   published per-subject alpha-ERD effect of Cohen's d ≈ X; Stress has
   no established single-recording neural correlate of DASS score; the
   FM ceiling tracks this gradient." That requires a baseline group-level
   analysis the authors have not reported.

**Action**: Rename the thesis from "contrast strength" to **"ground-
truth neural effect size"** or **"published-literature neural contrast"**,
and back it with citations to expected effect sizes for each paradigm
rather than with circular FM-performance numbers.

---

## 4. UCSD Stress dataset (Komarov/Ko/Jung 2020 TNSRE 28(4):795) — design fit

From PubMed and the abstract (PMID 32070988, Ko et al.):

- 18 graduate students, ~94 combined DASS+EEG sessions, longitudinal
  across an academic semester.
- **Stated scientific goal**: identify associations between emotional
  state (DASS dimensions), sleep quality, and resting-state EEG spectra
  at the **population level across the semester**.
- The paper's reported findings are **correlational spectral** — theta
  and low-alpha increases with anxiety/stress, high-frequency temporal
  activity with depression — at the **group level**, not
  single-recording classification.

**Implication**: The dataset was **not designed** as a single-recording
binary classification benchmark. The Wang 2025 90.47% trial-level BA is
not only subject-leakage inflated (Axis 1 in F-B) but also pushes the
dataset well beyond its original design envelope. The manuscript should
state this in Methods or Limitations:

> "The Komarov et al. 2020 dataset was collected as a longitudinal
> group-level correlational study of DASS dimensions and resting-state
> spectra. Using it as a single-recording binary classification
> benchmark — as in Wang et al. 2025 and in our re-evaluation — extends
> it beyond its validated design; this partly motivates the power-floor
> framing in §8."

This is a **major missing caveat**. TNSRE reviewers who know the dataset
paper will notice.

**Known DASS-21 / DSS limitations as EEG-correlated measures**:
- DASS-21 is a past-week trait/state screener, not a session-level state probe.
- DSS is a daily-event count, not a neural-state instrument.
- Neither has published single-recording EEG validation.
- Both are Likert-scale subjective self-report → low test-retest precision
  at short timescales (ICCs ~0.6 at best for daily measures).

---

## 5. Missing neural evidence a TNSRE reviewer will expect

The paper is currently **100% representation-space / behavioral**. For a
TNSRE venue, this is sparse. The following neural-level analyses are
standard and their absence weakens the paper:

1. **ERP / ERD on EEGMAT** — per-subject alpha ERD during arithmetic
   versus rest, as a ground-truth contrast against which to compare FM
   band-importance results (§7). This is an afternoon's analysis with
   MNE-Python and would give the §7 convergence a real anchor.
2. **Group-level PSD contrast on Stress** — cluster-permutation test on
   high-DASS vs low-DASS recordings (cross-subject, not within-subject)
   replicating / failing to replicate Ko et al. 2020's theta+low-alpha
   report on the 70-recording subset. If the group-level signal is
   absent, §8 power-floor becomes airtight. If present, F-D.3 needs
   more work to explain why FMs miss it.
3. **Frontal alpha asymmetry (L-R log ratio)** — explicit FAA feature
   on Stress, compared to RF's raw channel-alpha importances. This
   directly tests the claim in 1.1 above.
4. **Microstates** — Khanna et al. 2015 (*Neurosci Biobehav Rev* 49:
   105-113) microstate framework applied to rest-state Stress data.
   Microstates A/B/C/D have documented stress/anxiety associations;
   their absence in this manuscript is notable for a TNSRE paper.
5. **Source localization (even minimum-norm / eLORETA)** — establishes
   whether "right-hemisphere" alpha in F-E is cortically plausible
   (e.g., parieto-temporal origin) or scalp-level volume-conducted.
6. **Subject-level test-retest of FM features** — complementing biometric
   story: correlate FM feature test-retest ICC with subject ICC in
   handcrafted spectra. If FMs have *higher* subject ICC than classical
   features, that mechanistically explains F-A.

**Minimum ask**: (1), (2), (3). (4)–(6) are nice-to-have but (1)–(3)
are close to required for TNSRE.

---

## 6. Rest-vs-task vs stress contrast — is weak-contrast biologically principled?

**Neural signatures DASS-state SHOULD differ on, according to the literature**:
- Theta / low-alpha band power (Ko et al. 2020, same cohort).
- Frontal midline theta (Fz, FCz) — associated with sustained anxiety
  and worry (Cavanagh & Shackman 2015, *J Physiol Paris* 109:3-15).
- Frontal alpha asymmetry (Coan & Allen 2004) — trait-level in young
  adults, weaker state effects.
- Default-mode-network vs task-positive-network balance as seen via
  alpha-beta ratio at midline posterior electrodes.
- Beta/gamma power in limbic-projecting cortex (anterior temporal) for
  chronic stress — weak EEG evidence, mainly from MEG/fMRI.

**Expected group-level effect sizes**: Small. Cohen's d ≈ 0.2–0.5 for
DASS / PSS vs resting spectra in most published cohorts. Contrast this
with EEGMAT rest-vs-task (d ≈ 0.8–1.5 for alpha ERD).

**Verdict**: The "weak contrast" framing is **biologically principled**
if operationalized via published effect sizes; it is **post-hoc** as
written (circular reasoning from FM performance). See §3 above.

---

## 7. Proposed neural-level experiments to strengthen the paper

Ranked by ROI (revision budget vs reviewer satisfaction):

| # | Experiment | Cost | Value |
|---|---|---|---|
| 1 | Group-level cluster-permutation PSD contrast on Stress (high-DASS vs low-DASS, cross-subject) | 1 day | Very high — validates or refutes ground-truth signal presence, makes §8 airtight |
| 2 | Alpha ERD analysis on EEGMAT (rest vs arithmetic, Cohen's d per subject) | 1 day | Very high — anchors §7 band-importance convergence in textbook neuroscience |
| 3 | Explicit FAA (L−R log ratio) feature on Stress, comparison to raw channel-alpha RF importance | 0.5 day | High — resolves 1.1 cleanly |
| 4 | Replicate Ko et al. 2020's theta+low-alpha finding on the 70-recording DASS+DSS subset | 1 day | High — connects the classical analysis in F-E to the dataset's original paper |
| 5 | Microstates (Khanna 2015) on Stress, correlate microstate parameters with DASS | 2-3 days | Medium-high — TNSRE-favored analysis, optional |
| 6 | eLORETA source localization of F-E's right-hemisphere alpha cluster | 1-2 days | Medium — confirms cortical origin |
| 7 | FM feature ICC vs classical-feature ICC | 0.5 day | Medium — mechanistic completion of F-A |

Items 1–4 together are ~3.5 person-days and would close **all** of the
major neuroscience objections in this report.

---

## 8. Verdict on the "contrast strength" thesis

**Neurobiologically defensible in principle; not yet demonstrated as written.**

- The *general claim* — that FM FT gains track published per-paradigm
  neural effect size — is sound and consistent with the biometrics /
  cognitive-load literature.
- The *specific operationalization* in this manuscript is circular: FM
  performance is used as both predictor and evidence of contrast
  strength. The thesis needs an independent ground-truth measure
  (experiments #1, #2, #4 above).
- The thesis also needs to be renamed. "Contrast strength" is vague
  and jargon-free in a way that invites post-hoc stretching. "Published
  neural effect size" or "paradigm-validated ground-truth contrast"
  is less graceful but harder to misuse.
- Once anchored, the thesis *does* do useful work: it predicts that
  FM FT will fail on affect/stress paradigms without published strong
  single-recording neural correlates, regardless of sample size — this
  is consistent with Brain4FMs / AdaBrain-Bench (N-F11) and is a
  publishable insight if properly grounded.

**Recommendation to the authors**: Complete experiments #1, #2, and #4
from §7. If #1 returns a null group-level result on Stress, the paper
becomes much stronger (§8 power floor gains ground-truth support). If
#1 returns a positive group-level result, F-D and F-D.3 need
substantial revision (FMs miss a signal that classical group statistics
detect). Either outcome is paper-worthy; not running the experiment is
not.

---

## Sources

- [Reznik & Allen 2018 — FAA updated primer](https://pmc.ncbi.nlm.nih.gov/articles/PMC6449497/)
- [Nature Sci Rep — Microstates-based resting FAA / approach-withdrawal](https://www.nature.com/articles/s41598-020-61119-7)
- [FAA and self-report stress / DASS relationship review](https://www.tandfonline.com/doi/full/10.1080/23279095.2024.2425361)
- [FAA and chronic stress / depression, not somatoform](https://www.sciencedirect.com/science/article/abs/pii/S0167876024000461)
- [Ko et al. 2020 — Associations among Emotional State, Sleep Quality, and Resting-State EEG Spectra (dataset paper, PMID 32070988)](https://pubmed.ncbi.nlm.nih.gov/32070988/)
- [La Rocca et al. 2014 — Stable EEG features for biometric recognition in resting state](https://link.springer.com/chapter/10.1007/978-3-662-44485-6_22)
- [Time-robustness of individual identification based on resting-state EEG](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.672946/full)
- [EEG biometrics — Challenges and future perspectives](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2018.00066/full)
- [Klimesch 1999 (PsycNet record) — EEG alpha/theta and cognitive performance](https://psycnet.apa.org/record/1999-13868-001)
- [Nakagome et al. 2023 — Empirical comparison of DL explainability approaches for EEG](https://www.nature.com/articles/s41598-023-43871-8)
- [Cui et al. 2023 — Best practice of interpreting DL models for EEG BCI](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2023.1232925/full)
