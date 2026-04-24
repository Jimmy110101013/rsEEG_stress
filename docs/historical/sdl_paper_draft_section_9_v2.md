# SDL Paper Draft — §9 A Pre-Flight Checklist for Clinical EEG-FM Deployment (v2, 2026-04-20)

**Role in SDL thesis**: *the operational output.* The preceding sections establish (a) that evaluation-protocol corrections can collapse headline EEG-FM benchmarks by 20–40 pp (§5); (b) that frozen LP separability, not fine-tuning gain, is the primary diagnostic of whether an EEG FM will be useful on a new cohort (§6); (c) that aperiodic 1/f structure causally carries subject identity in FM embeddings (§6.5); (d) that cross-dataset fine-tuning shows 1 genuine rescue, 3 ties, and 6 degradations across 12 cells (§8). §9 translates these findings into a concrete, actionable decision rule for clinical researchers contemplating EEG-FM use on their own cohort.

---

## §9 A Pre-Flight Checklist for Clinical EEG-FM Deployment

### 9.1 Who this section is for

Three audiences:
1. **Clinical researchers** with a small-to-medium rsEEG cohort (20–500 subjects) who want to know whether an EEG foundation model will help on their data.
2. **Methodologists** reviewing EEG-FM papers and looking for concrete red-flags that indicate reported accuracies will not generalise.
3. **EEG-FM developers** choosing benchmarks against which to validate architectural choices.

All three questions reduce to the same core: under honest subject-disjoint evaluation, will this (FM × cohort × label) combination yield a clinically useful model, and if not, why not?

### 9.2 The checklist (three decisions)

Before investing compute in fine-tuning an EEG FM on a clinical rsEEG cohort, work through the three checks below in order. Each gate is designed to be cheap to run, producing a binary or ordinal signal that determines whether the next gate is worth passing through.

**Check 1 — Does the label have a published within-recording neural anchor?**

*Purpose*: Separates labels that have a known-detectable spectral or temporal signature from labels that are trait-level aggregations without direct per-recording neural correlate.

*How*: Search the neurophysiology literature for the task or diagnostic category. A qualifying anchor is a peer-reviewed effect at the epoch/recording level — e.g., alpha desynchronisation for mental arithmetic [Klimesch 1999], posterior alpha slowing for Alzheimer's [Babiloni 2020], beta-power reduction for Parkinson's [Cassani 2018]. A non-qualifying anchor is a group-level longitudinal correlation with no published per-recording EEG signature — e.g., DASS-21 score, Big Five personality factors, depression severity on long timescales.

*Expected outcome on our grid*:
- ADFTD (alpha slowing): ✓ qualifying anchor
- EEGMAT (rest vs arithmetic, alpha ERD): ✓ qualifying anchor
- TDBRAIN MDD (heterogeneous; weak within-recording anchor): ✗ non-qualifying
- Stress DASS per-recording: ✗ non-qualifying (no published neural correlate at the relevant temporal resolution)

*Decision*: If Check 1 fails, be prepared to accept classification ceilings of 0.50–0.58 BA under honest evaluation (our Stress finding). Don't invest in fine-tuning. Classical band-power baselines are likely within 5 pp of anything an FM will achieve on this label. Consider collecting additional structured tasks or choosing a label with a published anchor.

**Check 2 — What is the frozen linear probe balanced accuracy under subject-disjoint CV?**

*Purpose*: Measures whether the FM's frozen representation already contains linearly separable state signal for this label.

*How*: Extract frozen features per-window from the FM (using its per-FM input norm — this matters; see CLAUDE.md §2). Train logistic regression on per-window training-fold samples with class balancing, pool test-fold window predictions per recording, compute recording-level BA under `StratifiedGroupKFold(5)` with `groups=patient_id`. Repeat over at least 5 seeds. This takes < 1 hour on CPU once features are extracted; feature extraction takes < 1 hour on a single consumer GPU for 100–500-subject cohorts.

*Expected outcome on our grid*:
- EEGMAT: LP BA = 0.72–0.74 across three FMs (strong signal present)
- ADFTD: LP BA = 0.57–0.65 (moderate)
- TDBRAIN: LP BA = 0.52–0.75 (varies wildly by FM: LaBraM strong, CBraMod/REVE near chance)
- Stress: LP BA = 0.43–0.53 (below classical band-power baseline 0.553)

*Decision*: LP BA > 0.70 → the frozen representation already works; deploy frozen LP. LP BA in [0.55, 0.70] → there is some signal; fine-tuning may rescue (see Check 3). LP BA < 0.55 → no linearly separable state signal; do not fine-tune (you will either overfit or exploit subject shortcut).

**Check 3 — What is the subject-ID decodability of the frozen embedding, and how does it compare to state-label decodability?**

*Purpose*: Flags cases where apparent signal is actually subject-identity memorisation rather than state discrimination. This is the pre-flight diagnostic for the subject-dominance limit.

*How*: Train a multi-class logistic regression on per-window features to predict `patient_id`, using session-held-out split within each subject (half of each subject's recordings for training, other half for test). Pool per-window probabilities to recording level. Compute multi-class balanced accuracy; the chance level is 1/N_subjects. Compare to the state-label LP BA from Check 2.

*Interpretation*:
- If subject-ID BA is > 10× chance (e.g., 0.50 for N = 20 subjects, where chance = 0.05), the frozen FM embedding is a strong subject fingerprint. This is structural — §4 shows this is true for LaBraM/CBraMod/REVE on every dataset we tested. The informative number is the *ratio* of state-label BA to subject-ID decodability.
- State-label BA >> subject-ID BA (chance-adjusted) → state signal is well-separated from subject identity. FT has scope to add value.
- State-label BA ≈ subject-ID BA → state and subject are entangled. FT on this label risks exploiting subject shortcut rather than state signal.

*Supplementary diagnostic (FOOOF ablation, §6.5)*: For research-grade evaluation, verify that subject-ID BA drops substantially when aperiodic 1/f is removed from the input signal. A drop > 10 pp confirms the standard pattern (subject identity encoded in aperiodic component, periodic oscillations carry state). Absence of this drop indicates an atypical FM representation and warrants caution.

*Decision*: If state-label and subject-ID BAs are within 1.5× of each other, treat fine-tuning gains on between-subject labels as potentially spurious. Run a permutation-null control: fine-tune with shuffled labels. If shuffled-label BA is not significantly lower than real-label BA, your rescue is shortcut, not signal.

### 9.3 Applying the checklist to our datasets

| Dataset | Check 1 (Anchor) | Check 2 (Frozen LP BA) | Check 3 (Subj-ID vs state-label) | Verdict |
|---|---|---|---|---|
| **EEGMAT** (rest/task) | ✓ Alpha ERD | 0.72–0.74 | State BA > Subj-ID BA | **Deploy frozen LP; FT does not add value but does not harm** |
| **ADFTD** (AD vs HC) | ✓ Alpha slowing | 0.57–0.65 | Comparable | **FT on LaBraM only (vector-quantised tokenizer has plasticity here); frozen LP on others** |
| **TDBRAIN** (MDD vs HC) | ✗ No published anchor | 0.52–0.75 (LaBraM-only) | Not checked | **Frozen LP on LaBraM; do not fine-tune** |
| **Stress** (DASS per-rec) | ✗ No published anchor | 0.43–0.53 | **Subject-ID ≫ state-label** | **Do not fine-tune; do not expect clinical utility at this cohort size with this label** |

### 9.4 What this changes for the field

The checklist operationalises two findings that the EEG-FM benchmark literature has not yet internalised:

- **Frozen LP is a diagnostic, not a weakness.** Most EEG-FM papers report LP as a secondary baseline to fine-tuning, sometimes downplayed when FT wins. Under honest evaluation on small clinical cohorts, LP is often the more informative number: it reveals whether the FM's pretraining has given the downstream task any linear separability at all. If LP is high, LP is enough; if LP is low, fine-tuning is gambling.

- **Subject-ID probe is the pre-flight diagnostic for shortcut risk.** Check 3 adds a single sklearn pipeline to any EEG-FM evaluation that turns "our FT BA went up 12 pp" from a celebration into a diagnostic signal: did the 12 pp come from state separation or from subject-fingerprint shortcut? With N ≤ 100 subjects this question is non-trivial and easy to get wrong.

We encourage future EEG-FM papers reporting cross-subject clinical benchmarks to report Check 2 and Check 3 explicitly, alongside the primary fine-tuning numbers.

### 9.5 Limitations of the checklist

The checklist is a deployment-time diagnostic, not a research-time ceiling. It assumes:
- The cohort uses a standard rsEEG montage (≥ 19 channels, standard 10–20).
- Labels are binary or low-cardinality categorical.
- The research question is cross-subject prediction, not within-subject tracking.

For within-subject longitudinal tracking or within-session state detection tasks (e.g., drowsiness over an 8-hour shift), different considerations apply; the within-subject contrast literature [Klimesch 1999; Pfurtscheller 1997] is more directly applicable than any EEG-FM. Where subject identity can be controlled by design (same subject, multiple conditions), SDL does not bind.

For very large cohorts (N > 10^4 subjects, which exist in some industrial contexts), fine-tuning dynamics may differ, and the pattern reported here may not transfer.

---

*Draft status: §9 v2 ~1,500 words. Written for clinical-engineering audience (Path A framing). Ready for review.*
