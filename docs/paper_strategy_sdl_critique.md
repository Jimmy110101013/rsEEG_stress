# Paper Strategy v2 — SDL as Benchmark Critique (Type C)

**Date**: 2026-04-18 (v2 revision after exp_30 results + structure discussion)
**Status**: ACTIVE PROPOSAL. Supersedes the earlier v2 draft that framed SDL
as a positive diagnostic. Empirical findings from `exp_30_sdl_vs_between`
required rewriting the claim structure.
**Relation to v1 (`paper_strategy.md`)**: v1 is the "positive finding"
framing (Type A). v2 is a "benchmark critique + mechanism" framing (Type C).
v1's machinery (variance decomposition, RSA, PERMANOVA) is *reused* as
evidence for v2, but the narrative spine is fundamentally different.

---

## 1. Paper type and why it matters

v2 is a **Type C paper**: critique of existing benchmark results + mechanistic
explanation. Standard Type C papers (e.g. Recht et al. on ImageNet
generalization, Geirhos et al. on shortcut learning) have two hallmarks:

- The Introduction starts with a **general phenomenon**, not the authors' own
  dataset.
- The main results are **differential** (A behaves differently from B), not a
  single-effect demonstration.

This rules out the original "Stress as protagonist" narrative chain. The
paper opens with the inconsistency in the EEG-FM benchmark literature, not
with UCSD Stress. Stress is one of six test datasets, treated in parallel
with the others in §3. Its role shifts from "flagship" to "test platform +
methodological motivator for dataset selection" (see §5 below).

---

## 2. The central claim (Abstract-level)

EEG foundation models (LaBraM, CBraMod, REVE) are marketed as general-
purpose encoders but their transfer behaviour across benchmarks is
inconsistent — diagnostic/between-subject benchmarks (AD/FTD, MDD,
expertise) report 70–90% BA while within-subject cognitive-state
benchmarks (mental arithmetic, stress, sleep deprivation) report 50–70%
BA. This paper argues the gap is **not** a dataset-hardness artefact but
a representation-level confound: **the FMs encode subject identity in
their frozen features, and in benchmarks where subject identity is
collinear with the label (between-subject designs), fine-tuning exploits
this collinearity as a shortcut.** We demonstrate this by showing:

1. FM frozen features are subject-dominated across six datasets and three
   models (C1).
2. The benefit of fine-tuning over linear probing (ΔBA = FT − LP) is
   predicted by subject-identity decodability in the between-subject arm
   (ρ ≈ 0.70, n = 9) but *not* in the within-subject arm (ρ ≈ 0, n = 6)
   (C2).

The critique: EEG-FM benchmark papers systematically overestimate
transferability by reporting numbers dominated by between-subject
diagnostic tasks. Within-subject tasks — the regime where subject
leakage cannot drive performance — are the honest benchmark.

---

## 3. Revised contribution list

**C1 (mechanism).** FM frozen features are subject-dominated (40–97%
subject variance, 0–28% label variance) across all six datasets and
three FMs. Demonstrated via nested/crossed SS decomposition,
corroborated by PERMANOVA and subject-ID probe BA.

**C2 (differential benchmark inflation).** Subject-ID decodability from
frozen features predicts ΔBA in between-subject benchmarks
(Spearman ρ = 0.70, 95% CI [0.12, 0.98], n = 9) but is null in clean
within-subject benchmarks (ρ ≈ 0, n = 6). This is the quantitative
signature of subject-leakage transfer.

**Dropped claims (relative to earlier v2 draft):**

- ~~C3 (pre-FT task-variance predicts ΔBA in within-arm)~~ — falsified.
  No predictor in the within-subject arm reliably forecasts ΔBA at
  n = 6 or n = 9. Reported as a **negative result** under Limitations.
- ~~C4 (F-HHSA gradient)~~ — HHSA contrast does not correlate with FM
  performance across datasets at this N. Demoted to supplementary.

---

## 4. Paper structure (Type C, minimal-spine)

```
§1  Introduction
    §1.1  The inconsistency phenomenon: EEG-FM benchmark gap table
          (between-subject vs within-subject reported BA across LaBraM,
          CBraMod, REVE) — motivates the paper
    §1.2  Existing explanations (dataset noise, label quality) and their
          failure to predict the across-benchmark pattern
    §1.3  Our hypothesis: representation-level subject-label confound
    §1.4  Contributions: C1 + C2

§2  Methods
    §2.1  Datasets — six datasets organised by within/between design
          (Stress treated in parallel; §2.1.1 notes Stress DASS
          degeneracy and justifies excluding it from the clean
          within-arm in §3.2)
    §2.2  Foundation models and evaluation protocol (subject-level
          5-fold CV; LP vs FT definitions; cuDNN-deterministic seeds)
    §2.3  Analysis metrics — variance decomposition (nested/crossed),
          RSA, subject-ID probe (LR + MLP), PERMANOVA, HHSA (optional)

§3  Results
    §3.1  C1 — FM frozen features are subject-dominated.
          Headline figure: variance atlas, 6 datasets × 3 FMs,
          Label%/Subject%/Residual% bar stack. Supporting panel:
          subject_id_ba per cell. No arm split here — this is a
          property of FM representations regardless of benchmark.
    §3.2  C2 — Subject leakage inflates between-subject benchmarks.
          Headline figure: ΔBA vs subject_id_ba, split by arm
          (within-strict: EEGMAT + SleepDep, n = 6; between: ADFTD +
          TDBRAIN + Meditation, n = 9). Bootstrap CI on each ρ.
          Supporting: same figure for rsa_label and
          subject_to_label_ratio (both show the same arm flip).
    §3.3  (Optional) Robustness — LR vs MLP probe, leave-one-dataset-out
          sensitivity, within-strict vs within-loose comparison.
          Could be condensed into §3.2 if space is tight.

§4  Discussion
    §4.1  Implications for EEG FM pretraining — motivates subject-
          invariant objectives (subject-adversarial loss, cross-subject
          contrastive, etc.)
    §4.2  Benchmark design recommendations — within-subject tasks as
          gold standard; between-subject benchmarks should report
          subject-ID decodability as leakage proxy
    §4.3  Limitations (including C3 negative result — no within-arm
          predictor at this N)

§5  Appendix / Supplementary
    Buckets (final decision at paper-finalisation, not now):
    - Full FT direction analysis (original §3.2 EEGMAT-Stress) →
      justifies Stress exclusion from §3.2 clean arm
    - Stress DASS design degeneracy analysis
    - DSS correlation with FT gain (flagged reporting bias)
    - Per-dataset variance decomposition + RSA + PERMANOVA tables
    - HHSA per-dataset contrast (untested as predictor but kept as
      signal-side documentation)
    - Permutation-null sanity check (from exp_newdata)
```

Only **two** main-text result sections (§3.1, §3.2), each carrying one
headline figure. Everything else is supplementary.

---

## 5. Stress's role in v2

Stress is NOT the protagonist in v2. It is one of six test datasets,
handled in parallel with the others. It appears in:

- **§2.1.1 dataset note**: 3/17 subjects crossing DASS classes — explicitly
  stated as a design degeneracy, with justification for exclusion from §3.2
  clean within-arm.
- **§3.1 variance atlas**: one of 6 datasets (3 × FM = 3 cells in the
  18-cell grid). C1 is descriptive, not arm-dependent.
- **§3.2 is computed on within-strict = EEGMAT + SleepDep only**. Stress
  is excluded but appears in a robustness row (within-loose incl. Stress)
  to show the conclusion is not sensitive to the exclusion.
- **Appendix**: DSS correlation + FT direction supporting analyses.

This preserves the Stress-specific work (variance decomposition, seed
sensitivity, η² pipeline) as evidence but removes it from the narrative
spine.

---

## 6. Dataset allocation (Option C, clean-within-strict)

**Clean within-subject arm (primary §3.2 evidence, n = 6 cells):**

| Dataset  | Label           | Design          | N rec / sub | Used in §3.2 |
|----------|-----------------|-----------------|-------------|--------------|
| EEGMAT   | Task vs rest    | 72 / 36         | 100% crossing | Yes |
| SleepDep | NS vs SD        | 72 / 36         | 100% crossing | Yes |

**Between-subject arm (n = 9 cells):**

| Dataset    | Label                 | N rec / sub   | Used in §3.2 |
|------------|-----------------------|---------------|--------------|
| ADFTD      | AD / FTD / CN         | 195 / 65      | Yes |
| TDBRAIN    | MDD / CN (binary)     | 734 / 359     | Yes |
| Meditation | Expert / Novice       | 40 / 24       | Yes |

**Excluded from §3.2 clean arms, present in §3.1 and appendix:**

| Dataset | Reason |
|---------|--------|
| Stress  | Only 3/17 subjects cross DASS classes — design degeneracy |
| SAM40   | 25 s recordings — incompatible with HHSA and underpowered |

---

## 7. Testable predictions (revised falsifiers)

**F1 — Variance dominance holds across all six datasets.**
`frozen_subject_frac` > 40% for every (FM, dataset) cell (18 cells total).
*Refuted if* > 2 cells show subject_frac < 40%.

**F2 — Subject-ID probe tracks ΔBA in between-arm only.**
Spearman ρ(subject_id_ba, ΔBA):
- between-arm (n = 9): CI excludes 0 with positive ρ
- within-strict (n = 6): CI crosses 0

*Refuted if* (a) between-arm CI crosses 0, or (b) within-strict CI
excludes 0 with positive ρ.

**F3 — Arm-flip robustness.**
Add at least one additional predictor (`rsa_label`, `subject_to_label_ratio`)
that shows the same sign-flip pattern between arms.

*Refuted if* no predictor other than subject_id_ba shows arm-dependent
behaviour.

**F4 — Leave-one-dataset-out sensitivity.**
Dropping any single dataset from the between-arm leaves ρ > 0 with
|ρ| > 0.3 and CI still on the positive side.

*Refuted if* the ρ = 0.70 finding collapses when any single dataset is
removed.

F2 and F3 together are the paper; F1 is foundational; F4 is robustness.

---

## 8. Current exp_30 evidence against falsifiers (v4 pipeline results, 2026-04-18)

From `results/studies/exp_30_sdl_vs_between/`:

- **F1 status**: ✅ CONFIRMED. All 18 cells show frozen_subject_frac > 39%.
- **F2 status**: ⚠️ PARTIALLY SUPPORTED BUT WEAK.
  - Between-arm ρ(subject_id_ba_lr, ΔBA) = +0.70, 95% CI [0.11, 0.98],
    p = 0.036, n = 9 — strong and near-significant.
  - Within_strict (EEGMAT + SleepDep, n = 6) ρ = +0.60 with CI
    [−0.50, +1.00]. CI crosses 0, so technically "null" under the
    pre-registered definition, BUT the point estimate is similar to
    between. The earlier within-loose ρ = −0.02 (n = 9, incl. Stress)
    was largely Stress-driven and NOT evidence of a genuine arm-flip.
  - Conclusion: **the differential claim (between-only leakage) is not
    currently supported by the data**. Point estimates are close; only
    statistical significance distinguishes them. Could become clean
    with more within-subject datasets (TUAB + HMC + ADFTD-3cls).
- **F3 status**: `subject_to_label_ratio` between ρ = −0.63 (p = 0.067)
  vs within_strict ρ = +0.14 — direction consistent with arm flip but
  p > 0.05. `rsa_label` arm flip disappears under within_strict
  (+0.09 vs +0.60 → the earlier −0.50 was Stress-driven). ⚠️ WEAK.
- **F4 status**: NOT YET TESTED. Scheduled as exp_30 v5 (LOO dataset
  bootstrap — drop 1 of 3 between-arm datasets, check ρ robustness).

**Three possible paper versions based on current evidence:**

- **(A) Full differential claim** ("subject leakage affects between-arm only"):
  NOT currently supported. Needs TUAB + HMC + ADFTD-3cls to lift N and
  genuinely test arm differential. ~20–30 h GPU cost.
- **(B) Unified claim** ("subject-ID decodability universally predicts
  FT gain"): supported at n = 6 + n = 9 = 18 pooled. Loses the
  benchmark-critique edge (not a Type C contribution; more like a
  Type A "universal predictor" finding).
- **(C) Hybrid narrow claim**: C1 (subject dominance) + "in between-arm
  specifically, subject-ID decodability and subject/label ratio correlate
  with FT gain" + "within-arm evidence inconclusive at current N". This
  is the honest default that current data support.

**Recommendation**: adopt (C) as the default framing; plan to upgrade to
(A) after TUAB/HMC/ADFTD-3cls are collected. DO NOT rewrite tex until
(A) is unlocked or (C) is explicitly accepted by the user.

### Seed-noise bootstrap (2026-04-18, critical robustness check)

Observation raised: our ΔBA magnitudes (0.01–0.10 BA) are small compared
to seed-to-seed noise (std 0.02–0.07 per cell). Literature FT gains are
an order of magnitude larger (TUAB +0.10, PhysioMI +0.19). Risk: our
ρ = 0.70 point estimate may be a lucky-seed artefact.

10 000-iter seed-noise bootstrap:

| Predictor | Arm | median ρ | 95 % CI | P(ρ expected direction) |
|---|---|---|---|---|
| subject_id_ba_lr | between | **+0.50** | [+0.03, +0.83] | **98.4 %** |
| subject_id_ba_lr | within_strict | +0.43 | [−0.43, +0.89] | 77.9 % |
| subject_id_ba_lr | within_loose (incl. Stress) | −0.05 | [−0.55, +0.38] | 40.5 % |
| subject_to_label_ratio | between | **−0.50** | [−0.85, +0.03] | **96.7 %** (neg) |
| subject_to_label_ratio | within_strict | +0.14 | [−0.54, +0.71] | 60.0 % |

**Revised seed-robust claim** (replaces point-estimate ρ = 0.70 / −0.63):

- Between-arm: both predictors robust at ρ ≈ 0.50 / −0.50 under seed
  noise. Direction is 96–98 % robust.
- Within-arm (strict): positive tendency but 77.9 % < 95 %, so
  "underpowered to confirm null" rather than "null confirmed".
- **Within-arm does NOT show the arm-flip we hoped for under seed-noise
  bootstrap either**. The "leakage is arm-specific" claim is not
  supported at current N regardless of how we handle noise.

**Root cause diagnosis**: ρ is attenuated by noise-in-outcome (ΔBA
reliability issue). Literature benchmarks with larger FT signals
(TUAB +0.10, HMC +0.12, PhysioMI +0.19) would reduce attenuation and
likely tighten ρ considerably. This reinforces the case for running
TUAB + HMC + ADFTD-3cls before finalising the paper.

### Current-data paper claim (what we can honestly write today)

> Subject-ID linear decodability from frozen features is consistently
> positively associated with FT gain across six EEG benchmarks. The
> relationship is statistically robust in between-subject benchmarks
> (ρ = +0.50, 95 % CI [+0.03, +0.83], n = 9 under seed-noise bootstrap)
> but underpowered in within-subject benchmarks (ρ = +0.43, n = 6).
> A complementary signature — subject-to-label variance ratio — shows a
> negative association with FT gain, also robust only in the
> between-subject arm (ρ = −0.50, 95 % CI [−0.85, +0.03]). Together
> these suggest subject identity acts as an extractable shortcut feature
> when label signal is not buried. Whether the relationship genuinely
> differs between arms requires higher-N replication.

This is the v4-data honest scope. Paper publishability hinges on
accepting this scope (path C/B) or lifting N via TUAB + HMC (path A).

---

## 9. Intro motivation table (in progress)

Table 1 in the Introduction will be a literature-sourced benchmark gap
table:

- Rows: benchmarks organised by within/between design
- Columns: LaBraM LP/FT, CBraMod LP/FT, REVE LP/FT
- Filter: ONLY benchmarks where the paper EXPLICITLY states subject-level
  vs trial-level CV (to avoid comparing apples to oranges)
- Data sources: LaBraM paper (ICLR 2024), CBraMod (2025), REVE (2025),
  EEG-FM-Bench if available, our own exp_30 numbers for Stress/EEGMAT/
  SleepDep/ADFTD/TDBRAIN/Meditation

Background literature search spawned (see `scripts/analysis/` TBD). If
the literature table shows the gap clearly, it goes into §1.1. If not,
we pivot: maybe the gap is smaller than intuition suggests and the real
story is just the mechanism (subject-ID probe, C2), not the gap itself.

---

## 10. Remaining open decisions (user input needed)

1. **Table 1 design**: once the literature search returns, decide whether
   the gap is visually strong (use as §1.1 motivator) or subtle (shift to
   §3.1 methods-level table).
2. **Appendix scope**: decided at finalisation, not now.
3. **FT hyperparameter re-tuning**: the current FT numbers used default
   hyperparameters; published papers often do HP search. Decide whether
   to re-run FT with HP search on between-arm datasets to rule out
   "our FT is just under-tuned" objection.
4. **Tex rewrite timing**: do not start until F2 and F4 are confirmed.
   Current estimate: one additional pipeline run + figure polish, then
   the rewrite is targeted (§1 intro swap, §3 consolidation, new
   Discussion).

---

## 11. Do not do (guardrails)

- Do not place Stress as the narrative protagonist. It is one of six
  test datasets, period.
- Do not claim within-arm predictors work. C3 is falsified; own it.
- Do not claim HHSA contrast predicts ΔBA. C4 is falsified; own it.
- Do not mix trial-level and subject-level CV numbers in the literature
  table. Exclude rows where split type is unclear.
- Do not rewrite tex until F2 (within-strict) and F4 (LOO robustness)
  are confirmed by exp_30 v4/v5 pipeline runs.
