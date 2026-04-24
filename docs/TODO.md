# TODO — Current Priorities

**Last updated**: 2026-04-23 (alignment axis pivot complete; Fig 3 2×2 is the paper anchor)

Update this file as priorities shift. Delete completed items; don't accumulate history.

> **ID scheme (2026-04-15)**: paper claims now use `F-A`…`F-E` (see `docs/findings.md`).
> Guardrails + notes use `G-F##` / `N-F##` / `A-F##` (see `docs/methodology_notes.md`).

---

## Current narrative direction (2026-04-23)

Primary axis = **task-substrate alignment** (strong vs weak-aligned); secondary structural context = **CV regime** (within-subject paired vs subject-label trait). The 2×2 factorial anchored by Fig 3 permutation null is the paper's central visual argument.

- **Strong-aligned**: EEGMAT (theta/alpha substrate) + ADFTD (1/f aperiodic slope) — FM clears null (p ≤ 0.05)
- **Weak-aligned**: Stress (DASS-21 behavioral) + SleepDep (self-reported state) — FM stays inside null (p > 0.1)
- Stress is a case study inside the weak-aligned × subject-label cell; not the protagonist
- Full framing: `docs/findings.md` §Central thesis + `docs/paper_outline.md`

---

## Needs decision (advisor / you)

### D2. How to present REVE injection (F18)
REVE paper uses 10s windows natively → use **10s-matched (+12.8pp)** as primary.
5s-matched (+4.8pp) goes to supplementary as robustness check.

### D3. Whether to re-run FT with HP search on between-arm datasets
Open. Our FT uses default hyperparameters across three seeds; published FM papers often do HP search. See "Lower priority" below.

---

## High priority — active work

1. **F4 leave-one-dataset-out robustness** (pipeline Stage K)
   - For each between-arm dataset, drop it and recompute ρ
   - Claim is robust only if dropping any single dataset keeps |ρ| > 0.3 with CI not collapsing
   - Priority: BEFORE any tex rewrite

2. **Add TUAB to pipeline** (gated on path A)
   - ~600 subjects, binary abnormal/normal; subject-level split standard
   - Adds 4th between-arm dataset → n = 9 → 12
   - Cost: ~5–8 h FT per FM (15–24 h total); need to download TUAB

3. **Add HMC (sleep staging) to within_strict arm** (gated on path A)
   - 151 subjects, 5-class, subject-level split standard
   - Adds 3rd within-strict dataset → n = 6 → 9; current CI is [−0.50, +1.00]
   - Cost: ~4–6 h FT per FM (12–18 h total); need to download HMC
   - **Most important single addition** for path A


4. **EEGMAT architecture ceiling → Fig 6 left** (alignment-conditional architecture claim)
   - Need EEGMAT counterpart showing ceiling rises to 0.70+ for same architectures as Stress flat ceiling
   - Check `results/studies/exp15_nonfm_baselines/sweep/*` for existing EEGMAT runs
   - If missing: eegnet + shallowconvnet × 3 seeds on EEGMAT (~2 h)

5. **2×2 axis text in Methods §3.1 + Intro** (axis renamed 2026-04-23)
    - Intro: state taxonomy upfront — row = CV regime; column = task-substrate alignment. Cite DEAP/SEED/DREAMER/DASPS as additional (subject-label × weak-aligned) instances (no new experiments)
    - Methods §3.1: map four datasets onto 2×2 with operational definitions — alignment: p ≤ 0.05 strong / p > 0.1 weak; regime: label design structure
    - Zero experimental cost; ~500 words

6. **Sweep remaining `docs/paper_outline.md` for stale "regime"/"signal coherence" language**
    - Sections 4.x, Discussion, Supplementary not yet updated to alignment terminology
    - Propagate when §1 Intro draft begins

---

## Lower priority — defer until paper direction locked

- **HP search on between-arm FT** (D3 resolution)
  - ΔBA magnitude is not claim-load-bearing under current framing
  - Only run if reviewers push back on "your FT is undertuned"
  - HP grid: lr ∈ {1e-5, 5e-5, 1e-4}, wd ∈ {0, 0.01, 0.1}, drop ∈ {0, 0.1}
  - Cost: 9 × 3 FMs × 3 datasets = 81 runs. Heavy.

---

## HHSA / WSCI pipeline (demoted — C4 falsified)

- HHSA contrast does not correlate with ΔBA at n=9 (ρ=−0.26, CI wide). Lives in appendix as signal-side documentation.
- HHSA L1 cache rebuild (N_ENSEMBLES=100) no longer gates paper submission.
- WSCI analysis similarly demoted.

---

## On hold — not pursuing unless narrative shifts

- **EEGNet re-sweep** — not relevant under current framing

---

## Ruled out — do not re-propose

- Subject-adversarial GRL/DANN / LEAD-style subject CE loss — N-F12
- Canonical 0.656 as "LaBraM ceiling" — cuDNN noise (G-F08)
- `--label subject-dass` — deprecated (G-F07)
- Within-subject longitudinal DSS reframing — F-D.3 negative + reporting-bias concern
- Sparse-label-subspace hypothesis — refuted
- Obs 1702 "5s/10s frozen LP identical" — wrong, F-C.4 shows ~5pp gap
- **C3 — within-arm task-variance predicts ΔBA** — falsified by exp_30
- **C4 — HHSA contrast tracks FM performance** — falsified by exp_30

---

## Paper discussion section — points to weave in

- C1 mechanism ties to pretraining without subject-invariant objectives
- Connection to N-F12 (GRL/DANN/LEAD negative results) — motivates C2's prescription for subject-invariant pretraining
- Split protocol inconsistency in EEG-FM-Bench (stratified ADFTD vs subject-level TUAB) — direct evidence for the critique
- Brain4FMs / EEG-FM-Bench / AdaBrain-Bench convergence (N-F11)
- Stress power floor (G-F08 cuDNN swing + N-F19 null-indistinguishable + F-D.2) — benchmark-design caveat; now in service of Type C critique not Type A positive claim
- **FT<LP prior art**: BENDR (Frontiers 2021) frozen+linear > full FT on 4/5 datasets under LOSO. AdaBrain-Bench (2025) EEGPT LP 47.9 vs FT 25.8 on BCI-IV-2a. EEGPT chooses LP as headline metric. Our negative ΔBA in 7/9 between cells rediscovers this pattern with correlation-level mechanism (subject_id_ba ~ FT degradation magnitude).
- Our 5-fold subject CV × 3 seeds is MORE rigorous than community standard (fixed split × 3-5 seeds). ±3 pp uncertainty includes data-sampling variance most papers structurally cannot report.
- Within-arm null result (C3 falsification) — honest limitation + open question for future benchmark design
