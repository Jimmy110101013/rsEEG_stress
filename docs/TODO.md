# TODO — Current Priorities

**Last updated**: 2026-04-26 (LaBraM input-norm aligned to /100; pre-fix results need re-validation)

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

1. **🚨 Re-validate LaBraM results under aligned input norm** (commit `ac1e115`, 2026-04-26)
   - Pre-fix LaBraM runs used per-window zscore; original LaBraM (and EEG-FM-Bench / Benchmarking_EEG_Analysis spec) uses `raw µV / 100`
   - Extractor now does `/100` internally (mirrors CBraMod pattern); all `MODEL_NORM` dicts and shell launchers pinned to `none`
   - **All pre-2026-04-26 LaBraM numbers are NOT directly comparable** to post-fix runs — ΔBA, F-A through F-E LaBraM cells, perm null, HP sweep all need re-validation before the paper rests on them
   - Action: rerun canonical 3-seed FT (`scripts/experiments/run_final_ft_labram.sh`) on Stress + EEGMAT + ADFTD + SleepDep first; then frozen LP (`extract_frozen_*` → `train_lp.py`) for the same 4 datasets
   - Cost: ~6–8 h FT × 4 datasets × 3 seeds + ~2 h LP extraction ≈ 1 GPU-day
   - **Block on this**: any rewriting/figures that cite a LaBraM number; once a representative subset re-runs, decide whether to swap in new numbers wholesale or annotate old ones as pre-alignment ablation
   - Trace: grep `2026-04-26` or commit `ac1e115` for all touched files; CLAUDE.md §2 carries the guardrail

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
