# TODO — Current Priorities

**Last updated**: 2026-04-15

Update this file as priorities shift. Delete completed items; don't accumulate history.

> **ID scheme (2026-04-15)**: paper claims now use `F-A`…`F-E` (see `docs/findings.md`).
> Guardrails + notes use `G-F##` / `N-F##` / `A-F##` (see `docs/methodology_notes.md`).
> Inline `F##` references below remain valid via each claim's `Absorbs:` mapping.

---

## Narrative direction (2026-04-14 decision)

**Stress digging paused**. Every Stress injection/erosion claim (F-C.2, ex-F05) has been
weakened by follow-ups (D.1 LaBraM p=0.70, F-C.4 REVE window-artifact, N-F19
CBraMod/REVE p=0.100, D.2 ShallowConvNet matches FMs). 70 rec / 14 positive regime is
below the statistical power floor — further Stress experiments will keep producing
"borderline" results.

**Paper reframe**: Stress → **power-floor case study**, not main result.
- Main signal: **F-C** cross-dataset taxonomy (ADFTD + TDBRAIN + EEGMAT + Stress HP sweep)
- **F-A**: subject-identity dominance across all FMs
- exp14: neuroscience interpretability (spatial + spectral, correlational + causal)
- Stress cautionary tale (**F-D**): G-F08 cuDNN swing + N-F19 null-indistinguishable +
  F-D.2 architecture-independent → warns small-sample EEG benchmarks can't support FM
  superiority claims
- Connects to N-F11 (Brain4FMs / EEG-FM-Bench / AdaBrain-Bench concerns)

---

## Needs decision (advisor / you)

### D1. Paper narrative framing
**Tentative**: Option B + C hybrid (task-property framing + cross-dataset taxonomy).
Confirm with advisor; then update `docs/paper_strategy.md`.

### D2. How to present REVE injection (F18)
REVE paper uses 10s windows natively → use **10s-matched (+12.8pp)** as primary.
5s-matched (+4.8pp) goes to supplementary as robustness check.

---

## High priority

1. **Align advisor on updated findings + reframe**
   - Walk through `docs/findings.md` (F-C multi-dataset taxonomy and F-D power-floor are the key claims)
   - Confirm D1 + D2 above

2. **Paper drafting** (narrative locked; outlines in progress 2026-04-14)
   - `paper/main.tex` + `paper/sections/*.tex` — outlines replacing obsolete
     "Subject Identity Dominance Bounds FM" draft
   - Table 1: `paper/sections/table1_master.tex` (LaTeX Master Table)
   - Figures: `paper/figures/main/` and `paper/figures/supplementary/`
     (provenance ledger in `paper/figures/README.md`)

3. ~~Fill 2 missing Master Table cells — CBraMod + REVE EEGMAT FT~~ **DONE 2026-04-14**
   - `results/studies/exp17_eegmat_cbramod_reve_ft/{cbramod,reve}_s{42,123,2024}/`
   - CBraMod EEGMAT FT **0.620 ± 0.058** (Δ from frozen = **−11.1 pp**, largest
     erosion in the table — pre-trained CBraMod is already optimal for EEGMAT
     rest/task contrast and FT breaks it)
   - REVE EEGMAT FT 0.727 ± 0.035 (Δ = +5.6 pp, modest injection)
   - Table regenerated in `paper/figures/source_tables/master_frozen_ft_table.md`

4. ~~Rerun trial-level CV under per-rec DASS for Fig 2~~ **DONE 2026-04-14**
   - `results/studies/exp18_trial_dass_multiseed/{labram,cbramod,reve}_s{42,123,2024}/`
   - Per-rec DASS trial/subject gap: LaBraM **+15.2 pp**, REVE **+6.5 pp**, CBraMod **+1.2 pp** (mean +7.6 pp)
   - Much smaller than subject-dass version (−21 pp uniform); CBraMod gap
     almost vanishes under honest labels
   - Fig 2 regenerated (`paper/figures/main/fig2_cv_gap.pdf`) with 3-seed error bars
   - F01 updated with the new table

## Lower priority

- **HNC Dementia + MDD private dataset** — branch `feat/hnc-dementia-mmd`, 308+400 subjects
  (if Stress power floor becomes central narrative, HNC could provide the
  high-powered counter-example)

## HHSA / WSCI pipeline

- **Rebuild HHSA L1 cache with N_ENSEMBLES=100** — current cache uses N=24
  (monkey-patched in `hhsa_build_cache.py:39-40`), design doc specifies 100.
  Residual noise is 2× higher. Sufficient for relative WSCI comparison across
  datasets, but rebuild if absolute holospectrum values look unclear or noisy.
  Affects all 5 datasets (stress, eegmat, sleep_dep, sam40, meditation).
  Cost: ~4× slower cache build (~6 hr for all datasets).
- **WSCI analysis** — run `scripts/run_wsci_from_holospectra.py` after
  holospectrum computation completes. Compare EEGMAT vs Stress as validation.
- **Extend WSCI to sleep_dep + sam40 + meditation** — L1 cache building;
  holospectrum + WSCI scripts need dataset-specific condition splits.

## On hold (not pursuing unless narrative shifts)

- **Extend perm null to 20 perms** (would bring CBraMod/REVE p=0.100 → potentially p≤0.05
  if injection is real, or confirm noise if not). Paused because current borderline is
  consistent with Stress being below power floor; tighter p-value doesn't change that story.
- **EEGNet re-sweep / multi-seed expansion** (s2024=0.411 is cuDNN-induced outlier,
  current 0.518 ± 0.079 is weak vs ShallowConvNet 0.557 ± 0.026). ShallowConvNet alone
  is sufficient to demonstrate F-D.2 architecture-agnostic ceiling.

---

## Ruled out — do not re-propose

- Subject-adversarial GRL/DANN / LEAD-style subject CE loss — N-F12
- Canonical 0.656 as "LaBraM ceiling" — cuDNN noise (G-F08)
- `--label subject-dass` — deprecated (G-F07)
- Within-subject longitudinal DSS reframing — F-D.3 negative
- Sparse-label-subspace hypothesis — refuted
- Obs 1702 "5s/10s frozen LP identical" — wrong, F-C.4 shows ~5pp gap

---

## Paper discussion section — points to weave in

- GRL/DANN and LEAD negative results (N-F12) as mitigation attempts
- Direction consistency (EEGMAT vs Stress) addresses self-report concern
- Brain4FMs / EEG-FM-Bench / AdaBrain-Bench convergence (N-F11)
- ADFTD 10s vs 5s window effect (exp12 vs exp07)
- Neuroscience triad (exp14): spatial + correlational spectral + causal spectral
- Stress power floor: G-F08 cuDNN swing + N-F19 null-indistinguishable + F-D.2 architecture
  doesn't matter — benchmark-design caveat for the field
- F-C.4: REVE injection is window-dependent — window choice is a first-class concern
