# TODO — Current Priorities

**Last updated**: 2026-04-14

Update this file as priorities shift. Delete completed items; don't accumulate history.

---

## Narrative direction (2026-04-14 decision)

**Stress digging paused**. Every Stress injection/erosion claim (F05) has been weakened
by follow-ups (F06 LaBraM p=0.70, F18 REVE window-artifact, F19 CBraMod/REVE p=0.100,
F20 ShallowConvNet matches FMs). 70 rec / 14 positive regime is below the statistical
power floor — further Stress experiments will keep producing "borderline" results.

**Paper reframe**: Stress → **power-floor case study**, not main result.
- Main signal: F17 cross-dataset taxonomy (ADFTD + TDBRAIN + EEGMAT)
- F02/F13: subject-identity dominance across all FMs
- exp14: neuroscience interpretability (spatial + spectral, correlational + causal)
- Stress cautionary tale: F08 cuDNN swing + F19 null-indistinguishable + F20 architecture
  doesn't matter → warns small-sample EEG benchmarks can't support FM superiority claims
- Connects to F11 (Brain4FMs / EEG-FM-Bench / AdaBrain-Bench concerns)

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
   - Walk through `docs/findings.md` (F17, F18, F19, F20 are the key updates)
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

## On hold (not pursuing unless narrative shifts)

- **Extend perm null to 20 perms** (would bring CBraMod/REVE p=0.100 → potentially p≤0.05
  if injection is real, or confirm noise if not). Paused because current borderline is
  consistent with Stress being below power floor; tighter p-value doesn't change that story.
- **EEGNet re-sweep / multi-seed expansion** (s2024=0.411 is cuDNN-induced outlier,
  current 0.518 ± 0.079 is weak vs ShallowConvNet 0.557 ± 0.026). ShallowConvNet alone
  is sufficient to demonstrate F20 architecture-agnostic ceiling.

---

## Ruled out — do not re-propose

- Subject-adversarial GRL/DANN / LEAD-style subject CE loss — F12
- Canonical 0.656 as "LaBraM ceiling" — cuDNN noise (F08)
- `--label subject-dass` — deprecated (F07)
- Within-subject longitudinal DSS reframing — F14 negative
- Sparse-label-subspace hypothesis — refuted
- Obs 1702 "5s/10s frozen LP identical" — wrong, F18 shows ~5pp gap

---

## Paper discussion section — points to weave in

- GRL/DANN and LEAD negative results (F12) as mitigation attempts
- Direction consistency (EEGMAT vs Stress) addresses self-report concern
- Brain4FMs / EEG-FM-Bench / AdaBrain-Bench convergence (F11)
- ADFTD 10s vs 5s window effect (exp12 vs exp07)
- Neuroscience triad (exp14): spatial + correlational spectral + causal spectral
- Stress power floor: F08 cuDNN swing + F19 null-indistinguishable + F20 architecture
  doesn't matter — benchmark-design caveat for the field
- F18: REVE injection is window-dependent — window choice is a first-class concern
