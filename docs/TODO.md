# TODO — Current Priorities

**Last updated**: 2026-04-23 (axis officially renamed to task-substrate alignment: strong / weak-aligned; Fig 3 2×2 anchors the column; regime becomes the row)

Update this file as priorities shift. Delete completed items; don't accumulate history.

> **ID scheme (2026-04-15)**: paper claims now use `F-A`…`F-E` (see `docs/findings.md`).
> Guardrails + notes use `G-F##` / `N-F##` / `A-F##` (see `docs/methodology_notes.md`).
> Inline `F##` references below remain valid via each claim's `Absorbs:` mapping.

---

## Narrative direction (2026-04-21 regime refinement)

Cross-session framing sharpened on top of Type C critique. Primary axis is **regime**, not dataset:

- **Subject-label regime** (label is a per-subject scalar, no within-subject contrast): Stress is the canonical instance. DEAP/SEED/DREAMER/DASPS share the archetype.
- **Within-subject regime** (label is a paired within-subject contrast): EEGMAT (rest vs arithmetic) and SleepDep (normal vs sleep-deprived) are the two datapoints.

Stress stays as a case study inside the subject-label regime — it is NOT the protagonist. The paper's diagnostic framework generalizes across regime, tested on two.

Full decision log: `docs/regime_framing_decision.md`.

**New open items (2026-04-21)** — flagged below in High priority (#11–#14).

---

## Narrative direction (2026-04-18 pivot)

**Active paper framing**: Type C benchmark critique. See
`docs/paper_strategy_sdl_critique.md` (revised 2026-04-18). Only two
main-text claims now:

- **C1** — FM frozen features are subject-dominated across all 6 datasets × 3 FMs
- **C2** — subject-ID decodability predicts ΔBA in between-subject arm
  (ρ=0.70, p=0.036, n=9) but is null in clean within-arm (n=6)

**Dropped (falsified by exp_30 data):**
- C3 (within-arm task-variance predicts ΔBA)
- C4 (HHSA contrast tracks FM performance)

**Stress is no longer the paper protagonist.** Stress is one of six
datasets in §3.1 variance atlas, excluded from §3.2 clean within-arm
(3/17 DASS-crossing subjects), supporting analyses in appendix.

The older "Stress power-floor case study" framing from 2026-04-14 is
superseded — but several of the downstream guardrails (G-F08 cuDNN
swing, F-D.2 architecture-agnostic ceiling, N-F11 Brain4FMs
concerns) remain relevant as supporting evidence for the new critique.

---

## Needs decision (advisor / you)

### D1. Paper narrative framing ✅ RESOLVED 2026-04-18
Type C benchmark critique. See `docs/paper_strategy_sdl_critique.md` §1–4.

### D2. How to present REVE injection (F18)
REVE paper uses 10s windows natively → use **10s-matched (+12.8pp)** as primary.
5s-matched (+4.8pp) goes to supplementary as robustness check.

### D3. Whether to re-run FT with HP search on between-arm datasets
Open. Our FT uses default hyperparameters across three seeds; published
FM papers often do HP search. See "Lower priority — extend exp_30" below.

---

## High priority — finish exp_30 + paper

**Decision gate**: v4 results show F2 is SUPPORTED only on CI-crosses-0
interpretation, not on point-estimate-magnitude. Within_strict ρ = +0.60
is similar to between ρ = +0.70; the "differential claim" is currently
underpowered. See `docs/paper_strategy_sdl_critique.md` §8 for three
possible paper versions (A/B/C).

1. **✅ v4 pipeline complete** — within_strict + MLP probe done.
   - REPORT.md §0 has honest v4 interpretation.
   - Moves the differential claim from "confirmed" to "partially supported, underpowered".

2. **✅ Seed-noise bootstrap done** (2026-04-18)
   - `tables/seed_noise_bootstrap.json` — 10k-iter bootstrap on ΔBA
   - Between-arm ρ ≈ +0.50 / −0.50 seed-robust (96–98% prob expected direction)
   - Original point estimates ρ = 0.70 / −0.63 were optimistic (lucky seed)
   - Within_strict ρ ≈ +0.43 (77.9% prob positive) — still underpowered
   - Differential arm claim **not** supported under seed noise either

3. **Add F4 leave-one-dataset-out robustness** (pipeline Stage K, next version)
   - For each between-arm dataset, drop it and recompute ρ
   - Paper claim is robust only if dropping any single dataset keeps |ρ| > 0.3 with CI not collapsing
   - Priority: BEFORE any tex rewrite.

4. **DECISION — paper version A / B / C**
   - If user picks (A) full differential: proceed to items 5–7 below.
   - If user picks (C) hybrid narrow: skip 5–7, go to paper drafting with current data.
   - (B) is a Type A unified claim, not recommended.

5. **Add TUAB to pipeline** — UPGRADED FROM LOWER PRIORITY
   (needed to lift between-arm N for path A)
   - ~600 subjects, binary abnormal/normal; subject-level split standard
   - Adds 4th between-arm dataset → n = 9 → 12
   - Cost: ~5–8 h FT per FM (15–24 h total); need to download TUAB
   - Justification: directly shores up F2 between-arm CI, fixes "too hard/small" critique

6. **Add HMC (sleep staging) to within_strict arm** — UPGRADED FROM LOWER PRIORITY
   (needed to lift within_strict N for path A)
   - 151 subjects, 5-class, subject-level split standard
   - Adds 3rd within-strict dataset → n = 6 → 9
   - Cost: ~4–6 h FT per FM (12–18 h total); need to download HMC
   - **Most important single addition** — currently within_strict CI is
     [−0.50, +1.00]; adding HMC should collapse this substantially.

7. **ADFTD 3-class (AD/FTD/CN) to match literature** — UPGRADED
   - Our current ADFTD is binary; literature uses 3-class
   - Adds comparability with EEG-FM-Bench ADFTD numbers
   - Cost: 3 seeds × 3 FMs × 2 modes = 18 runs ≈ 8 h
   - Lower priority than TUAB + HMC; include if time permits.

8. **Literature gap table** ✅ DRAFTED 2026-04-18
   - `paper/figures/source_tables/table1_benchmark_gap.md` — main table
   - `paper/figures/source_tables/table1_excluded.md` — exclusion reasoning
   - Source: EEG-FM-Bench (arXiv 2508.17742) + our exp_30

9. **Figure + table plan for TNSRE** ✅ FINALISED 2026-04-18
   - Target venue: IEEE TNSRE (journal, not conference)
   - Main: **5 figures + 4 tables** (was 8+5 in earlier draft — refined for TNSRE scope)
   - Supp: 4 items (excluded benchmarks, seed-noise histograms, Stress degeneracy, HHSA + raw seeds)
   - Full detail in `docs/paper_strategy_sdl_critique.md` §4b
   - Execution order: Fig 4 (C2 main) → Fig 3 (variance atlas) → Fig 2 (gap) → Fig 1 (schematic) → Tables → Supp
   - Production estimate: ~18–20 h figure/table work + tex writing

10. **Paper drafting (gated on F2 path A confirmation OR explicit (C) acceptance)**
    - Do NOT start tex rewrite until path decision is made and data collected
    - Target sections: swap Abstract + §1 Intro + §3 overview + Discussion
    - Keep §3.1 variance atlas structure; replace §3.2/§3.3 with exp_30 C2 figure

11. **Rewrite Fig 6 drift verdict logic** (blocked by N-F22)
    - Current labels `rescue_consistent_with_subject_shortcut` / `rescue_consistent_with_label_signal` assume `subject_frac ↑ = shortcut`, which mis-classifies EEGMAT/SleepDep healthy FT as shortcut.
    - Option A: make verdict CV-regime-conditional (within-subject paired → `subject_frac ↑` = healthy reference encoding).
    - Option B: remove verdict labels entirely, let arrow direction + cell position in the 2×2 speak.
    - Source file: `results/studies/representation_drift/lp_vs_ft_stress.json` build script.

12. **Add EEGMAT architecture ceiling to Fig 6 left** (cell-conditional architecture claim)
    - Current Fig 6 left shows classical + non-FM + FM all collapsing on Stress (0.43–0.58 ceiling). Under the task-substrate alignment framing, needs EEGMAT counterpart showing ceiling rises to 0.70+ for the same architectures — supports "architecture irrelevance is alignment-conditional".
    - Check if `results/studies/exp15_nonfm_baselines/sweep/*` has EEGMAT runs; if not, needs 2 archs (eegnet, shallowconvnet) × 3 seeds on EEGMAT (~2 h).

13. **Expand Fig 4 with SleepDep trajectory row** (gated on paired protocol check)
    - Needs SleepDep paired windows (same subject, rested vs sleep-deprived) to compute direction-consistency.
    - Check `pipeline/sleepdep_dataset.py` (or equivalent) for paired structure.
    - If protocol supports it, compute within-subject FT trajectory analogue to EEGMAT rest→task. Expected: weak-aligned (SleepDep) → low / negative dir_consistency across all FMs.

14. **Write 2×2 axis text in Methods §3.1 + Intro** (axis renamed 2026-04-23)
    - Intro: state the axis taxonomy upfront — row = CV regime (within-subject paired vs subject-label trait); column = task-substrate alignment (strong vs weak). Cite DEAP/SEED/DREAMER/DASPS as additional instances of (subject-label × weak-aligned) cell to establish prevalence (no new experiments needed).
    - Methods §3.1: map the four datasets onto the 2×2 with operational definitions — alignment classified by permutation null (`p ≤ 0.05` strong / `p > 0.1` weak); regime classified structurally by label design.
    - Zero experimental cost; ~500 words.

15. **exp31 aug_overlap audit — decision gate** ✅ DONE 2026-04-22
    - `pipeline/dataset.py:196` hard-codes aug_overlap to `label==1` — only correct when label==1 is the window-level minority.
    - 15-seed sweep (LaBraM/CBraMod/REVE × EEGMAT/ADFTD × aug=0.5/no-aug/baseline) found no BA change warranting a recipe switch. See `results/studies/exp31_aug_audit/REPORT.md`.
    - Paper methods caveat: aug hardcoded to label==1 is benign across our datasets (within seed noise); no action.

16. **ADFTD recipe switch cascade** ❌ CANCELLED (not triggered by #15)

17. **Fig 3 → 4-panel (2×2) null** ✅ DONE 2026-04-23
    - 30-seed null × 4 datasets complete; real FT clears null on EEGMAT (p=0.03) and ADFTD (p=0.03, subject-level perm), stays inside null on Stress (p=0.32) and SleepDep (p=0.19).
    - Output: `paper/figures/main/fig3_honest_evaluation_4panel.{pdf,png}` via `scripts/figures/build_fig3_perm_null_4panel.py`.
    - Axis labels: rows = within-subject paired / subject-label trait; columns = strong-aligned task / weak-aligned task.

18. **Task-substrate alignment axis pivot** — partial ✅ 2026-04-23
    - ✅ Memory: `project_regime_framing.md` → `project_task_substrate_alignment.md`; MEMORY.md index updated.
    - ✅ `docs/findings.md` §Central thesis + 2×2 table + axis definitions rewritten (strong/weak aligned).
    - ✅ `docs/paper_outline.md` header + 2×2 + §1.3 + §1.5.1 + §3.1.2 rewritten.
    - ✅ Fig 3 builder + regenerated figure use "strong-aligned task / weak-aligned task" column labels.
    - 🟡 Remaining: sweep rest of `docs/paper_outline.md` for stale "regime" / "signal coherence" language (sections 4.x, Discussion, Supplementary); propagate to eventual tex writeup when §1 Intro is drafted.

## Lower priority — defer until paper direction locked

10. **HP search on between-arm FT** (D3 resolution)
   - Under current Type C framing, ΔBA magnitude is not claim-load-bearing
   - Only run if reviewers push back on "your FT is undertuned"
   - HP grid: lr ∈ {1e-5, 5e-5, 1e-4}, wd ∈ {0, 0.01, 0.1}, drop ∈ {0, 0.1}
   - Cost: 9 × 3 FMs × 3 datasets = 81 runs. Heavy.

## HHSA / WSCI pipeline (demoted — C4 falsified)

- HHSA contrast does not correlate with ΔBA at n=9 (ρ=−0.26, CI
  wide). **Will NOT be the Type C paper's main evidence.** Lives in
  appendix as signal-side documentation.
- **HHSA L1 cache rebuild (N_ENSEMBLES=100)** still useful for the
  signal-documentation role but no longer gates paper submission.
- WSCI analysis similarly demoted.

## On hold — not pursuing unless narrative shifts

- **Extend perm null to 20 perms** — pre-critique claim; Type C paper
  doesn't need this.
- **EEGNet re-sweep** — not relevant under Type C framing.

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
- Connection to N-F12 (GRL/DANN/LEAD negative results) — motivates C2's
  prescription for subject-invariant pretraining
- Split protocol inconsistency in EEG-FM-Bench (stratified ADFTD vs
  subject-level TUAB) — direct evidence for the critique
- Brain4FMs / EEG-FM-Bench / AdaBrain-Bench convergence (N-F11)
- Stress power floor (G-F08 cuDNN swing + N-F19 null-indistinguishable
  + F-D.2) — benchmark-design caveat for the field; now in service of
  Type C critique not Type A positive claim
- **FT<LP prior art** (added 2026-04-18 after literature audit):
  BENDR (Frontiers 2021) reports frozen+linear > full FT on 4/5 datasets
  under LOSO. AdaBrain-Bench (2025) documents EEGPT LP 47.9 vs FT 25.8
  on BCI-IV-2a. EEGPT chooses LP as headline metric. Our negative
  ΔBA in 7/9 between cells rediscovers this pattern with correlation-
  level mechanism (subject_id_ba ~ FT degradation magnitude).
- Our 5-fold subject CV × 3 seeds is MORE rigorous than community
  standard (fixed split × 3-5 seeds). Our ±3 pp uncertainty includes
  data-sampling variance most papers structurally cannot report.
- Within-arm null result (C3 falsification) — honest limitation + open
  question for future benchmark design
