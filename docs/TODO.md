# TODO — Current Priorities

**Last updated**: 2026-04-26 (LaBraM input-norm aligned to /100; CBraMod head + aggregation alignment options noted)

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

2. **CBraMod head + window-aggregation alignment** (audit 2026-04-26 — verified norm matches; head + agg do not)
   - **Norm**: ✅ already aligned — original `datasets/*_dataset.py` does `data/100`; our `cbramod_extractor.py:102` does same. No code change needed.
   - **Backbone (`model.py`, `criss_cross_transformer.py`)**: ✅ mathematically identical to original; cosmetic-only differences (deleted dead helpers `_get_seq_len`, `_detect_is_causal_mask`, `_generate_square_subsequent_mask`, telemetry, unused `is_causal` plumbing).
   - **Head — divergence #1**: original `models/model_for_stress.py` uses 4 selectable classifiers (default `avgpooling_patch_reps` = `Rearrange + AdaptiveAvgPool2d → Linear(200, 1)` + BCE); we use `Linear(200,128) + GELU + Dropout + Linear(128,2)` + multiclass CE.
   - **Aggregation — divergence #2**: original is **strict per-window** (each window independently → 1 logit, BCE per window, no recording-level vote). Our FT path does per-window classify → majority vote → recording label; our LP path does feature avg-pool over windows → single classify per recording. Neither matches original exactly.
   - **Action (when ready)**: add config flags
     - `--cbramod-head {mlp,linear}` — `linear` reproduces paper's single `Linear(200,2)` head (keep CE + 2-class for our binary stress framing rather than copying their BCE+1-output)
     - `--cbramod-agg {feature-pool,window-vote,per-window}` — `per-window` = strict paper alignment (no recording-level aggregation, evaluate per window); other two = our current modes
   - **Block scope**: not blocking the paper — current head/agg are reasonable design choices, not bugs. Run only if reviewers push back or we want a "paper-strict" ablation row. Cost: head change is trivial (one Linear); agg change requires touching `evaluate_recording_level()` and feature-cache pooling.
   - **Same audit for LaBraM** (open): LaBraM original head is `nn.Linear(embed_dim, num_classes)` with `init_scale=0.001`; our `head_cls` is also 2-layer MLP (head_hidden=128). If we add the CBraMod flags, mirror them for LaBraM (`--labram-head {mlp,linear}`) for symmetry.

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
