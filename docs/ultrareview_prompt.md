# Ultrareview Prompt — Pipeline Audit for FM Underperformance

**Context for ultrareview agents**: The UCSD_stress paper compares EEG Foundation Models (LaBraM, CBraMod, REVE) against classical baselines (EEGNet, ShallowConvNet, DeepConvNet, EEGConformer) across 4 datasets (Stress/UCSD, EEGMAT, ADFTD, SleepDep). Current finding: FMs often **underperform** LP / classical baselines on subject-level CV. The advisor (professor) suspects a **pipeline bug** — i.e. the FMs are being crippled by a silent data/config mistake rather than being genuinely weak.

**Your job**: adversarially audit the codebase for bugs, inconsistencies, or unfair comparisons that would systematically disadvantage FMs vs baselines, or that would break the validity of the headline claims. Report each finding with: evidence (file path + line number), affected experiment(s), severity (would it flip a claim?), and suggested fix.

---

## Hypothesis to stress-test

> *"FMs look bad because the pipeline underfeeds them — wrong input scale, wrong preprocessing, wrong hyperparameters, or unfair comparison to baselines — not because FMs are genuinely weak on these tasks."*

Treat this as the null; try to falsify it by finding concrete pipeline defects. If you can't falsify it after checking the directions below, report that too.

---

## Directions to check (ranked by suspicion)

### 1. Input normalization (highest suspicion)
CLAUDE.md §2 states `--norm` is **per-model**:
- `labram` → `zscore`
- `cbramod` → `none` (extractor internally does `x / 100` in `baseline/cbramod_extractor.py:102`)
- `reve` → `none` (µV-scale)
- classical baselines → `zscore`

**Audit every training / feature-extraction / LP / FT script**:
- `train_ft.py`, `train_trial.py`, `train_lp.py`
- `scripts/experiments/*.sh` and `scripts/experiments/run_*.py`
- `scripts/features/*`

For each `(dataset × FM × experiment)` combination, verify the `--norm` flag matches the FM's requirement. A double-zscore on CBraMod (already divided by 100) silently collapses input to ~0. Report any mismatch.

Also verify: what is the **unit** the dataset loader produces (µV / mV / normalized)? Does it match each FM's pretraining assumption? HNC is mV-scale — confirm this does not leak into UCSD_stress loaders.

### 2. CBraMod `/100` hardcoded scaling × FOOOF ablation
Recent finding (memory obs 3363–3365): CBraMod's hardcoded `x/100` interacts with FOOOF amplitude manipulation to produce artifactual results on EEGMAT. Check:
- Does the same artifact affect **ADFTD / SleepDep / Stress** FT runs, not just EEGMAT / FOOOF ablation?
- Is the `/100` applied **before or after** any per-recording z-scoring in the dataset layer?
- Does any script pass already-scaled input to CBraMod, producing a double-scaling bug?

### 3. Cache / window / split consistency
Recent bugs caught: w5.0 vs w10.0 mismatch (obs 3350–3353), SleepDep `.failed` markers vs EEGMAT having none (obs 3378), ADFTD accidentally using `n_splits=3` instead of 1 (obs 3359–3361).

**Exhaustive sweep**: for every `(dataset × FM × script)`:
- `window_sec` used for training vs `window_sec` encoded in cache filename
- `n_splits` parameter
- cache path construction (`pipeline/dataset.py`, `pipeline/eegmat_dataset.py`, ADFTD / SleepDep loaders)
- Fallback logic in `WindowDataset` (obs 3353, 3380): does on-demand cache generation **silently drop** samples when a record fails? Does it raise? Report any silent-failure path.
- Failure marker system: EEGMAT has none (obs 3378) — could it be silently skipping records the same way SleepDep does?

### 4. Subject-level CV leakage
Primary metric is `StratifiedGroupKFold(5)` with `groups=patient_id`.

- Verify `patient_id` (or equivalent) is correctly extracted for **every** dataset. Schemas differ across ADFTD / SleepDep / EEGMAT — confirm `groups=` is always the true subject identifier, not a recording ID.
- `RecordingGroupSampler` + `--aug-overlap 0.75` on minority class: verify augmented windows inherit the correct group label so they cannot leak across train/val folds.
- Global prediction pooling (train_ft.py): confirm pooling is **within fold, within subject**. A pool that crosses fold boundaries would inflate metrics; a pool that crosses subjects within the test set would change the metric's meaning.

### 5. Baseline fairness (professor will ask this first)
For the FM-vs-baseline comparison to be valid, classical baselines must receive the **same treatment** as FMs:
- Same early-stopping policy (SMA-3, patience 15)
- Same `--aug-overlap` minority-class augmentation
- Same `cudnn.deterministic=True`, same seed pool (≥3 seeds)
- Same CV splitter (`StratifiedGroupKFold(5)` by patient_id)
- Same class balancing (pos_weight / sampler / weighted loss)
- Same LR schedule, same optimizer family
- Same input normalization convention (zscore — matches Lawhern/Schirrmeister defaults)

**Check every baseline training script** and compare side-by-side with the FM training scripts. Any asymmetry that favors baselines is the story.

### 6. FM-specific preprocessing vs pretraining conditions
Each FM was pretrained with specific preprocessing:
- LaBraM: 0.1–75 Hz bandpass, 200 Hz, specific channel ordering
- CBraMod: 0.3–75 Hz, 200 Hz
- REVE: check repo for patch config + bandpass

**Verify** per-dataset preprocessing matches each FM's pretraining. Using one shared preprocessing pipeline for all FMs is a red flag.

**Channel ordering / montage**: FM pretraining typically assumes 10-20 canonical order. Does the dataset loader **re-order** channels to match, or pass raw-file order? Wrong channel order silently destroys attention / patch embeddings that learned position-specific features. Stress is 30-channel, ADFTD/EEGMAT 19-channel, TDBRAIN 26→19 — all different schemes.

### 7. Head, loss, optimizer, LR schedule
- `src/model.py` `DecoupledStressModel`: is the classifier head (init, dropout, LayerNorm) identical across FMs? A head tuned for LaBraM's 200-dim may underserve REVE's 512-dim.
- Class imbalance handling (14/70 positive on Stress): is it applied symmetrically to FMs and baselines?
- **Layer-wise LR decay**: LaBraM and CBraMod official FT recipes use LLRD for the backbone with a higher LR on the head. If `train_ft.py` applies a single flat LR to the whole model, FMs are being underoptimized. This is a classic FM-vs-scratch trap and would explain FM underperformance on its own.
- Warmup schedule, weight decay, AdamW betas — compare against each FM's official recipe (LaBraM paper + repo; CBraMod repo).

### 8. Permutation null correctness
`train_ft.py --permute-labels <seed>` with `--permute-level {recording,subject}` (obs 3379, 3381):
- Does subject-level permutation truly shuffle labels within subject-group structure, preserving within-subject correlations?
- Is the null-run seed pool **independent** of the real-run seed pool?
- Does the permutation happen **before** CV split construction (correct) or **after** (incorrect — would leak structure)?

### 9. Feature extraction / Linear Probing consistency
- `extract_pooled` in each FM extractor: which layer is pooled, and **how** (mean / CLS token / attention-weighted)?
- Is the pooling used for LP the same pooling used for FT? If LP pools layer L but FT pools layer L', the comparison is confounded.
- Frozen LP features cache: confirm cache is regenerated when normalization or window config changes.

### 10. Statistical reporting
- Every `FT < LP` citation in `docs/findings.md` and `docs/paper_outline.md` must annotate the protocol (subject-level CV vs trial-level vs LOSO, number of seeds). Mixed protocols would invalidate the claim.
- Architecture ceiling table (memory S1529, obs 3385): confirm FM and baseline numbers are on matched CV scheme + matched seeds. If FMs are reported as single-seed and baselines as multi-seed (or vice versa), the comparison is unfair.

---

## Out-of-scope for this review

- Narrative / framing in the paper outline
- Figure aesthetics
- Writing quality of prose

Focus on **code correctness and experimental validity** only.

---

## Reporting format

For each finding, produce:

```
### [SEVERITY] <one-line title>
**Location**: <file>:<line>
**Affected experiments**: <list of dataset × FM pairs>
**Evidence**: <code snippet or config>
**Impact on claims**: <which findings.md claim flips if this is a bug?>
**Suggested fix**: <concrete change>
**Confidence**: high / medium / low
```

Severity scale:
- **CRITICAL**: would flip a headline claim (F-A … F-E) if fixed
- **HIGH**: would change a number by more than the seed-noise floor (±5–10 pp on Stress)
- **MEDIUM**: correctness issue but unlikely to change claims
- **LOW**: code smell, inconsistency, or doc drift

End with a **bottom-line verdict**: is the hypothesis "pipeline bug is hiding FM capability" supported, refuted, or inconclusive?

---

## Quick-access context

- CLAUDE.md (§2 Architecture, §4 Guard rails) — primary reference
- `docs/findings.md` — the 5 claims being defended
- `docs/methodology_notes.md` — known guardrails (G-F07, G-F08)
- `docs/TODO.md` — active work
- Key scripts: `train_ft.py`, `train_lp.py`, `train_trial.py`, `pipeline/dataset.py`, `pipeline/eegmat_dataset.py`, `baseline/*_extractor.py`, `src/model.py`, `scripts/experiments/*`
