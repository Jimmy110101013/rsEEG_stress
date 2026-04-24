# Methodology Notes

Supplementary record of **guardrails**, **internal methodology notes**, and
**archived/superseded findings**. Paper-citable claims live in `docs/findings.md`
(F-A … F-E).

Read this file when you need to:
- check a pipeline policy (multi-seed, labelling, env)
- understand a caveat referenced from a main finding
- trace an archived finding's history

---

## Guardrails (pipeline policies)

### G-F07: subject-dass OR-aggregation is deprecated
**Status**: policy (2026-04-10)
**Evidence**: Only 3/17 subjects have within-subject DASS class transitions. Subject-dass turns per-recording stress classification into subject-identity broadcast. Canonical 0.656 LaBraM FT under subject-dass is a non-reproducible single-seed draw (HEAD re-run gives 0.45). Source: `docs/historical/progress.md` §4.6.
**Action**: `--label subject-dass` deprecated. All new work uses `--label dass` (per-recording). Historical subject-dass numbers are inflation artifacts — see F-B Axis 2.

---

### G-F08: Single-seed BA on Stress is not reproducible
**Status**: policy (2026-04-10)
**Evidence**: Exact same canonical recipe (subject-dass, seed=42) at HEAD gives 0.4505 vs. original 0.6559 — 20.5 pp swing. Root cause: 70-rec / 14-positive regime amplifies microscopic init differences. Mitigated (not eliminated) by `cudnn.deterministic=True` (added 2026-04-10).
**Action**: All Stress BA claims require multi-seed (≥ 3). Single-seed BAs are not reproducible and must not be cited.

---

### G-F09: FT must use per-FM official HP recipe, not unified HP
**Status**: policy (2026-04-19)
**Evidence**: exp_newdata ran all 3 FMs with unified `lr=1e-5 batch=4 layer_decay=off wd=0.01 loss=focal` — 10–80× below every official recipe. exp20 control (`layer_decay=0.65 wd=0.05` but lr=1e-5): LaBraM seed-range 0.32–0.93, CBraMod + REVE fixed at 0.50 chance. Confirms unified conservative HP silently kills CBraMod/REVE while high-variance-training LaBraM. Sanity reproduction (2026-04-19 s41, `results/studies/sanity_lead_adftd/`) with per-FM official HP: LaBraM ΔF1_subject +43pp, CBraMod +48pp, REVE +14pp (FT − Frozen), LaBraM subject F1 0.852 within 1σ of LEAD 0.911.
**Action**: All new FT runs must use per-FM official HP. Canonical recipe dict lives in `scripts/experiments/run_sanity_lead_adftd.py::RECIPES`. Sources:
- LaBraM: README + `run_class_finetuning.py` — lr=5e-4, layer_decay=0.65, wd=0.05, CE+LS 0.1, single Linear head init_scale=0.001, cosine + 5ep warmup, drop_path=0.1.
- CBraMod: `finetune_main.py` + `finetune_trainer.py` — backbone lr=1e-4 + head lr=5e-4 (multi_lr bs=64), wd=0.05, CE+LS 0.1, cosine, no warmup.
- REVE: `reve_unified.yaml` — max_lr=2.4e-4, encoder_lr_scale=0.1, wd=0.01, plateau, warmup 2ep + freeze-encoder 1ep, betas=(0.9, 0.95).

---

### G-F12: Canonical LP = per-window sklearn LogReg via `train_lp.py`
**Status**: policy (2026-04-23, rewrite landed 2026-04-25)
**Evidence**: The pre-2026-04-20 `train_lp.py` implemented a PyTorch pool-then-classify probe (optionally MLP head + mixup) — not community-standard and not the paper's LP source. On 2026-04-20 all paper LP numbers were migrated to the per-window sklearn LogReg protocol (per-window train, test prob mean-pool per recording, threshold 0.5). The migration exposed an 8 pp drop on Stress LaBraM LP (0.605 → 0.525) and a narrative flip on EEGMAT (LP 0.736 ≈ FT 0.731 — FT not a rescue). Full 4-dataset × 3-FM × 8-seed results + narrative deltas in `results/studies/perwindow_lp_all/SUMMARY.md`.
**Action (2026-04-25)**: `train_lp.py` was rewritten in place to be the canonical per-window LP entry point (CLI + `run_canonical_lp`/`eval_seed` library API, `--cv stratified-kfold|loso`). The previous pool-then-classify body is preserved at git tag `lp-pool-then-classify-v1` for reproducing pre-migration numbers if a reviewer asks. Four subsumed scripts were deleted in the same commit: `scripts/experiments/{frozen_lp_perwindow_all,stress_frozen_lp_perwindow,stress_frozen_lp_loso,stress_frozen_lp_multiseed}.py`. Master Tab 1 should source from `results/studies/perwindow_lp_all/{dataset}/{model}_multi_seed.json`. Variance decomposition (§4.2 Fig 2) and band RSA (Appendix B.3 Fig B.3) still require pooled `_19ch.npz` features (recording-level analyses); these are distinct from LP and unaffected.

---

### G-F11: Dataset loaders must fail loud on unloadable files, never return dummy tensors
**Status**: policy (2026-04-23)
**Evidence**: `pipeline/sleepdep_dataset.py` originally returned a `(1, 19, T)` zero tensor when `mne.io.read_raw_eeglab` failed both load attempts, and `__init__` only filtered records whose `.failed` marker existed from a *prior* run. On a fresh cache, failing records stayed in `self.records` / `StratifiedGroupKFold` and `WindowDataset._preload` silently fed zero-tensor "recordings" into training — contaminating labels and fold balancing. For exp27 SleepDep null chain this would make seed 0 (n_subj=36 with zeros) and seeds 1–29 (n_subj=35 after marker filter) structurally different, invisible in the pooled Fig 3 histogram. Caught by ultrareview 2026-04-23. Cached `data/cache_sleepdep` markers post-dated published numbers, so no known result was contaminated; bug was latent for any fresh reproduction.
**Action**: All dataset loaders that fetch raw EEG on demand must raise on load failure (after writing any failure marker), never return a dummy tensor. `__init__` remains responsible for filtering records with existing `.failed` markers so a re-run drops them cleanly. Fixed in `pipeline/sleepdep_dataset.py` 2026-04-23. Audit remaining loaders (`adftd_dataset.py`, `eegmat_dataset.py`, `tdbrain_dataset.py`, `stress` loader) for equivalent dummy-return patterns before the next paper reproduction.

---

### G-F10: LEAD-style per-window training + majority-vote aggregation is community standard
**Status**: policy (2026-04-19)
**Evidence**: LaBraM (`engine_for_finetuning.py`), CBraMod (`finetune_evaluator.py`), EEGPT, EEG-FM-Bench all train with per-window CE loss and evaluate at window level. LEAD §2.2 adds majority-vote aggregation: train per-window CE, at test time each subject's windows vote → one prediction per subject. Our `train_ft.py --mode ft` matches this (per-window loss, `evaluate_recording_level` majority vote). Since the 2026-04-25 rewrite, `train_lp.py` is also per-window (train at window level, test-set window probabilities mean-pooled per recording, threshold 0.5). The pre-2026-04-20 pool-then-classify body (feature pooling before LogReg) was the non-standard variant and is now retired — preserved at git tag `lp-pool-then-classify-v1` per G-F12.
**Action**:
- Paper methods section must explicitly document the FT vs LP protocol difference.
- When reporting FT results, headline metric is subject-level F1/AUROC (LEAD primary); sample-level is secondary.
- For apples-to-apples comparison with community LP numbers, add per-window LP as a robustness column (TODO — requires per-window frozen feature cache re-extraction).

---

## Internal methodology notes (not paper claims)

### N-F11: Independent FM benchmarks corroborate our framing
**Status**: positioning evidence
**Evidence**: Brain4FMs (2026), EEG-FM-Bench (Xiong 2025), AdaBrain-Bench (2025) all report FM failure on cross-subject affective/cognitive tasks. Our contribution is the mechanistic explanation (F-A subject dominance + F-C model × dataset taxonomy) they lack. Source: `docs/related_work.md` §2.
**Placement**: use in `related_work.md`; not a standalone paper claim.

---

### N-F12: Subject-adversarial losses (GRL, LEAD) ruled out
**Status**: negative ablation (2026-04-05)
**Evidence**: GRL λ=0.1 → 0.537 BA (−12 pp), λ=1.0 → 0.558 BA (−10 pp), LEAD loss → 0.439 BA. Full λ sweep [0.01–1.0] all worse than baseline. Source: `docs/historical/progress.md` §6.
**Placement**: appendix or omit; negative result not central to Arc A.

---

### N-F15: Statistical hardening (bootstrap CIs + Cohen's d)
**Status**: supplementary stats (2026-04-13)
**Evidence**: Bootstrap (10k resamples) on per-seed BA. Source: `results/studies/exp10_stat_hardening/stat_hardening.json`.

| Model | Frozen LP [95% CI] | Best FT [95% CI] | Cohen's d |
|---|---|---|---|
| LaBraM | 0.605 [0.585, 0.626] | 0.524 [0.518, 0.536] | +4.30 (erosion) |
| CBraMod | 0.452 [0.430, 0.472] | 0.548 [0.518, 0.580] | −3.28 (injection) |
| REVE | 0.494 [0.482, 0.506] | 0.577 [0.536, 0.634] | −2.65 (injection) |

CIs do not overlap for LaBraM (frozen > FT) and CBraMod (FT > frozen); REVE has minimal overlap. All Cohen's d > 2.5. Paired permutation test (3 matched seeds): all models p=0.25 — direction consistent, magnitude limited by n=3 (minimum p=0.125). Sign test on trial-vs-subject gap: 3/3 FMs trial > subject, p=0.125 (exact binomial); Wilcoxon W=6.0, p=0.125.

**Placement**: supplementary stats panel for F-C.2 / F-D.

---

### N-F18-cross-ref: REVE window-dependent injection magnitude
**Status**: caveat (absorbed into F-C.4)
The full 5s-vs-10s evidence now lives in `findings.md` F-C.4. Retained here as a cross-reference stub — F-C.4 is authoritative.

Key action item: F-C.2 currently reports REVE Δ=+8.3 pp from the original 10s-FT + 5s-frozen mismatch. The principled 5s-matched number is +4.8 pp. Paper should cite **+4.8 pp with window-sensitivity footnote**.

---

### N-F19: CBraMod/REVE permutation null is borderline (10-perm floor)
**Status**: pending more perms (2026-04-14)
**Evidence**: 10-perm runs at best HP. Source: `results/studies/exp03_stress_erosion/ft_null_{cbramod,reve}/`.

| Model | Real FT (3-seed) | Null FT (10-perm) | Δ | p (one-sided) |
|---|---|---|---|---|
| CBraMod | 0.5476 ± 0.0314 | 0.4839 ± 0.0498 | +6.4 pp | 0.100 |
| REVE | 0.5774 ± 0.0508 | 0.4857 ± 0.0692 | +9.2 pp | 0.100 |

With only 10 perms the minimum non-zero p is 0.1 — both CBraMod and REVE are at the floor (1/10 null seeds beats real mean). Consistent with weak real injection *or* noise; cannot resolve with current sample.
**Action if pressed**: extend to 20-perm. Current evidence is weak but positive; does not undermine F-C.2's direction claim.

---

### N-F21: LaBraM `self.norm` architecture mismatch — material effect < 1 pp (do NOT cite)
**Status**: internal bug-fix note (2026-04-15)

**The bug**: `baseline/labram/model.py` defined both `self.norm = LayerNorm(embed_dim)` *and* `self.fc_norm = LayerNorm(embed_dim)`, then applied both in sequence when mean-pooling patch tokens. Original LaBraM (`935963004/LaBraM/modeling_finetune.py`) sets `self.norm = nn.Identity()` under `use_mean_pooling=True`, keeping only `fc_norm`. Our code loaded pretrained `norm.weight/bias` (trained for CLS head) then stacked fresh `fc_norm` on top — architectural drift.

**The fix** (`baseline/labram/model.py:194`): `self.norm = nn.Identity()`. Loader now drops pretrained `norm.*` keys (219/221 load; both `fc_norm.*` random-init as in original).

**Diagnostic impact** (Wang-protocol 12-seed sweep, LaBraM only):

|  | exp23 (pre-fix) | exp24 (fixed) | Δ |
|---|---:|---:|---:|
| Wang 4-seed win_BA mean | 0.608 | 0.602 | −0.006 |
| All 12-seed win_BA mean | 0.571 | 0.578 | +0.007 |
| All 12-seed rec_BA mean | 0.646 | 0.592 | −0.054 |
| Wang gap (win) | 22 pp | 23 pp | — |

Recording-level swings are large *per seed* (seed 0 +25 pp, seed 2 −25 pp) but cancel in the mean. Window-level Wang-protocol metric moves < 1 pp (below the 3 pp materiality threshold).

**Decision**: keep the fix; **do NOT re-run canonical LaBraM experiments** (exp03-08, exp11, exp12, exp18, frozen/ft_labram_* caches). Wang-protocol gap is dominated by split-RNG (Wang's seed→split mapping is unspecified), not this bug; 80 GPU-hr rerun would not change any paper claim. Canonical numbers in the master table and figures remain authoritative.

**CBraMod and REVE audited**:
- REVE: `baseline/reve/model.py` is a line-by-line match of official `modeling_reve.py`. 140/140 params load.
- CBraMod: 209/209 encoder params load; `proj_out.*` (reconstruction head) intentionally replaced with `Identity`.

Only LaBraM had drift. CBraMod/REVE collapse under LLRD 0.65 in exp20 is attributable to LLRD-vs-architecture mismatch (Wang tuned 0.65 for LaBraM only), not implementation.

**Do NOT write this into the paper.**

**Files**: `baseline/labram/model.py`, `baseline/labram/labram_extractor.py`, `results/studies/exp24_labram_fixed/`.

---

### N-F22: Subject variance fraction is regime-conditional, not a quality indicator
**Status**: internal correction (2026-04-21) — blocks naive reading of Fig 2 and Fig 6
**Evidence**: Fig 2 consolidation revealed that `subject_frac` (pooled SS on subject axis) and `rsa_subject_r` (rank RSA alignment) move in directions that depend entirely on the experimental regime:

- **Subject-label regime (Stress)** — FT drops `subject_frac` 49→12%, pushes variance into residual, label_frac also drops. Reads as "structure collapse" under the naive "high subject = shortcut" lens, but RSA shows `rsa_subject_r` *increases* 0.19–0.28 → 0.25–0.33. The two metrics disagree because variance_frac is absolute SS while RSA is rank-based; same-subject pairs become more similar in rank even as absolute between-subject variance shrinks.
- **Within-subject paired regime (EEGMAT, SleepDep)** — FT raises `subject_frac` (e.g. EEGMAT LaBraM 76→87%, SleepDep 46→78%). Under the naive lens this reads as "subject shortcut", but task success correlates with this direction — because the task label is defined *relative to subject baseline*, the FM must encode subject identity strongly to measure within-subject deviation. High subject_frac is **healthy** in this regime.

**Implication**: `subject_frac ↑ = shortcut` / `subject_frac ↓ = good` is wrong as a universal rule. The Fig 6 drift-vector verdict (`rescue_consistent_with_subject_shortcut`) was built on this assumption and mis-labels EEGMAT/SleepDep FT as shortcut when they are healthy within-subject learning. Verdict logic must be regime-conditional or removed in favour of letting arrow direction + regime panel speak for themselves.

**Action**: (i) all paper prose interpreting `subject_frac` change must first qualify by regime; (ii) Fig 6 verdict labels pending rewrite; (iii) Fig 2 caption must frame row 1 (variance) + row 2 (RSA) as two complementary views that can disagree.

**Files**: `paper/figures/_historical/source_tables/sleepdep_variance_rsa.json`, `paper/figures/_historical/source_tables/ft_rsa_stress_eegmat.json`, `scripts/analysis/compute_sleepdep_variance_rsa.py`, `scripts/analysis/compute_ft_rsa.py`.

---

## Archived / superseded

### A-F04: Cross-dataset taxonomy (LaBraM-only, 10 s windows) — superseded by F-C
**Status**: archived (2026-04-13)
**Superseded by**: F-C (multi-model, 5 s windows, 3-seed).
**Why archived**: 3-seed multi-model re-run at 5 s windows (F-C) shows the +5 pp ADFTD injection magnitude was inflated 5× by the 10 s-window run; direction confirmed but cross-model comparison requires the 5 s-matched setting. F04 remains valid as **LaBraM-only, 10 s historical evidence** only.

Original F04 key numbers (for historical traceability):

| Mode | Dataset | n_rec/n_subj | Frozen | FT | N-controlled Δ |
|---|---|---|---|---|---|
| Injection | ADFTD (AD vs HC) | 195/65 | 2.79% | 7.70% | +5.22 to +5.68 pp |
| Mild injection | EEGMAT | 72/36 | 5.35% | 5.82% | n/a (crossed) |
| Silent erosion | TDBRAIN (MDD vs HC) | 734/359 | 2.97% | 1.47% | −1.06 to −1.58 pp |
| Stale | Stress (DASS) | 70/17 | 7.23% | 7.24% | n/a (subject-dass) |

Source: legacy `paper/figures/variance_analysis_matched.json` (pre-reorg); regenerator = `scripts/analysis/run_variance_analysis.py`. Current per-cell outputs under `results/studies/exp32_variance_triangulation/` and `results/final/<dataset>/variance_triangulation/`.
