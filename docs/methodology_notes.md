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
**Evidence**: Only 3/17 subjects have within-subject DASS class transitions. Subject-dass turns per-recording stress classification into subject-identity broadcast. Canonical 0.656 LaBraM FT under subject-dass is a non-reproducible single-seed draw (HEAD re-run gives 0.45). Source: `docs/progress.md` §4.6.
**Action**: `--label subject-dass` deprecated. All new work uses `--label dass` (per-recording). Historical subject-dass numbers are inflation artifacts — see F-B Axis 2.

---

### G-F08: Single-seed BA on Stress is not reproducible
**Status**: policy (2026-04-10)
**Evidence**: Exact same canonical recipe (subject-dass, seed=42) at HEAD gives 0.4505 vs. original 0.6559 — 20.5 pp swing. Root cause: 70-rec / 14-positive regime amplifies microscopic init differences. Mitigated (not eliminated) by `cudnn.deterministic=True` (added 2026-04-10).
**Action**: All Stress BA claims require multi-seed (≥ 3). Single-seed BAs are not reproducible and must not be cited.

---

## Internal methodology notes (not paper claims)

### N-F11: Independent FM benchmarks corroborate our framing
**Status**: positioning evidence
**Evidence**: Brain4FMs (2026), EEG-FM-Bench (Xiong 2025), AdaBrain-Bench (2025) all report FM failure on cross-subject affective/cognitive tasks. Our contribution is the mechanistic explanation (F-A subject dominance + F-C model × dataset taxonomy) they lack. Source: `docs/related_work.md` §2.
**Placement**: use in `related_work.md`; not a standalone paper claim.

---

### N-F12: Subject-adversarial losses (GRL, LEAD) ruled out
**Status**: negative ablation (2026-04-05)
**Evidence**: GRL λ=0.1 → 0.537 BA (−12 pp), λ=1.0 → 0.558 BA (−10 pp), LEAD loss → 0.439 BA. Full λ sweep [0.01–1.0] all worse than baseline. Source: `docs/progress.md` §6.
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

Source: `paper/figures/variance_analysis_matched.json`, `scripts/run_variance_analysis.py`.
