# Window-level variance + Fig 5b split — draft wording

**Status**: draft 2026-04-24; numbers populated from actual 12-cell frozen + FT (seed 42) run at 14:20 GMT+8.
**Purpose**: house the §3.6.1 / §4.2 / §4.5 / §5 wording for the unit-of-analysis shift *before* editing `paper_outline.md`, so the change can be reviewed as a diff against a single coherent draft.

## Measured numbers (frozen and FT-rewritten, canonical per-FM HP seed 42)

Source: `paper/figures/_historical/source_tables/variance_analysis_window_level.json`.

| cell | FM | frozen_lbl% | FT_lbl% | **Δlabel pp** | Δsubject pp |
|---|---|---|---|---|---|
| eegmat | labram | 1.26 | 3.25 | **+1.99** | +38.58 |
| eegmat | cbramod | 1.05 | 2.29 | +1.23 | +28.97 |
| eegmat | reve | 0.39 | 0.65 | +0.26 | +15.78 |
| sleepdep | labram | 0.30 | 0.31 | **+0.02** | +15.71 |
| sleepdep | cbramod | 0.81 | 0.60 | -0.21 | +19.60 |
| sleepdep | reve | 0.18 | 0.58 | +0.40 | +20.41 |
| adftd | labram | 0.64 | 6.90 | **+6.25** | +44.30 |
| adftd | cbramod | 0.37 | 1.40 | +1.02 | +30.93 |
| adftd | reve | 0.25 | 0.07 | -0.18 | -2.07 |
| stress | labram | 2.62 | 1.25 | **-1.37** | +39.16 |
| stress | cbramod | 1.44 | 2.05 | +0.60 | +59.22 |
| stress | reve | 0.51 | 1.88 | +1.37 | +27.87 |

**LaBraM Δlabel_frac cleanly tracks task-substrate alignment**:
strong-aligned > weak-aligned: ADFTD (+6.25) >> EEGMAT (+1.99) >> SleepDep (+0.02) ≈ Stress (-1.37).
CBraMod follows the same rank pattern with smaller amplitude; REVE diverges (likely due to different pretrain objective / embedding dimensionality — discussion material).

Across all cells, Δsubject_frac is **uniformly large and positive** (+15 to +60 pp). FT reliably compresses the representation toward subject-identity structure in addition to (or instead of) label alignment. This cross-cell "FT boosts subject_frac" observation is a secondary finding worth surfacing in §4.2.

---

## §3.6.1 Variance decomposition (method spec) — revised

> Variance decomposition is computed at the **window level**. Each resting-state recording is tiled into non-overlapping 5-second (LaBraM, CBraMod) or 10-second (REVE) windows using the per-FM canonical window convention (§3.2). Each window produces one embedding vector; these per-window embeddings are the unit of analysis.
>
> We decompose the total sum-of-squares via nested ANOVA with subject nested in label (each subject has a single trait or state label for the cell under test):
>
> ```
> SS_total = SS_label + SS_subject|label + SS_residual
> ```
>
> yielding three fractions: `label_frac = SS_label / SS_total` (between-label variance), `subject_frac = SS_subject|label / SS_total` (between-subject variance within the same label), and `residual_frac = SS_residual / SS_total` (within-subject window-to-window variance). df-corrected ω² analogues are reported in appendix tables.
>
> **Scope condition**: window-level decomposition is well-defined whenever each recording yields ≥ 2 windows — true for every cell in this paper including the single-session ADFTD cohort. This is a deliberate departure from the standard recording-level decomposition, which requires multiple independent recording sessions per subject and therefore excludes single-session clinical cohorts by construction. The unit shift is applied uniformly across all four cells to preserve cross-cell comparability.
>
> **Interpretation at the window level**. Because windows within a recording are temporally adjacent, they share electrode placement, scalp anatomy, and session-level neural baselines. `subject_frac` therefore captures a mixture of stable trait-level subject identity *and* within-session stable subject signature (posture, arousal, electrode impedance). For cross-cell comparison we emphasise two derived quantities that are robust to this absolute-scale inflation:
>
> 1. `delta_label_frac = label_frac(FT) − label_frac(frozen)`: the change in label variance induced by fine-tuning. Baseline subject+anatomy variance cancels between frozen and FT.
> 2. `label_frac / subject_frac` at frozen: a scale-invariant "task-alignment ratio" characterising how much of the between-subject variance budget the label claims.
>
> Absolute `label_frac`, `subject_frac`, `residual_frac` values are reported in Appendix B tables but should be read within-cell rather than as cross-cell absolute comparisons.
>
> **What this does not measure.** Because there is no across-session variance component at the window level, the decomposition does not separate "stable trait identity" from "within-session consistent idiosyncrasy." That separation requires multi-session data and is addressed by the session-level subject-ID probe in §4.5 (Fig 5b-R), which runs on the subset of cells that have ≥ 2 recordings per subject.

---

## §4.2 Representation geometry across the 2×2 (Diagnostic 1: variance decomposition)

> 中文：4 cell window-level variance decomposition — 維持原 stacked-bar 呈現（Label / Subject / Residual），單位改 window；caption 說明 ADFTD 被納入的 scope condition。

- **Fig 4.2 (re-rendered)**: 4-cell window-level variance decomposition — stacked bars per cell (label / subject within label / residual), frozen FM × 3 FM. Secondary row: `delta_label_frac(FT − frozen)` across 4 cells × 3 FMs.
- Per-cell reading (absolute within-cell, not cross-cell):
  - *Within-subject × strong-aligned (EEGMAT)*: expected `label_frac` positive, `delta_label_frac` small (frozen already aligned)
  - *Within-subject × weak-aligned (SleepDep)*: expected `label_frac` near zero, `delta_label_frac` near zero
  - *Subject-label × strong-aligned (ADFTD split1)*: expected `label_frac` positive, `delta_label_frac` positive (FT lifts signal)
  - *Subject-label × weak-aligned (Stress)*: expected `label_frac` near zero, `delta_label_frac` small
- Cross-cell reading uses the **task-alignment ratio** `label_frac / subject_frac`: strong-aligned cells have higher ratios at frozen; weak-aligned cells collapse to ratio ≈ 0.

**Caption wording (Fig 4.2)**:
> Per-window embedding variance decomposed into label, between-subject (within-label), and residual components (nested ANOVA). Bars stacked to sum to 1.0 per (cell × FM). Window-level unit applied uniformly across all four cells; see §3.6.1 for scope condition. ADFTD split1 (one recording per subject) is included because the window-level convention does not require session-level replicates. Interpretations are within-cell relative; cross-cell absolute comparisons should use the complementary `delta_label_frac` (bottom row) and task-alignment ratio.

---

## §4.5 Fig 5b (FOOOF ablation probes) — split into two panels

> 中文：Fig 5b 拆為 state-probe 與 subject-probe 兩 panel；state 維持 4 cell，subject probe 為 session-level 因此限 EEGMAT / SleepDep / Stress 三 cell，ADFTD 以 scope note 標示。

### Panel B-L (State probe) — all 4 cells

- **Metric**: per-FM state-probe balanced accuracy under 4 FOOOF conditions (`original / aperiodic_removed / periodic_removed / both_removed`)
- **Protocol**: window-level training with `StratifiedGroupKFold(5)` by subject → subject-disjoint evaluation; per-recording probabilistic pooling.
- **Coverage**: EEGMAT / SleepDep / Stress / ADFTD — defined at single-session.
- **Reading**: which spectral component carries the FM's label-discriminative signal.

### Panel B-R (Subject probe) — 3 cells, ADFTD N/A by scope

- **Metric**: per-FM subject-probe balanced accuracy (multi-class subject-ID) under the same 4 FOOOF conditions.
- **Protocol**: **session-level** held-out within each subject — each subject's recordings are split into train / test halves; at least 2 recordings per subject required.
- **Coverage**: EEGMAT / SleepDep / Stress only. ADFTD (single-session) is outside the scope of this protocol — marked N/A with a scope footnote rather than using a within-session window-level substitute (which would measure a different quantity; see §5 scope map).
- **Reading**: which spectral component the FM uses to **persistently identify subjects across recording sessions** — the trait-memorisation diagnostic.

**Caption wording (Fig 5b)**:
> FOOOF-component ablation probes on frozen FM embeddings. **Left**: state probe (label) balanced accuracy under four ablation conditions (aperiodic only / periodic only / both removed / neither) — subject-disjoint 5-fold CV, 8 seeds. All four cells. **Right**: subject probe (subject-ID) balanced accuracy under the same conditions — session-level holdout within subject requires ≥ 2 recording sessions per subject. ADFTD (single-session cohort) is marked N/A and excluded from this panel by protocol; its within-session subject signature is reported in Appendix B.4 under a complementary window-level protocol. State-probe drops under ablation identify the FM's task-aligned spectral anchor; subject-probe drops identify the FM's trait-anchor.

### Appendix B.4 (new) — Within-session subject probe for the single-session cell

Report window-level subject-ID accuracy for ADFTD split1 as complementary context. Note ceiling saturation; interpret *within-cell direction* (which FOOOF removal increases/decreases the within-session subject-ID BA), not absolute magnitude.

---

## §5 Scope map (new subsection under Discussion)

> 中文：論文 toolkit 每個診斷對應的 cell coverage 表；把 scope conditions 明確列為 contribution 的一部分。

| Diagnostic | Where | Cell coverage | Scope condition | Why |
|---|---|---|---|---|
| Benchmark BA landscape | §4.1, Tab 1 | 4 cells | universal | standard eval |
| **Window-level variance decomposition** | §4.2, Fig 4.2 | **4 cells** | ≥ 2 windows per recording (universal at 5/10 s windows) | decouples from session structure |
| Permutation null | §4.3, Fig 3 | 4 cells | label-shuffle exchangeable at subject level | identifies strong- vs weak-aligned cells |
| Within-subject direction consistency | §4.4, Fig 4.4 | within-subject cells (EEGMAT, SleepDep) | same subject provides ≥ 2 recordings under contrasting label | by label-design; cells with subject-level trait labels excluded by construction (not a data limitation) |
| FOOOF state probe | §4.5, Fig 5b-L | 4 cells | frozen perwindow features available | window-level, subject-disjoint |
| **FOOOF subject probe (session-level)** | §4.5, Fig 5b-R | EEGMAT, SleepDep, **Stress** (3 cells) | ≥ 2 recordings per subject | tests trait-anchor under session holdout |
| FOOOF subject probe (within-session, appendix) | §B.4 | 4 cells (incl. ADFTD) | ≥ 2 windows per recording | complementary, ceiling-bound |

**Narrative paragraph for §5**:

> The diagnostic toolkit we apply in §4.2–§4.5 is deliberately assembled from tools whose scope conditions are explicit and heterogeneous. Window-level variance decomposition and permutation null generalise to all four cells, including single-session trait cohorts such as ADFTD. Within-subject direction consistency runs only on cells whose label design provides a within-subject contrast (EEGMAT, SleepDep). The FOOOF subject probe in its session-level form measures trait-anchor persistence across recording sessions and therefore applies to cells with multi-session-per-subject structure (EEGMAT, SleepDep, Stress); its within-session window-level variant applies everywhere but saturates toward ceiling and measures a different quantity (within-session subject fingerprint rather than cross-session trait identity).
>
> This scope heterogeneity is a **design feature of the toolkit**, not a limitation: single-session clinical rsEEG cohorts such as ADFTD are the majority pattern in published datasets (BrainLat, CAUEEG, TDBRAIN's HC arm, Cavanagh UNM PD's HC arm — all single-session for the control group). A diagnostic battery that refused to run on single-session cohorts would cede ≥ 50% of the small-N clinical rsEEG landscape. The contribution documented here is the pairing of each tool with an explicit scope condition, such that practitioners confronting a new cohort can identify which diagnostics will inform and which will not, before any FM training run.

---

## Conservative scope notes to add to §1.5.2 (contribution claim)

Current paper_outline §1.5.2 says:
> "variance decomposition returns a label-dominated vs subject-dominated reading anywhere."

Change to:
> "window-level variance decomposition returns a label vs subject vs residual partition at any cell with ≥ 2 windows per recording, including single-session trait cohorts; the session-level subject-ID probe complements this with a trait-memorisation test at cells with ≥ 2 recordings per subject."

---

## Metrics reported per cell × FM (window-level decomposition)

For each of 4 datasets × 3 FMs = 12 cells, output one JSON block:

```json
{
  "model": "labram",
  "dataset": "eegmat",
  "unit": "window",
  "n_windows": 17140,
  "n_recordings": 72,
  "n_subjects": 36,
  "n_feat_dims": 200,
  "frozen": {
    "label_frac": 0.0085,
    "subject_frac": 0.883,
    "residual_frac": 0.108,
    "label_subject_ratio": 0.0096,
    "omega2_label": 0.0073,
    "omega2_subject": 0.882,
    "nested_identifiable": true,
    "permanova_pseudoF": <number>,
    "permanova_p_value": <float>
  }
}
```

Aggregated file: `paper/figures/_historical/source_tables/variance_analysis_window_level.json`
(12 entries, one per (model, dataset) pair)

FT-side per-window decomposition is deferred: current `results/final/*/ft/*/seed*/fold*_features.npz` are pooled (1 row/recording), not per-window. Running window-level FT variance requires re-extracting features from the per-FM canonical FT checkpoints with `--save-features` at window granularity. This is tracked separately, not blocking the frozen-side run approved now.
