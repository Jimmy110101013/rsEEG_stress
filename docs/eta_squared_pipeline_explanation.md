# Variance-decomposition methodology for the cross-dataset signal-strength analysis

**Scope.** This document specifies how we measure "does fine-tuning rewrite
the LaBraM representation?" in a way that survives reviewer scrutiny.
It is the reference for the numbers in `paper/figures/variance_analysis.json`
and the figures in `paper/figures/label_subspace.*` and
`paper/figures/cross_dataset_signal_strength.*`.

**Naming.** The file is called `eta_squared_*` for historical reasons
(the first draft of the analysis used naive per-dim η²). We no longer use
η²; the reviewer-ready metric is the pooled label fraction (§2). All math
now lives in `src/variance_analysis.py`.

---

## 1. Problem setup

One vector per recording: LaBraM outputs a 200-d embedding.

| Regime | Model | Feature source |
|---|---|---|
| **Frozen** | Pretrained LaBraM, no fine-tuning | `results/cross_dataset/features_{dataset}_19ch.npz` |
| **Fine-tuned (pooled)** | 5-fold subject-level CV; each recording's embedding comes from the fold-model that held it out | `results/..._feat/fold{1..5}_features.npz` concatenated |

Every fine-tuned embedding is out-of-fold: the model that produced it
never saw that recording during training. The pooled matrix has the same
sample size as the frozen matrix for each dataset.

Metadata per recording: `subject_id` (subject-level, pure within each
subject for all three datasets) and `label` (binary: normal/increase,
HC/AD, HC/MDD).

Dataset sizes:

| Dataset | n recordings | n subjects | Label |
|---|---|---|---|
| Stress (UCSD) | 70 | 17 | stress-increase vs normal |
| ADFTD | 195 | 65 | AD vs HC |
| TDBRAIN | 734 | 359 | MDD vs HC |

---

## 2. Primary metric: pooled label fraction

For feature matrix $F \in \mathbb{R}^{N \times D}$ and label vector $y$,
define per-dim sums of squares:

$$
SS_{\text{label}}^{(d)} = \sum_{\ell} n_\ell (\bar{x}^{(d)}_\ell - \bar{x}^{(d)})^2
\qquad
SS_{\text{total}}^{(d)} = \sum_{i} (x_i^{(d)} - \bar{x}^{(d)})^2
$$

Then the **pooled label fraction** is:

$$
\text{label\_frac} \;=\; \frac{\sum_d SS_{\text{label}}^{(d)}}{\sum_d SS_{\text{total}}^{(d)}}
$$

This is the fraction of the representation's total variance that the
label explains — in the whole 200-d cloud, not dim-by-dim averaged. It
is directly comparable to PERMANOVA's R² and to the multivariate ANOVA
interpretation reviewers expect.

**Why "pooled" not "per-dim averaged"?** The two differ when per-dim
variances are heterogeneous. A per-dim average overweights high-label
dims with small total variance (implicitly weighting by inverse
variance), which can show improvements that don't exist at the global
level. On Stress the per-dim average misleadingly shifts 5.9% → 6.9%
while the pooled stays 7.23% → 7.24%. The pooled version is the reviewer-
defensible statistic; the per-dim version is kept in the JSON for
completeness but not reported in the paper.

**Code**: `src/variance_analysis.py:analyze_regime` populates
`pooled_fractions.label` for every regime it analyzes.

---

## 3. Production results (2026-04-08)

Full analysis: `conda run -n stats_env python scripts/run_variance_analysis.py`
(n_boot=1000, n_perm=999, all 200 feature dims for mixed-effects).

| Dataset | BA | Frozen label frac | FT pooled label frac | Change |
|---|---|---|---|---|
| Stress  | 0.656 | **7.23%** | **7.24%** | **→ (unchanged, ×1.00)** |
| ADFTD   | 0.752 | 2.79%  | 7.70%  | **↑ ×2.76** |
| TDBRAIN | 0.681 | 2.97%  | 1.47%  | **↓ ×0.49** |

Supporting metrics, all consistent with the primary result:

| Dataset | Frozen PERMANOVA p | FT PERMANOVA p | Frozen mixed-effects ICC | FT mixed-effects ICC |
|---|---|---|---|---|
| Stress  | 0.247 | 0.363 | 0.44 | 0.45 |
| ADFTD   | 0.050 | **0.001** | 0.69 | 0.72 |
| TDBRAIN | 0.001 | 0.001 | 0.61 | 0.62 |

**Interpretation in one paragraph.**

Fine-tuning only measurably rewrites the LaBraM representation on ADFTD
(label fraction 2.8% → 7.7%, PERMANOVA p 0.050 → 0.001). On Stress, the
global label fraction is unchanged (7.23% → 7.24%) and PERMANOVA
correctly reports no change — the classifier achieves BA=0.66 by
learning a projection through an unchanged representation, not by
reshaping it. On TDBRAIN, the pooled label fraction actually halves
after fine-tuning (the five fold-models drift in ways that dilute the
global label signal), while classifier BA remains at 0.68 because the
large sample size still gives enough signal for the head to read.

---

## 4. What we tried that didn't work (brief)

Kept for reference so the same traps aren't rediscovered.

1. **Naive one-way η² subject/label ratio** (original draft). Structurally
   biased: because every subject is pure-label, $SS_{\text{subject}} \supseteq SS_{\text{label}}$
   by design, so the ratio is bounded below by 1 regardless of the
   representation. Replaced by nested decomposition + pooled label
   fraction.
2. **Per-dim averaged ω² ratio** (second attempt). Differs from the
   pooled version because it overweights low-variance dims; gave
   misleadingly rosy Stress numbers (see §2 above). Still computed and
   stored in the JSON as `nested_omega2` for cross-checking, but not
   the reported metric.
3. **Per-fold nested ω² on Stress.** Every Stress CV fold has exactly 1
   subject in the increase class — you cannot have a 5-way subject-
   disjoint split of 17 subjects without some fold having only one of
   the minority class. With 1 subject in the positive class,
   $SS_{\text{subject}|\text{label}=1}$ is mathematically forced to 0
   and the decomposition collapses into a "label captures everything
   the one subject varies on" degenerate form. The resulting per-fold
   "ratio dropped from 9× to 1.6×" number was an artifact, not signal.
   `src/variance_analysis.py:nested_decomposition_is_identifiable` now
   flags this condition and the JSON's `nested_identifiable` field is
   `false` for every Stress per-fold entry. **Downstream code and the
   paper must not cite per-fold Stress ω².**
4. **Sparse label subspace hypothesis** (an explanation I briefly floated
   for why per-dim ω² and PERMANOVA disagreed on Stress). Tested with
   `label_subspace_analysis`: dims_for_80% of label variance is 53 → 67
   (Stress), 58 → 65 (ADFTD), 63 → 63 (TDBRAIN) — essentially unchanged
   everywhere. Fine-tuning does not concentrate label signal into a
   sparse subspace on any dataset. The real answer is simpler: on
   Stress, the representation just didn't change.
5. **Bootstrap CIs on recordings.** Recordings within a subject are not
   independent; a recording-level bootstrap produces anti-conservative
   intervals. All bootstraps (`cluster_bootstrap`) resample *subjects*
   with replacement.
6. **PERMANOVA permuting recording labels.** Same pseudoreplication
   issue. `subject_level_permanova` permutes the subject→label mapping
   and propagates to all of a subject's recordings.

---

## 5. Code reference

All math lives in `src/variance_analysis.py`:

- `load_ft_features(run_dir)` / `load_ft_features_per_fold(run_dir)` /
  `load_frozen_features(npz_path, ft_run_dir)` — feature-matrix loaders.
- `nested_ss(features, subject, label)` — numpy implementation of
  $SS_{\text{label}} + SS_{\text{subject}|\text{label}} + SS_{\text{residual}} = SS_{\text{total}}$.
- `nested_decomposition_is_identifiable(subject, label)` — guard that
  flags per-fold regimes with <2 subjects per label class.
- `omega_squared_from_ss(ss)` — df-corrected ω² (kept for completeness;
  not the primary metric).
- `mixed_effects_variance(features, subject, label)` — REML
  `feat_d ~ label + (1|subject)` per dim. Lazy-imports statsmodels;
  must run in `stats_env`.
- `cluster_bootstrap(features, subject, label, stat_fn, n_boot)` —
  subject-level resampling.
- `subject_level_permanova(features, subject, label, n_perm)` —
  multivariate PERMANOVA, subject-level permutation.
- `label_subspace_analysis(features, label)` — concentration diagnostic
  in raw-dim and PCA bases.
- `analyze_regime(f, s, y)` — unified per-regime entry, produces the
  nested dict written to `variance_analysis.json`.
- `analyze_dataset(frozen, ft_per_fold)` — runs both frozen and
  fine-tuned paths, stores `frozen`, `ft_pooled`, `ft_per_fold`,
  `ft_aggregated`.

CLIs and figure producers:

- `scripts/run_variance_analysis.py` — produces `paper/figures/variance_analysis.json`.
  Run under `stats_env`.
- `scripts/build_cross_dataset_figure.py` — Panel A: BA bars; Panel B:
  pooled label fraction with fold-change annotations. Reads JSON,
  renders figure. Runs under `timm_eeg`.
- `scripts/build_label_subspace_figure.py` — 3×3 diagnostic figure
  (label fraction bar, cumulative label SS curve, top-2 PCA projection)
  plus t-SNE frozen-vs-fine-tuned. Runs under `stats_env` (needs
  scikit-learn).

Tests: `tests/test_variance_analysis.py` — 10 tests including synthetic
variance-component recovery, legacy reproduction, cluster bootstrap CI,
PERMANOVA null uniformity, mixed-effects ICC recovery, and label-subspace
concentration/spread detection.

---

## 6. How to reproduce

```bash
# 1. Full statistical analysis → variance_analysis.json
conda run -n stats_env python scripts/run_variance_analysis.py

# 2. Cross-dataset summary figure (pooled label fraction panel)
conda run -n timm_eeg python scripts/build_cross_dataset_figure.py

# 3. Diagnostic figure (label subspace + t-SNE)
conda run -n stats_env python scripts/build_label_subspace_figure.py

# 4. Tests
conda run -n timm_eeg python tests/test_variance_analysis.py    # numpy-only
conda run -n stats_env python tests/test_variance_analysis.py   # full suite
```

Two conda environments because `timm_eeg` has a broken
`scipy.interpolate._fitpack_impl` import that cascades to scipy.stats,
statsmodels, and sklearn.manifold.TSNE. `stats_env` has a clean
scipy/statsmodels/sklearn install and is used for any analysis that
touches those. The two envs communicate exclusively via on-disk
`.npz` / `.json` files.
