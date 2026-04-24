# Variance-decomposition methodology for the cross-dataset signal-strength analysis

**Scope.** This document specifies how we measure "does fine-tuning rewrite
the LaBraM representation?" in a way that survives reviewer scrutiny.
It is the reference for the variance-analysis numbers (canonical source:
`paper/figures/_historical/source_tables/variance_analysis_all.json` for the
legacy 4-dataset pooled run; current per-cell triangulation lives under
`results/studies/exp32_variance_triangulation/` and
`results/final/<dataset>/variance_triangulation/`).

**Naming.** The file is called `eta_squared_*` for historical reasons
(the first draft of the analysis used naive per-dim η²). We no longer use
η²; the reviewer-ready metric is the pooled label fraction (§2). All math
now lives in `src/variance_analysis.py`.

---

## 1. Problem setup

One vector per recording: LaBraM outputs a 200-d embedding.

| Regime | Model | Feature source |
|---|---|---|
| **Frozen** | Pretrained LaBraM, no fine-tuning | `results/features_cache/frozen_labram_{dataset}_19ch.npz` |
| **Fine-tuned (pooled)** | 5-fold subject-level CV; each recording's embedding comes from the fold-model that held it out | `results/features_cache/ft_labram/{dataset}_YYYY-MM-DD/fold{1..5}_features.npz` concatenated |

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

### 2.1 Why this metric, vs. the alternatives in EEG-FM-Bench

The closest methodological prior for tracking "does fine-tuning rewrite
an EEG FM representation" is EEG-FM-Bench (Xiong et al. 2025,
arXiv:2508.17742, §4.3). Their pipeline uses two diagnostics:

- **Gradient-mass per parameter group** during FT (Fig. 7 left). They
  observe that in pretrained-FT settings the gradient norm concentrates
  on the input Temporal Embedding while the Attention/MLP/Norm backbone
  receives small gradients. This is an *input-side*, *qualitative*,
  *multi-task averaged* observation.
- **CKA / RSA between Scratch-trained and Pretrained checkpoints** over
  training steps (Fig. 7 mid/right). This tracks whether multi-task FT
  pulls a scratch model toward the pretrained representation. It is
  *unsupervised* (label-blind) and *not* tracking how a pretrained
  representation evolves during FT on a single small clinical task.

Our pooled label fraction is the *output-side, label-aware,
dataset-stratified, per-fold-aware* counterpart to those measures. The
two are mechanistically consistent on Stress (small backbone gradients →
unchanged pooled label fraction 7.23%→7.24%) but our metric makes
behaviors visible that theirs cannot:

| Behavior | Visible to gradient-mass? | Visible to CKA/RSA? | Visible to pooled label fraction? |
|---|---|---|---|
| Backbone receives small gradients on small data | yes | no (compares scratch↔pretrained, not pre↔post FT) | indirectly — output is unchanged |
| Pretrained representation rewrites cleanly on medium data | no (gradient mass story is uniform) | partially (geometry shifts, but not toward what) | **yes**, with sign and magnitude (ADFTD 2.79%→7.70%) |
| Per-fold OOF models drift apart, **diluting** the global label signal on large subject-saturated data | no (per-fold gradients not analyzed) | no (multi-task averaging) | **yes** — TDBRAIN 2.97%→1.47% is the signature |
| Failure mode is dataset-size-dependent | no (multi-task averaged) | no | **yes** (the three-mode taxonomy) |

The TDBRAIN fold-drift dilution row is the diagnostic that justifies
the new metric: it is structurally invisible to any pipeline that
averages over folds or over tasks before computing the diagnostic, but
it is the dominant effect on subject-saturated datasets and inverts the
naive scaling-laws expectation.

---

## 3. Production results (2026-04-08)

Full analysis (Stress, ADFTD, TDBRAIN):
`/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python scripts/analysis/run_variance_analysis.py`
(n_boot=1000, n_perm=999, all 200 feature dims for mixed-effects).

EEGMAT analysis (within-subject design needs the crossed decomposition path):
legacy `scripts/analyze_eegmat.py` was removed — current crossed-path output is regenerated via `scripts/analysis/run_variance_analysis.py --dataset eegmat` or by the per-cell triangulation pipeline under `results/studies/exp32_variance_triangulation/`.

| Dataset | BA | n_rec / n_subj | Label type | Frozen label frac | FT pooled label frac | Change |
|---|---|---|---|---|---|---|
| ~~Stress~~ | ~~0.656~~ | 70 / 17  | between-subject | ~~**7.23%**~~ | ~~**7.24%**~~ | **STALE** — computed under subject-dass; see `docs/historical/progress.md` §4.6 |
| EEGMAT   | 0.736 | 72 / 36  | **within-subject** | **5.35%** | **5.82%** | → (×1.09) |
| ADFTD    | 0.752 | 195 / 65 | between-subject | 2.79% | 7.70% | ↑ ×2.76 |
| TDBRAIN  | 0.681 | 734 / 359| between-subject | 2.97% | 1.47% | ↓ ×0.49 |

> **Note (2026-04-10)**: The Stress row above is stale. The underlying FT
> feature run used `--label subject-dass` (trait-memorization artifact,
> see `docs/historical/progress.md` §4.6). Regenerating requires re-running Stress FT with
> `--label dass --save-features` and re-running
> `scripts/analysis/run_variance_analysis.py` after adding the new ft_dir to its
> DATASETS dict. The 2026-04-10 Stress result is currently only
> available at the **behavioral** level: Frozen LP 0.605 ± 0.030 vs
> Real FT 0.443 ± 0.068 (`results/studies/exp03_stress_erosion/`).

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
reshaping it. The same projection-only failure mode holds on **EEGMAT**
(5.35% → 5.82% at n=72, classifier BA=0.736), even though every EEGMAT
subject contributes both labels (paired baseline rest vs arithmetic
task) — the within-subject task contrast does *not* rescue
representation rewriting at small n. Subject identity dominates ~71%
of EEGMAT variance both before and after FT (mixed-effects ICC 0.56 →
0.63), confirming that the projection-only behavior is structural to
subject-dominated FM pretraining and depends on training-set size, not
on whether the label is biologically separable per subject. On TDBRAIN,
the pooled label fraction actually halves after fine-tuning (the five
fold-models drift in ways that dilute the global label signal), while
classifier BA remains at 0.68 because the large sample size still gives
enough signal for the head to read.

**Why EEGMAT needs a different code path.** `nested_ss` requires
subject-pure labels (every recording from a subject shares one label).
EEGMAT violates this by design — every subject contributes both rest
and task recordings. The pooled label fraction $SS_{\text{label}}/SS_{\text{total}}$
itself is still well-defined (it doesn't care about the subject design),
but the nested decomposition $SS_\text{label} + SS_{\text{subject}|\text{label}} + SS_\text{residual}$
needs to be replaced by the *crossed* decomposition $SS_\text{label} + SS_\text{subject} + SS_\text{interaction+resid}$.
The crossed path is exercised by `scripts/analysis/run_variance_analysis.py`
(EEGMAT branch) and reports the same pooled label fraction in a directly
comparable form.

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
  must run in `stress` (unified env, see §6).
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

- `scripts/analysis/run_variance_analysis.py` — produces the variance-analysis
  JSON. Default `--out paper/figures/variance_analysis.json` is legacy; point
  `--out` at `results/final/<dataset>/variance_triangulation/variance_analysis.json`
  (or the study subdir) to land under the current layout. Run under `stress`.
- Legacy figure builders (`build_cross_dataset_figure.py`,
  `build_label_subspace_figure.py`) were part of the pre-SDL layout and have
  been superseded. Current per-cell variance visuals are produced by
  `scripts/figures/build_fig2_2x2.py` (see `paper/figures/fig2/`) and by the
  appendix builders in `scripts/figures/` — consult
  `docs/paper_experiments_index.md` §4.2 for the current source-of-truth
  mapping.

Tests: `tests/test_variance_analysis.py` — 10 tests including synthetic
variance-component recovery, legacy reproduction, cluster bootstrap CI,
PERMANOVA null uniformity, mixed-effects ICC recovery, and label-subspace
concentration/spread detection.

---

## 6. How to reproduce

```bash
STRESS=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python

# 1. Full statistical analysis → variance_analysis.json
$STRESS scripts/analysis/run_variance_analysis.py

# 2. EEGMAT within-subject analysis (legacy: scripts/analyze_eegmat.py, removed;
#    current per-cell path goes through exp32_variance_triangulation/)

# 3. Cross-dataset summary + label-subspace diagnostic figures
#    Legacy builders (scripts/build_cross_dataset_figure.py,
#    scripts/build_label_subspace_figure.py) removed. Current figure builders:
#    scripts/figures/build_fig2_2x2.py and appendix builders in scripts/figures/.

# 5. Tests
conda run -n timm_eeg python tests/test_variance_analysis.py    # numpy-only
$STRESS tests/test_variance_analysis.py                          # full suite
```

All analysis (training, feature extraction, stats) now runs in the
unified `stress` conda env (scipy 1.17 + statsmodels 0.14 + sklearn 1.8
+ torch 2.5.1+cu124, built clean from conda-forge on 2026-04-08). The
legacy `stats_env` was removed after verifying bit-exact reproduction of
all pooled label fractions (Stress 7.23%→7.24%, ADFTD 2.79%→7.70%,
TDBRAIN 2.97%→1.47%) under `stress`. `timm_eeg` still has the broken
`scipy.interpolate._fitpack_impl` import, so numpy-only tests can run
there but anything touching `statsmodels`/`scipy.stats`/`sklearn.manifold.TSNE`
must use `stress`.
