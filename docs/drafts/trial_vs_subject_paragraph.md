# Draft paragraph: trial-level vs subject-level split framing

**Status:** Draft 2026-04-09 → **superseded 2026-04-10** by behavioral erosion
reframing. Keeping the split-framing points but replacing the "gap
explanation" paragraph with the erosion claim. Still needs:
- CBraMod / REVE per-rec dass runs to show erosion isn't LaBraM-specific
- Classical RF per-rec dass to sanity-check the method-independent erosion
- Final placement decision (end of Results vs start of Discussion)

**Stance:** Do NOT engage with the "calibration" framing of Lin et al. 2025.
Note the two layers of leakage in prior work (trial-level split + our own
subject-dass OR-aggregation), adopt the honest per-recording protocol, and
pivot to the erosion finding as the paper's positive contribution.

---

## Draft (2026-04-10 version)

> Prior work on this dataset reports trial-level cross-validation results
> (Lin et al., 2025, BA 0.81). We identify two layers of inflation in
> this regime. First, trial-level splits place recordings from the same
> subject in both training and test folds, conflating within-subject
> temporal structure with cross-subject generalization. Second, our own
> initial attempts used subject-level OR-aggregation of the DASS class
> (if any of a subject's recordings is labelled "increase", all of them
> are), which turns the task into subject-identity broadcast rather than
> stress-state classification. We therefore adopt the stricter protocol
> of Lin et al.'s per-recording labels combined with subject-level
> StratifiedGroupKFold splits.
>
> Under this honest protocol, fine-tuning LaBraM does not merely fail to
> generalize — it **actively degrades** the pretrained representation.
> An 8-seed frozen linear probe on cached 200-d LaBraM features achieves
> BA 0.605 ± 0.030, while 3-seed fine-tuning of the same encoder with
> the canonical recipe (lr=1e-5, encoder_lr_scale=0.1, layer decay=1.0,
> 50 epochs, focal loss, subject-level 5-fold CV) drops to 0.443 ± 0.068.
> A 10-run label-permutation null (shuffled recording-level labels, same
> recipe) gives 0.497 ± 0.081: fine-tuning on real labels is
> *statistically indistinguishable* from training on random labels
> (one-sided p(null ≥ real) = 0.70), and in several permutation seeds
> the shuffled-label model outperforms every real-label fine-tuned
> model. The 16 pp gap between frozen and fine-tuned BA, combined with
> frozen LP's tighter seed variance (±0.030 vs ±0.068 real / ±0.081
> null), indicates that fine-tuning under weak cross-subject DASS labels
> erases the linearly-separable stress signal that the pretrained
> encoder already carries. The remainder of this paper positions this
> erosion mode within a four-dataset taxonomy (ADFTD injection, EEGMAT
> mild injection, TDBRAIN silent erosion, Stress behavioral erosion)
> driven by label–biomarker correspondence strength rather than
> dataset size.

## Key requirements

1. One sentence to credit Lin et al. 2025 (number + regime), no criticism.
2. One sentence on *why* trial-level is not strict enough — technical, not
   moral.
3. One sentence to announce our stricter protocol (StratifiedGroupKFold) +
   the new BA + the gap.
4. One sentence to pivot the gap to the paper's positive contribution
   (pooled label fraction + cross-dataset refutation structure).

## Do NOT

- Do not use the word "calibration" or frame Lin 2025 as an implicit
  calibration regime (advisor's suggestion — user pushed back, wants clean
  methodological framing only).
- Do not add "sympathetic interpretation" language.
- Do not criticize Lin et al. explicitly; the stricter protocol speaks for
  itself.

## TODO before finalization

- [ ] Confirm Lin et al. 2025 citation key in `paper/references.bib`.
- [ ] Re-verify trial-level BA = 0.862 number against current
  `train_trial.py` run log (latest run in `results/`).
- [ ] Re-verify subject-level BA = 0.656 against canonical LaBraM FT run.
- [ ] Decide final placement: end of §Results (as a transition to
  §Discussion) vs start of §Discussion (as the framing sentence for
  the variance-analysis section).
