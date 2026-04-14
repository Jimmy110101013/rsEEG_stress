# TODO — Current Priorities

**Last updated**: 2026-04-14

Update this file as priorities shift. Delete completed items; don't accumulate history.

---

## High priority

1. **Align advisor on updated findings**
   - Cross-dataset taxonomy is model-specific, not universal (F17)
   - Erosion is LaBraM-specific on Stress (F05); CBraMod/REVE show injection
   - Classical features collapse under honest labels while FM frozen features survive (F03/F16)
   - Within-subject direction consistency explains EEGMAT vs Stress gap (exp11)
   - Neuroscience interpretability: channel ablation + band RSA + band-stop ablation (exp14)
   - Show `docs/findings.md` as the current state

2. **Decide paper narrative given F14 + F17**
   - F14: longitudinal pivot ruled out → paper stays diagnosis-only
   - F17: cross-dataset taxonomy is model-specific → incorporate model-architecture effects
   - Update `docs/paper_strategy.md`

3. **Paper drafting** (after figures + narrative locked)
   - `paper/main.tex` or `paper/draft.md`
   - All experiment figures generated (exp01–exp14)

## Reviewer-driven experiments (complete before drafting)

All three DONE (2026-04-14). See F18, F19, F20 in `docs/findings.md`.

- ~~Item 4 Stress 5s window FT~~ → F18: REVE injection window-dependent (5s +4.8pp, 10s +12.8pp)
- ~~Item 5 Non-FM baselines~~ → F20: ShallowConvNet 0.557 matches FM range
- ~~Item 6 CBraMod/REVE perm null~~ → F19: both borderline p=0.100

Optional follow-up:
- Extend exp 6 to 20 perms for tighter p-value bound (if reviewer pushes)

## Lower priority

7. **HNC Dementia + MDD private dataset**
   - Branch `feat/hnc-dementia-mmd`
   - Encrypted HDF5+pkl, 30ch mV-scale, 308+400 subjects

## Completed this session

- ~~exp12 ADFTD 10s FT~~ — 9/9 done, 10s degrades BA 3-8pp vs 5s
- ~~exp13 cross-dataset figure~~ — updated with 5s ADFTD, 10s in supplementary
- ~~exp09 matched N BA curves~~ — ADFTD BA at N=17 ≈ Stress; EEGMAT robust at N=10
- ~~exp14 channel importance topomap~~ — posterior bias across all FMs
- ~~exp14 band RSA~~ — Stress: no band selectivity; EEGMAT: alpha/theta preference
- ~~exp14 band-stop ablation~~ — running now
- ~~exp02/06/10 figures~~ — classical vs FM, fitness heatmap, stat hardening
- ~~Cleanup~~ — deleted 5 obsolete scripts, all .ipynb_checkpoints, updated TODO
- ~~NUMA discovery~~ — 2× parallel speedup with numactl binding, skill updated

## Ruled out — do not re-propose

- Subject-adversarial GRL/DANN (~0.12 BA drop) — F12
- LEAD-style subject CE loss (0.439 BA) — F12
- Canonical 0.656 as "LaBraM ceiling" — cuDNN noise (F08)
- `--label subject-dass` for any new experiment — deprecated (F07)
- Within-subject longitudinal DSS reframing — all FMs below chance (F14)
- Sparse-label-subspace hypothesis — refuted, script deleted

## Notes for paper discussion section

- GRL/DANN and LEAD negative results (F12) should be mentioned as mitigation attempts
- Stress ground truth (DSS/DASS) is self-report → direction consistency comparison with EEGMAT addresses this
- Three independent FM benchmarks (Brain4FMs, EEG-FM-Bench, AdaBrain-Bench) confirm our findings (F11)
- ADFTD 10s vs 5s: 10s degrades BA 3-8pp and increases seed variance 2-6× (exp12 vs exp07)
- Neuroscience triad: channel ablation (spatial) + band RSA (correlational spectral) + band-stop ablation (causal spectral) → FM captures broadband subject fingerprint on Stress, frequency-specific task signal on EEGMAT
