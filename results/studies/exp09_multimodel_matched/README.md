# exp09: Multi-model Matched Subsample Analysis

Tests whether FM performance degrades at Stress-equivalent sample sizes (N=17 subjects).

## Contents

### Variance decomposition (label fraction vs N)
- `matched_subsample_multimodel.json` — Label fraction (%) at multiple N rungs, 100 draws, 3 FMs × 2 datasets (ADFTD, TDBRAIN). Frozen vs FT comparison with permutation null.
- `matched_subsample_curves.{pdf,png}` — 6-panel figure (3 FMs × 2 datasets)

### Classification BA vs N
- `matched_n_ba_curve.json` — Frozen LP BA at multiple N rungs (10–full), 100 draws, 3 FMs × 3 datasets (ADFTD, TDBRAIN, EEGMAT). Includes Stress reference BA.
- `matched_n_ba_curve.{pdf,png}` — 3-panel figure (1 per dataset, 3 FM curves)

## Key findings
- ADFTD LaBraM BA at N=17 (0.616) ≈ Stress LaBraM BA (0.605) — N contributes to low performance
- EEGMAT BA at N=10 already 0.645 — strong signal survives small N
- Conclusion: Stress failure is signal weakness × small N interaction, not sample size alone

## Scripts
- `scripts/matched_n_frozen_ba_curve.py` — BA curves
- `scripts/build_matched_subsample_figure.py` — Variance decomposition figure
