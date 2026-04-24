# SUPERSEDED

**Date**: 2026-04-23
**Superseded by**: `results/final/adftd/ft/{labram,cbramod,reve}/seed{42,123,2024}/`

## What this dir contains

Fine-tuning runs of 3 FMs on ADFTD under the **old n_splits=3** cache (each subject cut into 3 pseudo-recordings, n_rec=195). Also used `lr=1e-5` + unified HP — the pre-G-F09 recipe that killed CBraMod / REVE performance.

## Why superseded

Two upstream decisions:
1. **2026-04-23** — n_splits dropped 3 → 1 (canonical binary ADFTD has one recording per subject; split3 inflated the LP/FT denominator). See `docs/adftd_refresh_plan.md`.
2. **2026-04-19** — per-FM canonical HP recipe (G-F09 in `docs/methodology_notes.md`) replaced the unified `lr=1e-5` recipe. CBraMod FT BA went 0.537 → 0.698 (+16 pp) under the change.

## Canonical location now

Paper cites ADFTD FT from:
- `results/final/adftd/ft/{labram,cbramod,reve}/seed{42,123,2024}/summary.json`
- `results/final/adftd/lp/{labram,cbramod,reve}.json`

Do **not** consume this directory in new analyses. Retained for audit / historical reference only.
