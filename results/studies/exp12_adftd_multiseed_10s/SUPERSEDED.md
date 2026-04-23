# SUPERSEDED

**Date**: 2026-04-23
**Superseded by**: `results/final/adftd/reve/ft/seed{42,123,2024}/`

## What this dir contains

ADFTD FT with **10 s window** (matching REVE canonical preprocessing) under the old `n_splits=3` cache + pre-G-F09 unified HP.

## Why superseded

- REVE now uses window=10 s as part of canonical per-FM preprocessing, extracted via `scripts/features/extract_frozen_all.py --extractor reve --window-sec 10 --adftd-n-splits 1`.
- `results/final/adftd/reve/ft/` FT runs consume the split1 / per-FM HP pipeline directly; the separate exp12 split (window-size-only variant for REVE) is no longer needed.

## Canonical location now

`results/final/adftd/reve/ft/seed{42,123,2024}/summary.json`

Retained for audit only.
