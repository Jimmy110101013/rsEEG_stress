# Normalization Strategy Ablation

## Question
Is per-window z-score optimal, or does subject-level normalization preserve stress-discriminative amplitude information?

## Current Behavior
- Per-window, per-channel z-score: `(window - mean) / std` along time axis
- Each 5s window is independently normalized — removes absolute amplitude
- This erases sustained power changes (e.g., elevated theta in stressed subjects)

## Alternatives to Test

| Strategy | How | Preserves amplitude | Removes hardware bias |
|---|---|---|---|
| Per-window (current) | mean/std per (window, channel) | No | Yes |
| Per-recording | mean/std computed over all windows in one recording | Partially | Yes |
| Per-subject | mean/std computed over ALL recordings for that subject | Yes | Yes |
| None + /100 | raw uV / 100 (CBraMod default) | Yes | No |

## Implementation Notes
- Cache is keyed by norm type — new strategies need new cache entries
- Per-subject norm requires two passes: first compute stats, then normalize
- `StressEEGDataset._build_cache()` is where norm is applied (pipeline/dataset.py:99-103)
- Would need to add `norm="subject-zscore"` option

## Hypothesis
Subject-level norm preserves within-subject amplitude dynamics while removing cross-subject hardware variance. This could improve stress detection since chronic stress manifests as sustained power changes (theta/beta ratio).

## Priority
Low — test after CBraMod baseline and core ablation experiments are complete.
