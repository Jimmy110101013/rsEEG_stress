# Stress FT erosion study — 2026-04-10

Evidence package: LaBraM fine-tuning on UCSD Stress per-recording DASS labels
actively *erodes* the pretrained representation's linearly-separable stress
signal, rather than improving it.

## Headline

| Condition | Subject BA | std | n seeds/perms |
|---|---|---|---|
| **Frozen LP** (logistic regression on cached LaBraM features) | **0.6049** | 0.030 | 8 |
| **Real FT** (canonical recipe, real labels) | **0.4434** | 0.068 | 3 |
| **Null FT** (canonical recipe, shuffled labels) | **0.4973** | 0.081 | 10 |

- Frozen − Real FT: **+16.1 pp** (robust across seed count)
- Real FT − Null FT: **−5.4 pp** (FT under real labels is worse than FT under random labels)
- One-sided p(null ≥ real): **0.70** (real FT is indistinguishable from null)

Verdict: **erosion**. `analysis.json` holds the full numeric breakdown.

## Layout

```
2026-04-10_stress_erosion/
├── README.md            ← you are here
├── analysis.json        ← single source of truth for all numbers
├── ft_real/             ← real-label FT runs (canonical recipe)
│   ├── s42_llrd1.0/
│   ├── s42_llrd0.65/    ← Lin 2025 layer_decay variant
│   ├── s123_llrd1.0/
│   └── s2024_llrd1.0/
├── ft_null/             ← permutation null (10 shuffled-label FT runs)
│   └── perm_s{0..9}/
├── ft_drift_check/      ← HEAD-vs-canonical reproducibility check
│   └── subjectdass_canonical_s42/
├── frozen_lp/
│   └── multi_seed.json  ← 8-seed frozen LP results
└── logs/                ← all driver + launcher logs
```

## How each piece was produced

### Frozen LP (`frozen_lp/multi_seed.json`)

```bash
/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python \
    scripts/stress_frozen_lp_multiseed.py
```

Loads `results/cross_dataset/features_stress_19ch.npz` (cached mean-pooled
200-d LaBraM features, no fine-tuning) + `data/comprehensive_labels.csv`
Group column (per-recording DASS class), then runs subject-level
StratifiedGroupKFold(5) with class-weighted LogisticRegression across 8
seeds.

### Real FT runs (`ft_real/`)

Canonical recipe:
`--lr 1e-5 --encoder-lr-scale 0.1 --llrd 1.0 --epochs 50 --patience 15
--warmup-freeze-epochs 1 --loss focal --aug-overlap 0.75 --label dass`

Driver commands are preserved in `logs/driver_s{42,123,2024}_llrd1.0.log`
and `logs/driver_s42_llrd0.65.log`.

### Permutation null (`ft_null/`)

```bash
/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python \
    scripts/run_perm_null.py --n-perms 10 --gpus 6 7
```

Each run shuffles recording-level labels via a new `--permute-labels <seed>`
flag in `train_ft.py` before CV splitting, then trains under the identical
canonical recipe and training seed=42. Class balance is preserved.

### Drift check (`ft_drift_check/`)

Re-runs the archived canonical recipe at HEAD with `--label subject-dass`
and seed=42, which on 2026-04-05 produced BA 0.6559. At HEAD it produces
0.4505 — a 20.5 pp gap that diagnostics trace to cuDNN non-determinism
amplified on 70 recordings with 14 positives, not to functional code drift.
See `analysis.json` → `ft_drift_check.root_cause` for the full write-up.

## Reading guide

- If you want the numbers: `analysis.json`.
- If you want to re-run the frozen LP: `scripts/stress_frozen_lp_multiseed.py`.
- If you want to re-run the null: `scripts/run_perm_null.py`.
- If you want the training recipe used for every FT run: see
  `ft_real/*/config.json`.
- If something looks wrong in the numbers: compare `analysis.json` against
  the individual `summary.json` files under `ft_real/` and `ft_null/` —
  `analysis.json` is a derived aggregation, not authoritative.

## Supersedes

- `results/lin_compare/snapshot_2026-04-09_analysis.json` (partial matrix,
  uses 3-seed frozen LP only). Retained as a historical snapshot.
