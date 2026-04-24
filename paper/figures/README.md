# `paper/figures/` — current figure layout

Per-figure subdirs for the current 2×2-factorial paper:

```
paper/figures/
├── main/                  # hand-composited master PNGs used in paper/main.tex
│   ├── fig2_representation_2x2.{pdf,png}   # Fig 2 — representation variance bars
│   ├── fig3_honest_evaluation_4panel.{pdf,png}  # Fig 3 — permutation null 2×2
│   ├── Fig4_dir_consist.png                # Fig 4 — within-subject trajectories
│   ├── Fig5_FOOOF_band_stop.png            # Fig 5 — causal anchor ablation
│   └── Fig6_ceiling.png                    # Fig 6 — architecture ceiling
├── fig2/                  # (room for per-panel fig2 outputs)
├── fig3/                  # ditto
├── fig4/{fig4a,fig4b,fig4c}_*.{pdf,png}    # per-panel outputs
├── fig5/{fig5a,fig5b,fig5c}_*.{pdf,png}    # per-panel outputs
├── fig6/                  # per-panel outputs
├── appendix/              # figA1_variance_atlas_disease, figB1_channel_ablation, figB2_band_stop_breakdown
└── _historical/           # pre-pivot SDL-critique + v1 figure archives (read-only)
```

## Current builders

| Figure / Table | Builder | Access |
|---|---|---|
| Fig 2 | `scripts/figures/build_fig2_2x2.py` | `src.results.source_table(...)` |
| Fig 3 | `scripts/figures/build_fig3_perm_null_4panel.py` | `src.results.perm_null_summaries(...)`, `labram_ft_ba_null_matched(...)` |
| Fig 4, 5, 6, figA1, figB1, figB2 | `notebooks/_build_figures_consolidated.py` → `figures_consolidated.ipynb` (execute cells) | (path-based for now; `src.results` migration pending) |
| Table 1 | `scripts/figures/build_master_performance_table.py` | `src.results.lp_stats_3seed(...)`, `ft_stats(...)` |
| Paper PDF | `scripts/figures/build_paper_pdf.py` | — (compiles `paper/main.tex`) |

**Rule**: builders MUST read paper-cited numbers through `src.results.*`, NOT by memorising paths. See `results/final/README.md` for the accessor contract.

## Navigation

- Which paper section each number belongs to: `docs/paper_experiments_index.md`
- Findings claims (`F-A`..`F-E`): `docs/findings.md`
- Methodology guardrails: `docs/methodology_notes.md`

## Historical ledger

The pre-2026-04-15 per-figure provenance ledger (F01–F20 numbering, SDL-critique era figure list with `fig1_pipeline` … `fig8c_non_fm_baselines`) is preserved at git tag / in `paper/figures/_historical/` alongside the frozen outputs.
