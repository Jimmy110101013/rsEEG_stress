# `results/final/` cleanup plan — snapshot layer for paper-final numbers

**Status**: draft 2026-04-23
**Related**: `docs/paper_experiments_index.md` (authoritative index), `docs/methodology_notes.md` (G-F*)

## Why this doc exists

Paper-relevant results currently span 15+ `results/studies/exp##_*/` directories + `results/features_cache/` + `paper/figures/_historical/source_tables/`. When a number appears in the paper, tracing it back to its source is 10+ min of searching. This is the "偷懶債" the user flagged 2026-04-23 after finding that `master_results_table.md` had stale pre-unified-HP FT values while `results/final/{cell}/{model}/ft/` already held the correct ones.

## Design principle (from advisor 2026-04-23)

**Don't move `results/studies/exp##_*/` trees.** Scripts hardcode those paths and each exp dir has per-fold curves / driver logs / fold features that are useful for reviewer questions beyond the paper-surface number.

Instead:

1. **`results/final/` is the snapshot layer.** Small JSONs + provenance pointers back to raw runs. Paper cites numbers from here.
2. **`results/studies/exp##_*/` stays the raw layer.** Untouched. Full per-fold data lives there.
3. **`scripts/analysis/rebuild_final_snapshots.py` is the bridge.** Re-runnable script that rebuilds every `results/final/*.json` from the raw runs. No silent drift: one command reproduces the snapshot layer.

Supersession policy: old raw dirs that are flat-out wrong (e.g. ADFTD split3 n_rec=195 runs) get a `SUPERSEDED.md` stub explaining what replaced them. They don't move to `results/archive/` unless truly dead.

## Target layout

```
results/final/
├── README.md                           # entry point; "this is the paper-surface layer"
├── tab1_benchmark.json                 # master table data (computed from below)
├── {cell}/                             # adftd, eegmat, sleepdep, stress (+ tdbrain for App A)
│   ├── ft/{model}/seed{42,123,2024}/
│   │   ├── summary.json                # already exists
│   │   └── provenance.json             # NEW: raw_dir, commit, date, HP
│   ├── lp/
│   │   └── {model}.json                # NEW: mirror of perwindow_lp_all/{cell}/{model}_multi_seed.json + provenance
│   ├── classical/
│   │   └── summary.json                # NEW: copy of exp02_classical_dass/{cell}/summary.json + provenance
│   ├── nonfm_deep/
│   │   ├── eegnet.json                 # NEW: aggregate 3 seeds from exp15_nonfm_baselines/{cell}/eegnet_lr*_s*/
│   │   └── shallowconvnet.json         # NEW: same pattern
│   ├── perm_null/
│   │   └── {model}_null.json           # NEW: 30-seed aggregate (mean/std/p-value) from exp27_paired_null/{cell}/ or exp03_stress_erosion/ft_null_*
│   ├── fooof_ablation/
│   │   └── probes.json                 # NEW: copy of fooof_ablation/{cell}_probes.json + provenance
│   └── band_stop/
│       └── probes.json                 # NEW: extract {cell} key from exp14_channel_importance/band_stop_ablation.json
└── cross_cell/
    ├── band_rsa.json                   # NEW: mirror of exp14_channel_importance/band_rsa.json + provenance
    └── variance_decomp.json            # NEW: mirror of variance_analysis_all.json + sleepdep_variance_rsa.json merged
```

Every JSON under `results/final/` has a top-level `provenance` field:

```json
{
  "provenance": {
    "raw_dir": "results/studies/exp07_adftd_multiseed",
    "snapshot_date": "2026-04-23",
    "commit": "d1517f4",
    "script": "scripts/analysis/rebuild_final_snapshots.py",
    "notes": "ADFTD split3 runs; superseded by results/final/adftd/ft/ under split1."
  },
  ...
}
```

## Rebuild script contract

`scripts/analysis/rebuild_final_snapshots.py` — one command rebuilds the snapshot layer:

```bash
python scripts/analysis/rebuild_final_snapshots.py --cell adftd        # one cell
python scripts/analysis/rebuild_final_snapshots.py --all               # all cells
python scripts/analysis/rebuild_final_snapshots.py --section lp        # one diagnostic across cells
python scripts/analysis/rebuild_final_snapshots.py --dry-run           # print what would change
```

Each section function:
- reads raw files
- computes aggregate (3-seed mean/std, or pass-through for single-seed)
- writes to `results/final/{cell}/{section}/*.json` with provenance

## Execution phases

### Phase 0 — directory skeleton (no data, just structure)
- [ ] Create `results/final/{adftd,eegmat,sleepdep,stress,tdbrain}/{ft,lp,classical,nonfm_deep,perm_null,fooof_ablation,band_stop}/` (FT already exists for 4 cells)
- [ ] Create `results/final/cross_cell/`
- [ ] Write `results/final/README.md` explaining the layer and rebuild command

### Phase 1 — LP snapshots (trivial, no agg needed)
- [ ] For each cell × FM: copy `perwindow_lp_all/{cell}/{model}_multi_seed.json` → `results/final/{cell}/lp/{model}.json` + add provenance
- [ ] Verify `docs/master_results_table.md` LP numbers match `results/final/{cell}/lp/` — should be no-op; if mismatch, investigate

### Phase 2 — Classical + non-FM deep snapshots
- [ ] Copy `exp02_classical_dass/{cell}/summary.json` → `results/final/{cell}/classical/summary.json` + provenance
- [ ] Aggregate 3 seeds from `exp15_nonfm_baselines/{cell}/eegnet_lr*_s*/`, write `results/final/{cell}/nonfm_deep/eegnet.json`; same for shallowconvnet
- [ ] Verify vs ceiling table.tex classical + nonfm rows

### Phase 3 — FT provenance stamps (no new numbers)
- [ ] For each existing `results/final/{cell}/{model}/ft/seed*/summary.json`: add sibling `provenance.json` with raw_dir, commit, HP recipe pulled from `config.json`

### Phase 4 — FOOOF + band-stop snapshots
- [ ] Copy `fooof_ablation/{cell}_probes.json` → `results/final/{cell}/fooof_ablation/probes.json` + provenance
- [ ] Extract each cell's rows from `exp14_channel_importance/band_stop_ablation.json` → `results/final/{cell}/band_stop/probes.json` + provenance
- [ ] Note: ADFTD FOOOF + band-stop currently pending per-FM window patch (separate task)

### Phase 5 — Perm null snapshots
- [ ] Aggregate 30 perm seeds from `exp27_paired_null/{cell}/perm_s*/summary.json` (ADFTD / SleepDep / EEGMAT) and `exp03_stress_erosion/ft_null_{model}/perm_s*/summary.json` (Stress); compute null mean, std, p-value of real FT BA → `results/final/{cell}/perm_null/{model}_null.json`
- [ ] Note: ADFTD null currently on split3 — pending re-run on split1 (separate task)

### Phase 6 — Cross-cell snapshots
- [ ] Copy `exp14_channel_importance/band_rsa.json` → `results/final/cross_cell/band_rsa.json` + provenance
- [ ] Merge variance JSONs → `results/final/cross_cell/variance_decomp.json` + provenance
- [ ] Build `tab1_benchmark.json` combining all 4 cells' summary numbers (used by ceiling table + master_results_table)

### Phase 7 — Docs integration
- [ ] Update `scripts/figures/build_frozen_vs_ft_table.py` to read from `results/final/{cell}/lp,ft/` instead of scattered paths
- [ ] Update `scripts/figures/build_fig2_2x2.py`, `build_fig3_perm_null_4panel.py`, etc., similarly — one script at a time
- [ ] Add a row to `docs/paper_experiments_index.md` for each `results/final/` path

### Phase 8 — Supersession stubs
- [ ] `results/studies/exp07_adftd_multiseed/SUPERSEDED.md`: ADFTD split3 n_rec=195 → replaced by `results/final/adftd/ft/` (split1). Keep dir for historical reference.
- [ ] Same for `exp12_adftd_multiseed_10s/`
- [ ] Grep-audit scripts that still read from superseded dirs

## Blast-radius checks before executing

Before each phase, run:

```bash
grep -rln "exp02_classical_dass\|exp07_adftd_multiseed\|exp15_nonfm_baselines\|exp17_eegmat\|exp27_paired_null\|fooof_ablation\|perwindow_lp_all\|exp14_channel_importance" scripts/ docs/ paper/
```

and confirm any script currently reading from paths we're snapshotting **does not break** when we add a `results/final/` mirror. Phase 7 then migrates consumers; until then, scripts keep reading the raw layer.

## Non-goals

- **Not** deleting or renaming existing `results/studies/exp##_*/` dirs (except truly-dead archives). Scripts keep working.
- **Not** touching `results/features_cache/` — feature caches are the extract artefact, they live in their canonical location.
- **Not** moving `paper/figures/_historical/source_tables/` content — those are figure-build intermediates; Phase 7 migrates consumers.

## Open decisions before starting

1. Phase 3 provenance stamps: include full HP dict or just `config.json` relative path? (Suggest: path only — dict duplicates config.json)
2. Phase 5 perm null aggregation: one-sided p-value, two-sided, or report both? (Suggest: report both; paper can pick)
3. TDBRAIN (App A): include in the final/ layout or leave in `results/studies/exp08_tdbrain_multiseed/`? (Suggest: include under `results/final/tdbrain/` for symmetry, even though App-only)
