"""Canonical accessors for paper-relevant results under ``results/final/``.

Contract: **paper figure/table builders read from this module, not from
`results/studies/` or `paper/figures/_historical/` directly**. When a
number the paper cites moves to a new file or a new experiment, patch
the accessor here and every caller inherits the fix.

The module grows incrementally — add a function when a builder needs
one, not before. Keep signatures shaped around what callers actually
ask ("give me LaBraM FT BA for dataset X") rather than around the
underlying file layout.

Scratchpad data under ``results/studies/`` is accessed here only when
no canonical snapshot exists yet; when the snapshot lands at
``results/final/``, redirect the accessor and delete the studies
branch — callers stay untouched.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FINAL = REPO / "results" / "final"
STUDIES = REPO / "results" / "studies"


# ---------------------------------------------------------------------------
# Source tables (aggregated JSONs at results/final/source_tables/)
# ---------------------------------------------------------------------------

def source_table(name: str) -> dict:
    """Load ``results/final/source_tables/<name>.json``.

    Raises FileNotFoundError with a discovery-friendly message if the table
    doesn't exist.
    """
    p = FINAL / "source_tables" / f"{name}.json"
    if not p.exists():
        available = sorted(p.name for p in (FINAL / "source_tables").glob("*.json"))
        raise FileNotFoundError(
            f"No source table at {p}. Available: {available}"
        )
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Permutation null (exp27_paired_null — 30-seed LaBraM FT null per dataset)
# ---------------------------------------------------------------------------

def perm_null_summaries(dataset: str, *, exp: str = "exp27_paired_null") -> list[dict]:
    """Return per-seed ``summary.json`` dicts for the LaBraM FT paired-null
    chain on ``dataset``, sorted by seed.

    Source: ``results/studies/<exp>/<dataset>/perm_s*/summary.json``
    (per-seed raw form). For the aggregated (mean/std) form, see
    :func:`perm_null_aggregate`.
    """
    root = STUDIES / exp / dataset.lower()
    paths = sorted(root.glob("perm_s*/summary.json"))
    if not paths:
        raise FileNotFoundError(f"No perm-null summaries under {root}")
    return [json.loads(p.read_text()) for p in paths]


def perm_null_aggregate(dataset: str, fm: str = "labram") -> dict:
    """Return the aggregated permutation-null snapshot for (dataset, fm).

    Path: ``results/final/<dataset>/perm_null/<fm>_null.json``. Schema:
    ``{provenance, model, cell, n_seeds, subject_bal_acc_per_seed,
    subject_bal_acc_mean, subject_bal_acc_std_ddof1, ...min, ...max}``.

    Cheaper than calling ``perm_null_summaries`` + reducing client-side.
    """
    p = FINAL / dataset.lower() / "perm_null" / f"{fm}_null.json"
    if not p.exists():
        raise FileNotFoundError(f"No aggregated perm-null at {p}")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# LaBraM FT balanced accuracy, matched to the perm-null training recipe
# ---------------------------------------------------------------------------

def labram_ft_ba_null_matched(dataset: str) -> tuple[float, float, int]:
    """Return (mean, std, n) of LaBraM FT BA under the recipe matching the
    exp27 perm-null training chain.

    Hides a historical bifurcation in the data layer:
    - EEGMAT / ADFTD → ``source_table('master_frozen_ft_table_v2')``
      (canonical, null-matched recipe at lr=1e-5).
    - Stress / SleepDep → ``results/studies/exp_30_sdl_vs_between/tables/
      fm_performance.json`` (per-dataset 3-seed FT under the recipe that
      matches the null chain's training config — Stress best-HP lr=1e-4;
      SleepDep canonical lr=1e-5 bs=4).

    TODO: promote Stress/SleepDep rows into ``master_frozen_ft_table_v2``
    once the data is re-homogenised, then drop the exp_30 branch here
    (caller API unchanged).
    """
    ds = dataset.lower()
    if ds in ("eegmat", "adftd"):
        tab = source_table("master_frozen_ft_table_v2")["table"]["labram"][ds]
        return float(tab["ft_mean"]), float(tab["ft_std"]), int(tab["ft_n"])

    perf_path = (STUDIES / "exp_30_sdl_vs_between" / "tables"
                 / "fm_performance.json")
    perf = json.loads(perf_path.read_text())
    bas = [r["bal_acc"] for r in perf
           if r["mode"] == "ft" and r["fm"] == "labram"
           and r["dataset"] == ds and r["bal_acc"] is not None]
    if not bas:
        raise RuntimeError(f"No real-FT seeds found for {ds} in {perf_path}")
    sd = statistics.stdev(bas) if len(bas) > 1 else 0.0
    return statistics.mean(bas), sd, len(bas)


def labram_ft_ba_null_matched_via_exp30(dataset: str) -> tuple[float, float, int]:
    """Public API for the exp_30 branch alone.

    Same return shape as :func:`labram_ft_ba_null_matched` but always reads
    from the exp_30 fm_performance table, no master-table fallback. Useful
    when a caller specifically wants the null-matched per-seed runs and not
    the master aggregate.
    """
    ds = dataset.lower()
    perf = exp30_fm_performance()
    bas = [r["bal_acc"] for r in perf
           if r["mode"] == "ft" and r["fm"] == "labram"
           and r["dataset"] == ds and r["bal_acc"] is not None]
    if not bas:
        raise RuntimeError(f"No real-FT seeds found for {ds} in exp_30 table")
    sd = statistics.stdev(bas) if len(bas) > 1 else 0.0
    return statistics.mean(bas), sd, len(bas)


# ---------------------------------------------------------------------------
# Linear probing (per-window LP, 8-seed output from train_lp.py)
# ---------------------------------------------------------------------------

def lp_multiseed(dataset: str, fm: str) -> dict:
    """Return the 8-seed per-window LP JSON for (dataset, fm).

    Current home: ``results/studies/perwindow_lp_all/<dataset>/<fm>_multi_seed.json``
    (scratchpad). When promoted to ``results/final/<dataset>/lp/<fm>.json``,
    flip this body; callers stay untouched.
    """
    p = STUDIES / "perwindow_lp_all" / dataset.lower() / f"{fm}_multi_seed.json"
    if not p.exists():
        raise FileNotFoundError(f"No LP multiseed at {p}")
    return json.loads(p.read_text())


def lp_stats_3seed(dataset: str, fm: str) -> dict:
    """3-seed (seeds 42/123/2024) LP statistics matching the paper's Table 1.

    Returns ``{mean, std, n_seeds, source}``. Std is sample std (ddof=1).
    """
    d = lp_multiseed(dataset, fm)
    return {
        "mean": float(d["mean_3seed_42_123_2024"]),
        "std":  float(d["std_3seed_42_123_2024_ddof1"]),
        "n_seeds": 3,
        "source": f"results/studies/perwindow_lp_all/{dataset}/{fm}_multi_seed.json",
    }


# ---------------------------------------------------------------------------
# Fine-tuning per-seed outputs (scattered across exp dirs — for now)
# ---------------------------------------------------------------------------

# Where each (dataset, fm) 3-seed FT run currently lives. Value is a callable
# taking a seed and returning the summary.json path. Absent keys → 1-seed
# fallback at results/features_cache/ft_<fm>_<ds>/summary.json.
#
# TODO: once all cells are snapshotted at results/final/<ds>/ft/<fm>/seed*/,
# replace this table with a single rule and delete per-cell branches.
_FT_3SEED = {
    ("sleepdep", "labram"):  lambda s: f"exp_newdata/sleepdep_ft_labram_s{s}/summary.json",
    ("sleepdep", "cbramod"): lambda s: f"exp_newdata/sleepdep_ft_cbramod_s{s}/summary.json",
    ("sleepdep", "reve"):    lambda s: f"exp_newdata/sleepdep_ft_reve_s{s}/summary.json",
    ("adftd",    "labram"):  lambda s: f"exp07_adftd_multiseed/labram_s{s}/summary.json",
    ("adftd",    "cbramod"): lambda s: f"exp07_adftd_multiseed/cbramod_s{s}/summary.json",
    ("adftd",    "reve"):    lambda s: f"exp07_adftd_multiseed/reve_s{s}/summary.json",
    ("stress",   "labram"):  lambda s: f"exp05_stress_feat_multiseed/s{s}_llrd1.0/summary.json",
    ("eegmat",   "labram"):  lambda s: f"exp04_eegmat_feat_multiseed/s{s}_llrd1.0/summary.json",
}


def ft_stats(dataset: str, fm: str, *, seeds=(42, 123, 2024)) -> dict | None:
    """Return ``{mean, std, n_seeds, source}`` for FT balanced accuracy.

    Prefers the 3-seed run at the known ``exp##`` location for (dataset, fm).
    Falls back to the 1-seed ``results/features_cache/ft_<fm>_<ds>/summary.json``
    if 3-seed isn't available. Returns ``None`` if no data exists.

    std is sample std (ddof=1); ``None`` for 1-seed fallbacks.
    """
    import numpy as np  # local to keep module import light

    ds = dataset.lower()
    key = (ds, fm)
    if key in _FT_3SEED:
        path_fn = _FT_3SEED[key]
        vals, sources = [], []
        for s in seeds:
            p = STUDIES / path_fn(s)
            if p.exists():
                vals.append(float(json.loads(p.read_text())["subject_bal_acc"]))
                sources.append(str(p.relative_to(REPO)))
        if len(vals) == len(seeds):
            return {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals, ddof=1)),
                "n_seeds": len(vals),
                "source": f"results/studies/{path_fn('{seed}').replace('{seed}', '{'+','.join(str(s) for s in seeds)+'}')}",
            }

    fallback = REPO / f"results/features_cache/ft_{fm}_{ds}/summary.json"
    if fallback.exists():
        d = json.loads(fallback.read_text())
        return {
            "mean": float(d["subject_bal_acc"]),
            "std":  None,
            "n_seeds": 1,
            "source": str(fallback.relative_to(REPO)),
        }
    return None


# ---------------------------------------------------------------------------
# FOOOF ablation probes (state + subject-ID), Fig 5
# ---------------------------------------------------------------------------

def fooof_ablation_probes(dataset: str) -> dict:
    """Return the FOOOF probes JSON (state-probe variant, Fig 5b).

    Reads from ``results/final/<dataset>/fooof_ablation/probes.json`` if
    present (the snapshotted form with ``provenance``); falls back to
    ``results/studies/fooof_ablation/<dataset>_probes.json`` otherwise.
    Both shapes have the same ``results: {<fm>: {<condition>: ...}}``
    payload at top level.
    """
    final = FINAL / dataset.lower() / "fooof_ablation" / "probes.json"
    if final.exists():
        return json.loads(final.read_text())
    studies = STUDIES / "fooof_ablation" / f"{dataset.lower()}_probes.json"
    if not studies.exists():
        raise FileNotFoundError(f"No FOOOF probes at {final} or {studies}")
    return json.loads(studies.read_text())


def subject_probe_temporal_block(dataset: str) -> dict:
    """Return the temporal-block subject-ID probe JSON (Fig 5b subject axis).

    Reads from ``results/final/<dataset>/subject_probe_temporal_block/
    probes.json`` if present (snapshotted form); falls back to
    ``results/studies/exp33_temporal_block_probe/<dataset>_probes.json``.
    """
    final = (FINAL / dataset.lower() / "subject_probe_temporal_block"
             / "probes.json")
    if final.exists():
        return json.loads(final.read_text())
    studies = STUDIES / "exp33_temporal_block_probe" / f"{dataset.lower()}_probes.json"
    if not studies.exists():
        raise FileNotFoundError(f"No temporal-block probe at {final} or {studies}")
    return json.loads(studies.read_text())


# ---------------------------------------------------------------------------
# exp14: band-stop + channel-importance (Fig 5c, Fig B.1, Fig B.2)
# ---------------------------------------------------------------------------

def band_stop_ablation() -> dict:
    """Return the cross-dataset band-stop ablation JSON.

    Schema: ``{<dataset>: {<fm>: {<band>: {"mean_distance": float, ...}}}}``.
    The cross-cell file at
    ``results/studies/exp14_channel_importance/band_stop_ablation.json``
    is the source of truth; per-cell snapshots at
    ``results/final/<dataset>/band_stop/probes.json`` are sliced views.
    """
    return json.loads((STUDIES / "exp14_channel_importance"
                       / "band_stop_ablation.json").read_text())


def band_stop_ablation_cell(dataset: str) -> dict:
    """Return the per-cell band-stop slice from results/final/.

    Schema: ``{"provenance": ..., "cell": ..., "bands": {<fm>: {<band>: ...}}}``.
    Convenience over :func:`band_stop_ablation`[dataset] when only one cell
    is needed.
    """
    p = FINAL / dataset.lower() / "band_stop" / "probes.json"
    if not p.exists():
        raise FileNotFoundError(f"No per-cell band-stop slice at {p}")
    return json.loads(p.read_text())


def channel_importance() -> dict:
    """Return the cross-dataset channel-importance JSON.

    Path: ``results/studies/exp14_channel_importance/channel_importance.json``.
    Used for Fig B.1 channel-ablation appendix figure.
    """
    return json.loads((STUDIES / "exp14_channel_importance"
                       / "channel_importance.json").read_text())


# ---------------------------------------------------------------------------
# Classical + non-FM deep baselines (Fig 6)
# ---------------------------------------------------------------------------

def classical_summary(dataset: str) -> dict:
    """Return the classical-baseline summary (Fig 6 squares).

    Reads from ``results/final/<dataset>/classical/summary.json`` if
    present; falls back to
    ``results/studies/exp02_classical_dass/<dataset>/summary.json``.
    Schema: per-method (logreg/svm/rf/xgb) per-seed BA, mean, std.
    """
    final = FINAL / dataset.lower() / "classical" / "summary.json"
    if final.exists():
        return json.loads(final.read_text())
    studies = STUDIES / "exp02_classical_dass" / dataset.lower() / "summary.json"
    if not studies.exists():
        raise FileNotFoundError(f"No classical summary at {final} or {studies}")
    return json.loads(studies.read_text())


# ---------------------------------------------------------------------------
# Feature cache paths (loader is caller's job — features are large)
# ---------------------------------------------------------------------------

def frozen_features_path(fm: str, dataset: str, channels: int) -> Path:
    """Path to ``results/features_cache/frozen_<fm>_<dataset>_<channels>ch.npz``.

    Returns the path; caller invokes ``np.load(...)``. Kept as a path-only
    accessor because feature arrays are large and loading is the caller's
    responsibility (and they often don't need every key).
    """
    return REPO / f"results/features_cache/frozen_{fm}_{dataset}_{channels}ch.npz"


def fooof_ablated_features_path(dataset: str, *, w5: bool = False) -> Path:
    """Path to FOOOF-ablated frozen features.

    ``w5=True`` returns the 5 s window variant
    (``results/features_cache/fooof_ablation/<dataset>_norm_none_w5.npz``);
    default returns the standard window length
    (``results/features_cache/fooof_ablation/<dataset>_norm_none.npz``).
    """
    suffix = "_w5" if w5 else ""
    return REPO / f"results/features_cache/fooof_ablation/{dataset}_norm_none{suffix}.npz"


# ---------------------------------------------------------------------------
# Pre-aggregated paper tables
# ---------------------------------------------------------------------------

def master_performance_table() -> dict:
    """Return ``paper/tables/_source/table1_master_performance.json``.

    Pre-aggregated by ``scripts/figures/build_master_performance_table.py``;
    schema is ``{<dataset>_<fm>: {"lp": {...}, "ft": {...}}}``.

    Used by the Fig 6 builder which reads pre-computed FM LP/FT numbers
    rather than re-aggregating per seed.
    """
    return json.loads((REPO / "paper/tables/_source"
                       / "table1_master_performance.json").read_text())


def exp30_fm_performance() -> list[dict]:
    """Return the raw per-seed FT BA list from the exp_30 sdl-vs-between
    aggregation (``results/studies/exp_30_sdl_vs_between/tables/
    fm_performance.json``).

    Used internally by ``labram_ft_ba_null_matched`` for the Stress/SleepDep
    bifurcation branch; exposed publicly so other consumers don't have to
    duplicate the path knowledge.
    """
    return json.loads((STUDIES / "exp_30_sdl_vs_between" / "tables"
                       / "fm_performance.json").read_text())
