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
    """Return the per-seed ``summary.json`` dicts for the LaBraM FT paired-null
    chain on ``dataset``, sorted by seed.

    Current home is ``results/studies/<exp>/<dataset>/perm_s*/summary.json``
    (scratchpad). When promoted to ``results/final/<dataset>/perm_null/``,
    redirect here and keep callers untouched.
    """
    root = STUDIES / exp / dataset.lower()
    paths = sorted(root.glob("perm_s*/summary.json"))
    if not paths:
        raise FileNotFoundError(f"No perm-null summaries under {root}")
    return [json.loads(p.read_text()) for p in paths]


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
