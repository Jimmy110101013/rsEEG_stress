"""Smoke tests for src/results.py canonical accessors.

Catches two classes of silent regression:
1. An accessor's internal path drifts and it stops returning data.
2. An accessor's return value diverges from the raw json.load at the
   same path (e.g. someone adds a coercion that silently changes types).

Tests are plain asserts (no pytest dependency) — matches the style of
tests/test_variance_analysis.py. Run under any env with the `src/`
package importable:

    /raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python tests/test_results.py

They should complete in < 1 second and do NOT validate numeric
correctness of any experiment — that's the raw run's job.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import results  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

SOURCE_TABLE_NAMES = [
    "variance_analysis_all",
    "variance_analysis_window_level",
    "sleepdep_variance_rsa",
    "sleepdep_within_subject",
    "f14_within_subject",
    "master_frozen_ft_table_v2",
]
DATASETS = ["eegmat", "adftd", "stress", "sleepdep"]
FMS = ["labram", "cbramod", "reve"]


def test_source_table_matches_raw_json():
    for name in SOURCE_TABLE_NAMES:
        via_api = results.source_table(name)
        p = REPO / "results/final/source_tables" / f"{name}.json"
        raw = json.loads(p.read_text())
        assert via_api == raw, f"{name}: accessor diverged from raw json.load"


def test_source_table_missing_reports_available():
    try:
        results.source_table("does_not_exist")
    except FileNotFoundError as e:
        assert "Available:" in str(e), "error message should list alternatives"
        return
    raise AssertionError("expected FileNotFoundError for missing source table")


def test_perm_null_summaries_returns_list_of_dicts():
    for ds in DATASETS:
        ss = results.perm_null_summaries(ds)
        assert isinstance(ss, list), f"{ds}: expected list"
        assert len(ss) > 0, f"{ds}: empty perm-null result"
        assert all(isinstance(s, dict) for s in ss), f"{ds}: non-dict entry"
        assert "subject_bal_acc" in ss[0], f"{ds}: missing subject_bal_acc"


def test_labram_ft_ba_null_matched_returns_triple():
    for ds in DATASETS:
        mean, sd, n = results.labram_ft_ba_null_matched(ds)
        assert 0.0 < mean < 1.0, f"{ds}: mean BA outside [0,1]"
        assert sd >= 0.0, f"{ds}: negative std"
        assert n >= 1, f"{ds}: n_seeds < 1"


def test_lp_multiseed_has_expected_keys():
    for ds in DATASETS:
        for fm in FMS:
            d = results.lp_multiseed(ds, fm)
            for key in ["extractor", "dataset", "mean_8seed", "std_8seed_ddof1",
                        "mean_3seed_42_123_2024", "std_3seed_42_123_2024_ddof1",
                        "per_seed_ba"]:
                assert key in d, f"{ds}×{fm}: missing {key}"


def test_lp_stats_3seed_matches_multiseed_fields():
    """The 3-seed convenience accessor must agree with the underlying
    multiseed record exactly (bit-identical floats)."""
    for ds, fm in [("stress", "labram"), ("eegmat", "cbramod"),
                   ("adftd", "reve")]:
        full = results.lp_multiseed(ds, fm)
        stats = results.lp_stats_3seed(ds, fm)
        assert stats["mean"] == full["mean_3seed_42_123_2024"]
        assert stats["std"] == full["std_3seed_42_123_2024_ddof1"]
        assert stats["n_seeds"] == 3


def test_ft_stats_3seed_paths_return_dicts():
    for ds, fm in [("stress", "labram"), ("eegmat", "labram"),
                   ("adftd", "cbramod"), ("sleepdep", "reve")]:
        d = results.ft_stats(ds, fm)
        assert d is not None, f"{ds}×{fm}: expected dict, got None"
        for key in ["mean", "std", "n_seeds", "source"]:
            assert key in d, f"{ds}×{fm}: missing {key}"


def test_ft_stats_nonexistent_combination_is_safe():
    """A combination with neither 3-seed nor 1-seed data must return None,
    not raise."""
    out = results.ft_stats("meditation", "reve")
    assert out is None or out["n_seeds"] == 1


def test_fooof_ablation_probes_have_4_conditions():
    for ds in DATASETS:
        d = results.fooof_ablation_probes(ds)
        for fm in FMS:
            keys = set(d["results"][fm].keys())
            assert {"original", "aperiodic_removed", "periodic_removed",
                    "both_removed"}.issubset(keys), f"{ds}×{fm}: missing condition"


def test_subject_probe_temporal_block_has_4_conditions():
    for ds in DATASETS:
        d = results.subject_probe_temporal_block(ds)
        for fm in FMS:
            keys = set(d["results"][fm].keys())
            assert {"original", "aperiodic_removed", "periodic_removed",
                    "both_removed"}.issubset(keys), f"{ds}×{fm}: missing condition"


def test_band_stop_ablation_has_all_datasets():
    bs = results.band_stop_ablation()
    for ds in DATASETS:
        assert ds in bs, f"missing dataset in band-stop: {ds}"
        for fm in FMS:
            assert fm in bs[ds], f"missing fm {fm} for {ds}"


def test_classical_summary_returns_dict():
    for ds in ["eegmat", "stress", "sleepdep"]:
        d = results.classical_summary(ds)
        assert isinstance(d, dict)
        assert "dataset" in d


def test_master_performance_table_indexed_by_ds_fm():
    t = results.master_performance_table()
    for ds in DATASETS:
        for fm in FMS:
            key = f"{ds}_{fm}"
            assert key in t, f"missing key in master table: {key}"


def test_exp30_fm_performance_returns_list():
    rows = results.exp30_fm_performance()
    assert isinstance(rows, list)
    assert len(rows) > 0
    assert "fm" in rows[0] and "dataset" in rows[0]


def test_path_accessors_resolve():
    p = results.frozen_features_path("labram", "stress", 30)
    assert p.suffix == ".npz"
    assert p.exists()
    p2 = results.fooof_ablated_features_path("stress")
    assert p2.suffix == ".npz"
    assert p2.exists()


if __name__ == "__main__":
    tests = [fn for name, fn in list(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERR   {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
