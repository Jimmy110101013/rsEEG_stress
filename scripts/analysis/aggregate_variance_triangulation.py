"""Aggregate the 3-method variance triangulation into a single cross-cell table.

Reads:
  results/studies/exp33_temporal_block_probe/{cell}_probes.json
  results/studies/exp32_variance_triangulation/{cell}_permanova.json
  results/studies/exp32_variance_triangulation/{cell}_cka.json
  paper/figures/_historical/source_tables/variance_analysis_window_level.json
    (trace-ANOVA baseline, already built)

Emits a cross-cell comparison at
  paper/figures/_historical/source_tables/variance_triangulation.json
with the shape:

  {
    "cells": [{
      "cell": "eegmat",
      "fms": [{
        "fm": "labram",
        "trace_anova":   {"frozen_label": .., "ft_label": .., "delta_label_pp": ..,
                          "frozen_subject": .., "ft_subject": .., "delta_subject_pp": ..},
        "permanova":     {same schema, plus "p_label_frozen", "p_label_ft"},
        "cka":           {"frozen_label": .., "ft_label": .., "delta_cka_label": ..,
                          "frozen_subject": .., "ft_subject": .., "delta_cka_subject": ..},
        "subject_probe": {"original": .., "aperiodic_removed": .., "periodic_removed": ..,
                          "both_removed": .., "chance": ..}
      }]
    }]
  }
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

CELLS = ["eegmat", "sleepdep", "stress", "adftd"]
FMS = ["labram", "cbramod", "reve"]


def pct(x):
    return float(x) * 100.0


def load_trace_anova():
    p = REPO / "paper/figures/_historical/source_tables/variance_analysis_window_level.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def get_trace_entry(trace, cell, fm):
    """variance_analysis_window_level.json has format { "<cell>_<fm>": {frozen, ft, delta_label_frac, ...} }"""
    if not trace:
        return None
    for k, v in trace.items():
        if isinstance(v, dict) and v.get("cell") == cell and v.get("fm") == fm:
            return v
    key = f"{fm}_{cell}"
    return trace.get(key)


def main():
    trace = load_trace_anova()
    out = {"cells": []}

    for cell in CELLS:
        probe_p = REPO / f"results/studies/exp33_temporal_block_probe/{cell}_probes.json"
        perm_p = REPO / f"results/studies/exp32_variance_triangulation/{cell}_permanova.json"
        cka_p = REPO / f"results/studies/exp32_variance_triangulation/{cell}_cka.json"

        probe = json.loads(probe_p.read_text()) if probe_p.exists() else {"results": {}}
        perm = json.loads(perm_p.read_text()) if perm_p.exists() else {"results": {}}
        cka = json.loads(cka_p.read_text()) if cka_p.exists() else {"results": {}}

        cell_row = {"cell": cell, "fms": []}
        for fm in FMS:
            fm_row = {"fm": fm}

            # Trace-ANOVA from window-level json
            t = get_trace_entry(trace, cell, fm)
            if t and "frozen" in t and "ft" in t:
                fm_row["trace_anova"] = {
                    "frozen_label_pct": pct(t["frozen"].get("label_frac", 0)),
                    "ft_label_pct":     pct(t["ft"].get("label_frac", 0)),
                    "delta_label_pp":   pct(t.get("delta_label_frac", 0)),
                    "frozen_subject_pct": pct(t["frozen"].get("subject_frac", 0)),
                    "ft_subject_pct":   pct(t["ft"].get("subject_frac", 0)),
                    "delta_subject_pp": pct(t.get("delta_subject_frac", 0)),
                    "label_design": t["frozen"].get("label_design"),
                }

            # PERMANOVA
            pm = perm.get("results", {}).get(fm, {})
            if "frozen" in pm and "ft" in pm:
                fm_row["permanova"] = {
                    "frozen_label_pct":   pct(pm["frozen"]["label_frac"]),
                    "ft_label_pct":       pct(pm["ft"]["label_frac"]),
                    "delta_label_pp":     pct(pm["ft"]["label_frac"] - pm["frozen"]["label_frac"]),
                    "frozen_subject_pct": pct(pm["frozen"]["subject_frac"]),
                    "ft_subject_pct":     pct(pm["ft"]["subject_frac"]),
                    "delta_subject_pp":   pct(pm["ft"]["subject_frac"] - pm["frozen"]["subject_frac"]),
                    "p_label_frozen":     pm["frozen"]["p_label"],
                    "p_label_ft":         pm["ft"]["p_label"],
                    "design":             pm["frozen"]["design"],
                }

            # CKA
            ck = cka.get("results", {}).get(fm, {})
            if "frozen" in ck and "ft" in ck:
                fm_row["cka"] = {
                    "frozen_cka_label":   ck["frozen"]["cka_label"],
                    "ft_cka_label":       ck["ft"]["cka_label"],
                    "delta_cka_label":    ck["ft"]["cka_label"] - ck["frozen"]["cka_label"],
                    "frozen_cka_subject": ck["frozen"]["cka_subject"],
                    "ft_cka_subject":     ck["ft"]["cka_subject"],
                    "delta_cka_subject":  ck["ft"]["cka_subject"] - ck["frozen"]["cka_subject"],
                }

            # Subject probe (temporal-block)
            pr = probe.get("results", {}).get(fm, {})
            if pr:
                fm_row["subject_probe"] = {
                    cond: pr.get(cond, {}).get("subject_probe_mean")
                    for cond in ("original", "aperiodic_removed", "periodic_removed", "both_removed")
                }
                orig = pr.get("original", {})
                if orig:
                    fm_row["subject_probe"]["chance"] = orig.get("chance_rate")
                    fm_row["subject_probe"]["n_subjects"] = orig.get("n_subjects")

            cell_row["fms"].append(fm_row)

        out["cells"].append(cell_row)

    out_path = REPO / "paper/figures/_historical/source_tables/variance_triangulation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"-> {out_path}")

    # Print a human summary
    print("\n{:<10} {:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "cell", "fm", "ΔΑΝΟVΑ", "ΔPERM", "p(FT)", "ΔCKAlbl", "subj@orig", "chance"))
    for row in out["cells"]:
        cell = row["cell"]
        for fm_row in row["fms"]:
            fm = fm_row["fm"]
            ta = fm_row.get("trace_anova", {}).get("delta_label_pp")
            pm = fm_row.get("permanova", {}).get("delta_label_pp")
            pft = fm_row.get("permanova", {}).get("p_label_ft")
            ck = fm_row.get("cka", {}).get("delta_cka_label")
            sp = fm_row.get("subject_probe", {}).get("original")
            ch = fm_row.get("subject_probe", {}).get("chance")
            print(f"{cell:<10} {fm:<8} "
                  f"{('%.2f' % ta) if ta is not None else '-':>10} "
                  f"{('%.2f' % pm) if pm is not None else '-':>10} "
                  f"{('%.3f' % pft) if pft is not None else '-':>10} "
                  f"{('%.4f' % ck) if ck is not None else '-':>10} "
                  f"{('%.3f' % sp) if sp is not None else '-':>10} "
                  f"{('%.3f' % ch) if ch is not None else '-':>10}")


if __name__ == "__main__":
    main()
