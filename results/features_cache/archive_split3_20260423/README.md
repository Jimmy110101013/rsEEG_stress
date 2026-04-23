# ADFTD frozen features — archived split3 backup

**Date archived**: 2026-04-23
**Reason**: ADFTD `n_splits=3` → `n_splits=1` policy change. Under `n_splits=3`, each ADFTD recording was cut into 3 pseudo-recordings (195 records instead of 65–82 subjects), inflating LP/variance denominators. The canonical binary protocol for the 4-dataset 2×2 factorial uses `n_splits=1` (one record per subject).
**Superseded by**: fresh `frozen_{model}_adftd_{19ch,perwindow}.npz` generated 2026-04-23 under `--adftd-n-splits 1` + per-FM window (labram/cbramod=5s, reve=10s).

See `docs/adftd_refresh_plan.md` for the full refresh log.

## Contents

Pre-2026-04-23 ADFTD frozen feature caches from `--adftd-n-splits 3` runs.
Do **not** consume these in new analyses. Retained for audit/diff only.
