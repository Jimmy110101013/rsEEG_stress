#!/bin/bash
# Variance triangulation + temporal-block subject probe across 4 cells x 3 FMs.
#
# Three analyses per dataset:
#   1. Temporal-block subject probe (subject probe BA for Fig 5b-R)
#   2. PERMANOVA cosine (variance partition robustness to Fig 2)
#   3. Linear CKA label/subject (alignment view)
#
# Outputs written to:
#   results/studies/exp33_temporal_block_probe/{cell}_probes.json
#   results/studies/exp32_variance_triangulation/{cell}_{permanova,cka}.json
#
# After all 4 cells complete, snapshot copies are made to
#   results/final/{cell}/subject_probe_temporal_block/
#   results/final/{cell}/variance_triangulation/

set -e
cd "$(dirname "$0")/../.."

PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
export PYTHONUNBUFFERED=1

LOGDIR=results/studies/exp32_variance_triangulation/logs
mkdir -p $LOGDIR
mkdir -p results/studies/exp33_temporal_block_probe/logs

CELLS=(eegmat sleepdep stress adftd)

for DS in "${CELLS[@]}"; do
  echo "=== ${DS} ==="

  echo "[1/3] temporal-block subject probe..."
  $PY scripts/analysis/run_temporal_block_subject_probe.py \
      --dataset $DS 2>&1 \
      | tee results/studies/exp33_temporal_block_probe/logs/${DS}.log \
      | grep -E "\[${DS}|mean=|seed=" || true

  echo "[2/3] PERMANOVA cosine..."
  $PY scripts/analysis/run_permanova_cosine.py \
      --dataset $DS --n-max 4000 --n-perm 999 2>&1 \
      | tee $LOGDIR/${DS}_permanova.log \
      | grep -E "\[${DS}|label_frac=|n_windows=" || true

  echo "[3/3] Linear CKA..."
  $PY scripts/analysis/run_cka_label_subject.py \
      --dataset $DS 2>&1 \
      | tee $LOGDIR/${DS}_cka.log \
      | grep -E "\[${DS}|CKA" || true

  echo "=== ${DS} done ==="
done

echo "ALL CELLS DONE"
