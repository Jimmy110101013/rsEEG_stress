#!/bin/bash
# ADFTD-only resume of run_variance_triangulation.sh
set -e
cd "$(dirname "$0")/../.."

PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
export PYTHONUNBUFFERED=1

LOGDIR=results/studies/exp32_variance_triangulation/logs
mkdir -p $LOGDIR
mkdir -p results/studies/exp33_temporal_block_probe/logs

DS=adftd
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
