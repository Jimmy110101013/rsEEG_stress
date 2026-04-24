#!/bin/bash
# CBraMod × Stress permutation-null chain (GPU 3, 10 seeds sequential)
set -e
cd /raid/jupyter-linjimmy1003.md10/UCSD_stress
PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
LOGDIR=results/studies/exp03_stress_erosion/logs
mkdir -p $LOGDIR results/studies/exp03_stress_erosion/ft_null_cbramod

SHARED="--mode ft --folds 5 --epochs 50 --patience 15 --batch-size 32 --seed 42 \
  --label dass --loss focal --head-hidden 128 --csv data/comprehensive_labels.csv \
  --weight-decay 0.05 --grad-clip 2.0 --warmup-epochs 3 --warmup-freeze-epochs 1 \
  --aug-overlap 0.75 --llrd 1.0 --extractor cbramod --norm none --lr 1e-5 \
  --encoder-lr-scale 0.1 --device cuda:0"

export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

for s in 0 1 2 3 4 5 6 7 8 9; do
  echo "=== cbramod perm_s$s start $(date +%H:%M:%S) ==="
  $PY train_ft.py $SHARED \
    --run-id studies/exp03_stress_erosion/ft_null_cbramod/perm_s$s \
    --permute-labels $s \
    > $LOGDIR/cbramod_perm_s$s.log 2>&1
  echo "=== cbramod perm_s$s done $(date +%H:%M:%S) ==="
done
echo "CBraMod chain complete"
