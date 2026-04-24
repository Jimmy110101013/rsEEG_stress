#!/usr/bin/env bash
# Final FT — REVE (GPU 7)
# Per-FM official HP: lr=2.4e-4, wd=0.01, llrd=1.0, elrs=0.1, bs=32, warmup=2, ls=0.0, loss=ce, beta2=0.95
# 4 datasets × 3 seeds = 12 runs
set -e
PYTHON=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
cd /raid/jupyter-linjimmy1003.md10/UCSD_stress

FM=reve
DEVICE=cuda:7
BASE="--mode ft --extractor $FM --lr 2.4e-4 --weight-decay 0.01 --llrd 1.0 \
  --label-smoothing 0.0 --head-hidden 0 --encoder-lr-scale 0.1 \
  --warmup-epochs 2 --batch-size 32 --norm none --loss ce \
  --adam-beta2 0.95 --epochs 200 --save-features --device $DEVICE"

for SEED in 2024 123; do
  PYTHONUNBUFFERED=1 $PYTHON train_ft.py $BASE \
    --dataset stress --label dass --aug-overlap 0.75 \
    --csv data/comprehensive_labels.csv \
    --seed $SEED --run-id final/stress/$FM/ft/seed${SEED}

  PYTHONUNBUFFERED=1 $PYTHON train_ft.py $BASE \
    --dataset eegmat --aug-overlap 0.75 \
    --seed $SEED --run-id final/eegmat/$FM/ft/seed${SEED}

  PYTHONUNBUFFERED=1 $PYTHON train_ft.py $BASE \
    --dataset adftd --n-splits 1 --aug-overlap 0.75 \
    --seed $SEED --run-id final/adftd/$FM/ft/seed${SEED}

  PYTHONUNBUFFERED=1 $PYTHON train_ft.py $BASE \
    --dataset sleepdep --aug-overlap 0.75 \
    --seed $SEED --run-id final/sleepdep/$FM/ft/seed${SEED}
done
