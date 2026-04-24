#!/usr/bin/env bash
# Final FT — LaBraM (GPU 3)
# Per-FM official HP: lr=5e-4, wd=0.05, llrd=0.65, elrs=1.0, bs=64, warmup=5, ls=0.1, loss=ce
# 4 datasets × 3 seeds = 12 runs
set -e
PYTHON=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
cd /raid/jupyter-linjimmy1003.md10/UCSD_stress

FM=labram
DEVICE=cuda:3
BASE="--mode ft --extractor $FM --lr 5e-4 --weight-decay 0.05 --llrd 0.65 \
  --label-smoothing 0.1 --head-hidden 0 --encoder-lr-scale 1.0 \
  --warmup-epochs 5 --batch-size 64 --norm zscore --loss ce \
  --epochs 200 --save-features --device $DEVICE"

for SEED in 42 2024 123; do
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
