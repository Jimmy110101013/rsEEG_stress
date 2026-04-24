#!/usr/bin/env bash
# Window-level FT feature extraction — CBraMod (single seed, 4 datasets)
# Canonical HP (G-F09) matching results/final/*/ft/cbramod/seed42/config.json.
# Output to results/final_winfeat/
set -e
PYTHON=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
cd /raid/jupyter-linjimmy1003.md10/UCSD_stress

FM=cbramod
GPU=${GPU:-4}
DEVICE=cuda:${GPU}
BASE="--mode ft --extractor $FM --lr 5e-4 --weight-decay 0.05 --llrd 1.0 \
  --label-smoothing 0.1 --head-hidden 0 --encoder-lr-scale 0.2 \
  --warmup-epochs 0 --batch-size 64 --norm none --loss ce \
  --epochs 200 --save-features --device $DEVICE"

SEED=42

PYTHONUNBUFFERED=1 $PYTHON train_ft.py $BASE \
  --dataset stress --label dass --aug-overlap 0.75 \
  --csv data/comprehensive_labels.csv \
  --seed $SEED --run-id final_winfeat/stress/$FM/seed${SEED}

PYTHONUNBUFFERED=1 $PYTHON train_ft.py $BASE \
  --dataset eegmat --aug-overlap 0.75 \
  --seed $SEED --run-id final_winfeat/eegmat/$FM/seed${SEED}

PYTHONUNBUFFERED=1 $PYTHON train_ft.py $BASE \
  --dataset adftd --n-splits 1 --aug-overlap 0.75 \
  --seed $SEED --run-id final_winfeat/adftd/$FM/seed${SEED}

PYTHONUNBUFFERED=1 $PYTHON train_ft.py $BASE \
  --dataset sleepdep --aug-overlap 0.75 \
  --seed $SEED --run-id final_winfeat/sleepdep/$FM/seed${SEED}

echo "[$(date +%H:%M:%S)] CBraMod winfeat chain DONE"
