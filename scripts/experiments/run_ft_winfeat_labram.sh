#!/usr/bin/env bash
# Window-level FT feature extraction — LaBraM (single seed, 4 datasets)
# Canonical HP (G-F09) matching results/final/*/ft/labram/seed42/config.json.
# Writes to parallel path `results/final_winfeat/` so existing results/final/
# seed42 BA references remain untouched.
#
# Output: each fold N features.npz includes both pooled `features` (1/rec)
# AND `window_features` + `window_rec_idx` (per-window), enabling both the
# recording-level and window-level variance decomposition on the same run.
#
# Usage: GPU=3 bash scripts/experiments/run_ft_winfeat_labram.sh
set -e
PYTHON=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
cd /raid/jupyter-linjimmy1003.md10/UCSD_stress

FM=labram
GPU=${GPU:-3}
DEVICE=cuda:${GPU}
BASE="--mode ft --extractor $FM --lr 5e-4 --weight-decay 0.05 --llrd 0.65 \
  --label-smoothing 0.1 --head-hidden 0 --encoder-lr-scale 1.0 \
  --warmup-epochs 5 --batch-size 64 --norm zscore --loss ce \
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

echo "[$(date +%H:%M:%S)] LaBraM winfeat chain DONE"
