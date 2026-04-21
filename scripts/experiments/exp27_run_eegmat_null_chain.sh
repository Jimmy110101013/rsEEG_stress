#!/usr/bin/env bash
# exp27 paired null — EEGMAT side, LaBraM canonical, GPU 4, 30 perms.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY="/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python"
GPU=4
N_PERMS=${1:-30}

for seed in $(seq 0 $((N_PERMS-1))); do
  RUN_ID="studies/exp27_paired_null/eegmat/perm_s${seed}"
  LOG="results/studies/exp27_paired_null/eegmat/logs/perm_s${seed}.log"
  echo "[$(date +%H:%M:%S)] START eegmat perm_s${seed} on cuda:${GPU}" | tee -a "results/studies/exp27_paired_null/eegmat/logs/_chain.log"
  PYTHONUNBUFFERED=1 "$PY" train_ft.py \
    --mode ft --extractor labram --dataset eegmat \
    --folds 5 --epochs 50 --patience 15 \
    --lr 1e-5 --encoder-lr-scale 0.1 --batch-size 32 --seed 42 \
    --loss focal --head-hidden 128 --norm zscore \
    --weight-decay 0.05 --grad-clip 2.0 --warmup-epochs 3 \
    --warmup-freeze-epochs 1 --aug-overlap 0.75 --llrd 1.0 \
    --device "cuda:${GPU}" \
    --run-id "${RUN_ID}" \
    --permute-labels "${seed}" \
    > "${LOG}" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  eegmat perm_s${seed}" | tee -a "results/studies/exp27_paired_null/eegmat/logs/_chain.log"
done
echo "[$(date +%H:%M:%S)] CHAIN COMPLETE: ${N_PERMS} eegmat perms" | tee -a "results/studies/exp27_paired_null/eegmat/logs/_chain.log"
