#!/usr/bin/env bash
# exp27 paired null — SleepDep side, LaBraM, GPU 6, 30 perms.
# HP mirrors results/studies/exp_newdata/sleepdep_ft_labram_s42/config.json
set -euo pipefail
cd "$(dirname "$0")/../.."

PY="/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python"
GPU=${GPU:-6}
N_PERMS=${1:-30}

OUT_ROOT="results/studies/exp27_paired_null/sleepdep"
mkdir -p "${OUT_ROOT}/logs"

for seed in $(seq 0 $((N_PERMS-1))); do
  RUN_ID="studies/exp27_paired_null/sleepdep/perm_s${seed}"
  LOG="${OUT_ROOT}/logs/perm_s${seed}.log"
  echo "[$(date +%H:%M:%S)] START sleepdep perm_s${seed} on cuda:${GPU}" | tee -a "${OUT_ROOT}/logs/_chain.log"
  PYTHONUNBUFFERED=1 "$PY" train_ft.py \
    --mode ft --extractor labram --dataset sleepdep \
    --folds 5 --epochs 50 --patience 15 \
    --lr 1e-5 --encoder-lr-scale 0.1 --batch-size 4 --seed 42 \
    --loss focal --head-hidden 128 --norm zscore \
    --weight-decay 0.01 --grad-clip 2.0 --warmup-epochs 3 \
    --warmup-freeze-epochs 1 --llrd 1.0 \
    --device "cuda:${GPU}" \
    --run-id "${RUN_ID}" \
    --permute-labels "${seed}" \
    > "${LOG}" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  sleepdep perm_s${seed}" | tee -a "${OUT_ROOT}/logs/_chain.log"
done
echo "[$(date +%H:%M:%S)] CHAIN COMPLETE: ${N_PERMS} sleepdep perms" | tee -a "${OUT_ROOT}/logs/_chain.log"
