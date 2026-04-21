#!/usr/bin/env bash
# exp27 paired null — ADFTD side, LaBraM, GPU 7, 30 perms.
# HP mirrors results/studies/exp07_adftd_multiseed/labram_s42/config.json
# ADFTD AD/HC is a subject-level trait (195 recs = 65 subjects × 3 splits),
# so --permute-level subject keeps each subject's splits label-consistent in
# the null; recording-level shuffle would create mixed-label subjects.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY="/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python"
GPU=${GPU:-7}
N_PERMS=${1:-30}

OUT_ROOT="results/studies/exp27_paired_null/adftd"
mkdir -p "${OUT_ROOT}/logs"

for seed in $(seq 0 $((N_PERMS-1))); do
  RUN_ID="studies/exp27_paired_null/adftd/perm_s${seed}"
  LOG="${OUT_ROOT}/logs/perm_s${seed}.log"
  echo "[$(date +%H:%M:%S)] START adftd perm_s${seed} on cuda:${GPU}" | tee -a "${OUT_ROOT}/logs/_chain.log"
  PYTHONUNBUFFERED=1 "$PY" train_ft.py \
    --mode ft --extractor labram --dataset adftd \
    --folds 5 --epochs 50 --patience 15 \
    --lr 1e-5 --encoder-lr-scale 0.1 --batch-size 32 --seed 42 \
    --loss focal --head-hidden 128 --norm zscore \
    --weight-decay 0.05 --grad-clip 2.0 --warmup-epochs 3 \
    --warmup-freeze-epochs 1 --aug-overlap 0.75 --llrd 1.0 \
    --device "cuda:${GPU}" \
    --run-id "${RUN_ID}" \
    --permute-labels "${seed}" --permute-level subject \
    > "${LOG}" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  adftd perm_s${seed}" | tee -a "${OUT_ROOT}/logs/_chain.log"
done
echo "[$(date +%H:%M:%S)] CHAIN COMPLETE: ${N_PERMS} adftd perms" | tee -a "${OUT_ROOT}/logs/_chain.log"
