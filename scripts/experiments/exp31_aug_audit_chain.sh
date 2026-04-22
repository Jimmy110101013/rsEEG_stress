#!/usr/bin/env bash
# exp31 — aug_overlap audit. The canonical FT recipe bakes in
# --aug-overlap 0.75 on label==1. On ADFTD (label==1=AD=majority) LaBraM
# no-aug shows +3.5pp over aug-on, suggesting aug is a harmful artifact.
# This chain lets us sweep FM × dataset × aug to check generality.
#
# Usage:
#   GPU=3 FM=labram  DATASET=eegmat AUG=     SEEDS="123 2024" bash exp31_aug_audit_chain.sh
#   GPU=3 FM=cbramod DATASET=adftd  AUG=     SEEDS="42 123 2024" bash exp31_aug_audit_chain.sh
#   GPU=4 FM=reve    DATASET=adftd  AUG=     SEEDS="42 123 2024" bash exp31_aug_audit_chain.sh
#
# Per-FM recipe (norm / lr) is selected automatically. AUG unset => no --aug.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY="/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python"
GPU=${GPU:?set GPU=<n>}
FM=${FM:?set FM=<labram|cbramod|reve>}
DATASET=${DATASET:?set DATASET=<eegmat|adftd>}
AUG=${AUG:-}
SEEDS=${SEEDS:-"42 123 2024"}

case "${FM}" in
  labram)  NORM=zscore ; LR=1e-5 ;;
  cbramod) NORM=none   ; LR=1e-5 ;;
  reve)    NORM=none   ; LR=3e-5 ;;
  *) echo "Unknown FM ${FM}" >&2; exit 2 ;;
esac

if [[ -n "${AUG}" ]]; then
  AUG_FLAG=(--aug-overlap "${AUG}")
  TAG="aug$(printf '%.0f' "$(echo "${AUG} * 100" | bc -l)")"
else
  AUG_FLAG=()
  TAG="noaug"
fi

OUT_ROOT="results/studies/exp31_aug_audit"
mkdir -p "${OUT_ROOT}/logs"
CHAIN_LOG="${OUT_ROOT}/logs/_chain_${FM}_${DATASET}_${TAG}.log"

for SEED in ${SEEDS}; do
  RUN_ID="studies/exp31_aug_audit/${FM}_${DATASET}_${TAG}_s${SEED}"
  # Legacy naming preserved for LaBraM+no-aug runs already saved as <dataset>_noaug_s<seed>
  if [[ "${FM}" == "labram" ]]; then
    RUN_ID="studies/exp31_aug_audit/${DATASET}_${TAG}_s${SEED}"
  fi
  LOG="${OUT_ROOT}/logs/${FM}_${DATASET}_${TAG}_s${SEED}.log"
  echo "[$(date +%H:%M:%S)] START ${FM} ${DATASET} ${TAG} s${SEED} on cuda:${GPU}" | tee -a "${CHAIN_LOG}"
  PYTHONUNBUFFERED=1 "$PY" train_ft.py \
    --mode ft --extractor "${FM}" --dataset "${DATASET}" \
    --folds 5 --epochs 50 --patience 15 \
    --lr "${LR}" --encoder-lr-scale 0.1 --batch-size 32 --seed "${SEED}" \
    --loss focal --head-hidden 128 --norm "${NORM}" \
    --weight-decay 0.05 --grad-clip 2.0 --warmup-epochs 3 \
    --warmup-freeze-epochs 1 --llrd 1.0 \
    "${AUG_FLAG[@]}" \
    --device "cuda:${GPU}" \
    --run-id "${RUN_ID}" \
    > "${LOG}" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  ${FM} ${DATASET} ${TAG} s${SEED}" | tee -a "${CHAIN_LOG}"
done
echo "[$(date +%H:%M:%S)] CHAIN COMPLETE: ${FM} ${DATASET} ${TAG}" | tee -a "${CHAIN_LOG}"
