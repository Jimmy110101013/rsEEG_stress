#!/usr/bin/env bash
# Non-FM architecture ceiling on ADFTD (Fig 6 expansion)
# 2 models × 3 seeds = 6 runs, sequential on one GPU.
# Usage: bash scripts/experiments/_run_nonfm_adftd.sh cuda:5
set -o pipefail
export PYTHONUNBUFFERED=1

GPU="${1:-cuda:5}"
PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
DATASET=adftd
EXP_ROOT="studies/exp15_nonfm_baselines/${DATASET}"

run_one() {
    local MODEL=$1 LR=$2 SEED=$3
    local TAG="${MODEL}_lr${LR}_s${SEED}"
    local RUN_ID="${EXP_ROOT}/${TAG}"
    local SUMMARY="results/${RUN_ID}/summary.json"

    if [ -f "$SUMMARY" ] && [ -f "results/${RUN_ID}/fold5_features.npz" ]; then
        echo "[$(date +%H:%M)] SKIP ${TAG} (done)"
        return 0
    fi
    mkdir -p "results/${RUN_ID}"

    echo "[$(date +%H:%M)] START ${TAG} on ${GPU}"
    $PY train_ft.py \
        --run-id "$RUN_ID" \
        --device "$GPU" \
        --dataset "$DATASET" \
        --n-splits 1 \
        --extractor "$MODEL" \
        --mode ft \
        --norm zscore \
        --folds 5 --epochs 80 --patience 20 \
        --batch-size 32 \
        --loss focal --head-hidden 128 \
        --weight-decay 0.01 --grad-clip 2.0 \
        --warmup-epochs 5 --warmup-freeze-epochs 0 \
        --encoder-lr-scale 1.0 --llrd 1.0 \
        --aug-overlap 0.75 \
        --lr "$LR" \
        --seed "$SEED" \
        --save-features \
        2>&1

    if [ -f "$SUMMARY" ]; then
        local BA=$($PY -c "import json; print(json.load(open('${SUMMARY}')).get('subject_bal_acc'))")
        echo "[$(date +%H:%M)] OK ${TAG} bal_acc=${BA}"
    else
        echo "[$(date +%H:%M)] FAIL ${TAG} (no summary)"
    fi
    $PY -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
}

echo "=========================================="
echo "NON-FM × ADFTD on ${GPU} — $(date)"
echo "=========================================="

ARCH_FILTER="${ARCH:-both}"  # both | eegnet | shallowconvnet
for SEED in 42 123 2024; do
    if [ "$ARCH_FILTER" = "both" ] || [ "$ARCH_FILTER" = "eegnet" ]; then
        run_one eegnet 1e-3 "$SEED"
    fi
    if [ "$ARCH_FILTER" = "both" ] || [ "$ARCH_FILTER" = "shallowconvnet" ]; then
        run_one shallowconvnet 5e-4 "$SEED"
    fi
done

echo "=========================================="
echo "DONE NON-FM × ADFTD — $(date)"
echo "=========================================="
