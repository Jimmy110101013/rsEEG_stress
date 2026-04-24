#!/usr/bin/env bash
# Non-FM architecture ceiling on SleepDep (Fig 6 expansion)
# 2 models × 3 seeds = 6 runs, sequential on one GPU.
# Usage: bash scripts/experiments/_run_nonfm_sleepdep.sh cuda:6
set -o pipefail
export PYTHONUNBUFFERED=1

GPU="${1:-cuda:6}"
PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
DATASET=sleepdep
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
echo "NON-FM × SleepDep on ${GPU} — $(date)"
echo "=========================================="

for SEED in 42 123 2024; do
    run_one eegnet 1e-3 "$SEED"
    run_one shallowconvnet 5e-4 "$SEED"
done

echo "=========================================="
echo "DONE NON-FM × SleepDep — $(date)"
echo "=========================================="
