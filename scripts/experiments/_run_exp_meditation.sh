#!/usr/bin/env bash
# Meditation FM experiments: 3 models × 3 seeds × 2 modes (LP, FT)
# Usage: bash scripts/_run_exp_meditation.sh cuda:5
set -o pipefail
export PYTHONUNBUFFERED=1

GPU="${1:-cuda:5}"
PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
DATASET=meditation
EXP_ROOT=results/studies/exp_newdata

run_one() {
    local MODEL=$1 NORM=$2 MODE=$3 SEED=$4
    local RUN_ID="${EXP_ROOT}/${DATASET}_${MODE}_${MODEL}_s${SEED}"
    local SUMMARY="${RUN_ID}/summary.json"

    if [ -f "$SUMMARY" ]; then
        echo "[SKIP] ${DATASET}_${MODE}_${MODEL}_s${SEED} (done)"
        return 0
    fi

    echo "=========================================="
    echo "[START] ${DATASET}_${MODE}_${MODEL}_s${SEED} on ${GPU}"
    echo "=========================================="
    $PY train_ft.py \
        --run-id "$RUN_ID" \
        --device "$GPU" \
        --dataset "$DATASET" \
        --extractor "$MODEL" \
        --mode "$MODE" \
        --norm "$NORM" \
        --seed "$SEED" \
        --folds 5 \
        --window-sec 5.0 \
        --loss focal \
        --save-features \
        2>&1
    echo "[DONE] ${DATASET}_${MODE}_${MODEL}_s${SEED}"
}

echo "=== Meditation experiments on ${GPU} ==="

# LP runs first (faster)
for SEED in 42 123 2024; do
    run_one labram zscore lp "$SEED"
    run_one cbramod none lp "$SEED"
    run_one reve none lp "$SEED"
done

# FT runs
for SEED in 42 123 2024; do
    run_one labram zscore ft "$SEED"
    run_one cbramod none ft "$SEED"
    run_one reve none ft "$SEED"
done

echo "=== Meditation ALL DONE ==="
