#!/bin/bash
# TODO item 3: CBraMod + REVE EEGMAT FT × 3 seeds for Master Table cells.
# Layout mirrors exp04_eegmat_feat_multiseed (LaBraM 3-seed).
# Usage: bash scripts/_run_exp17_eegmat_cbramod_reve.sh <model> <device>
#   model: cbramod | reve
#   device: cuda:3 | cuda:4 | cuda:7

set -o pipefail
export PYTHONUNBUFFERED=1

MODEL="$1"
GPU="$2"

if [ -z "$MODEL" ] || [ -z "$GPU" ]; then
    echo "Usage: $0 <model:cbramod|reve> <device:cuda:N>"
    exit 2
fi

PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
cd /raid/jupyter-linjimmy1003.md10/UCSD_stress

EXP_ROOT="results/studies/exp17_eegmat_cbramod_reve_ft"
mkdir -p "$EXP_ROOT/logs"

case "$MODEL" in
    cbramod) NORM=none;  LR=1e-5; EXTRA="" ;;
    # REVE default is 10s windows, but the Master Table uses 5s frozen features
    # (frozen_reve_eegmat_19ch.npz) — force 5s FT to keep the row internally consistent.
    reve)    NORM=none;  LR=3e-5; EXTRA="--window-sec 5.0" ;;
    *) echo "Unknown model $MODEL"; exit 2 ;;
esac

COMMON="--mode ft --dataset eegmat --folds 5 --epochs 50 --patience 15 \
  --batch-size 32 --loss focal --head-hidden 128 --weight-decay 0.05 \
  --grad-clip 2.0 --warmup-epochs 3 --aug-overlap 0.75 --save-features \
  --encoder-lr-scale 0.1 --llrd 1.0 --label dass --threshold 50"

run_one() {
    local seed="$1"
    local run_sub="${MODEL}_s${seed}"
    local run_dir="${EXP_ROOT}/${run_sub}"

    if [ -f "${run_dir}/summary.json" ] && [ -f "${run_dir}/fold5_features.npz" ]; then
        echo "[$(date +%H:%M)] SKIP ${run_sub} (already complete)"
        return 0
    fi

    mkdir -p "$run_dir"
    echo "[$(date +%H:%M)] START ${run_sub} on ${GPU}"

    timeout 1800 $PY train_ft.py \
        --run-id "studies/exp17_eegmat_cbramod_reve_ft/${run_sub}" \
        --device "$GPU" \
        --extractor "$MODEL" \
        --norm "$NORM" \
        --lr "$LR" \
        --seed "$seed" \
        $EXTRA \
        $COMMON \
        > "${EXP_ROOT}/logs/driver_${run_sub}.log" 2>&1
    local rc=$?

    if [ $rc -eq 124 ]; then
        echo "[$(date +%H:%M)] TIMEOUT ${run_sub}"
    elif [ $rc -ne 0 ]; then
        echo "[$(date +%H:%M)] FAIL ${run_sub} (exit=$rc)"
    elif [ -f "${run_dir}/summary.json" ]; then
        local ba=$($PY -c "import json; print(json.load(open('${run_dir}/summary.json')).get('subject_bal_acc'))")
        echo "[$(date +%H:%M)] OK ${run_sub} (subject_bal_acc=${ba})"
    else
        echo "[$(date +%H:%M)] FAIL ${run_sub} (no summary.json)"
    fi
    $PY -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
}

echo "=========================================="
echo "exp17 EEGMAT ${MODEL} 3-seed — $(date)"
echo "GPU: ${GPU}"
echo "=========================================="

run_one 42
run_one 123
run_one 2024

echo "=========================================="
echo "DONE exp17 ${MODEL} — $(date)"
echo "=========================================="
