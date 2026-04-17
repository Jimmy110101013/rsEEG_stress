#!/bin/bash
# exp20 — Wang 2025 strict-protocol reproduction.
#
# Differences from exp19 (closes 2 of the 3 identified gaps):
#   1. Single 80/10/10 split with Wang's exact counts (15/2/2 elev, 50/6/7 normal)
#      instead of 5-fold StratifiedKFold. Seeds {0,1,2,42} change the split, as in
#      the paper's Fig. 2.
#   2. Layer-wise LR decay = 0.65 (Wang Table II) for encoder param groups.
#
# Still divergent (need EEGLAB re-preprocess to close):
#   - Bandpass 1-50 Hz + ASR + ICLabel 80%. User reports EEGLAB preprocessing
#     has already been applied upstream, so we treat the cached .set files as
#     Wang-equivalent input.
#
# Usage: bash scripts/_run_exp20_wang_split.sh <model> <device>

set -o pipefail
export PYTHONUNBUFFERED=1

MODEL="$1"
GPU="$2"

if [ -z "$MODEL" ] || [ -z "$GPU" ]; then
    echo "Usage: $0 <model:labram|cbramod|reve> <device:cuda:N>"
    exit 2
fi

PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
cd /raid/jupyter-linjimmy1003.md10/UCSD_stress

EXP_ROOT="results/studies/exp20_wang_split_llrd"
mkdir -p "$EXP_ROOT/logs"

case "$MODEL" in
    labram)  NORM=zscore; LR=1e-5 ;;
    cbramod) NORM=none;   LR=1e-5 ;;
    reve)    NORM=none;   LR=3e-5 ;;
    *) echo "Unknown model $MODEL"; exit 2 ;;
esac

# patience=50 effectively disables SMA-3 early-stop (never triggers in 50 epochs),
# so we train the full 50 epochs and pick the best-val checkpoint (Wang's protocol).
COMMON="--mode ft --split-mode wang --folds 5 --epochs 50 --patience 50 \
  --batch-size 32 --loss focal --weight-decay 0.05 --grad-clip 1.0 \
  --warmup-epochs 3 --aug-overlap 0.75 --layer-decay 0.65 --label dass \
  --csv data/comprehensive_labels_stress.csv --max-duration 400"

run_one() {
    local seed="$1"
    local run_sub="${MODEL}_s${seed}"
    local run_dir="${EXP_ROOT}/${run_sub}"

    if [ -f "${run_dir}/summary.json" ]; then
        echo "[$(date +%H:%M)] SKIP ${run_sub} (already complete)"
        return 0
    fi

    mkdir -p "$run_dir"
    echo "[$(date +%H:%M)] START ${run_sub} on ${GPU}"

    timeout 1800 $PY train_trial.py \
        --run-id "studies/exp20_wang_split_llrd/${run_sub}" \
        --device "$GPU" \
        --extractor "$MODEL" \
        --norm "$NORM" \
        --lr "$LR" \
        --seed "$seed" \
        $COMMON \
        > "${EXP_ROOT}/logs/driver_${run_sub}.log" 2>&1
    local rc=$?

    if [ $rc -eq 124 ]; then
        echo "[$(date +%H:%M)] TIMEOUT ${run_sub}"
    elif [ $rc -ne 0 ]; then
        echo "[$(date +%H:%M)] FAIL ${run_sub} (exit=$rc)"
    elif [ -f "${run_dir}/summary.json" ]; then
        local ba=$($PY -c "import json; s=json.load(open('${run_dir}/summary.json')); print(s.get('bal_acc'))")
        echo "[$(date +%H:%M)] OK ${run_sub} (bal_acc=${ba})"
    else
        echo "[$(date +%H:%M)] FAIL ${run_sub} (no summary.json)"
    fi
    $PY -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
}

echo "=========================================="
echo "exp20 Wang-protocol ${MODEL} (LLRD=0.65, single split, seeds 0/1/2/42)"
echo "GPU: ${GPU} — $(date)"
echo "=========================================="

run_one 0
run_one 1
run_one 2
run_one 42

echo "=========================================="
echo "DONE exp20 ${MODEL} — $(date)"
echo "=========================================="
