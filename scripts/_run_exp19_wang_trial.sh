#!/bin/bash
# Supplementary: Wang 2025 benchmark reference — trial-level CV on 82 recordings
# (comprehensive_labels_stress.csv + --max-duration 400). Same --label dass
# (per-rec Group column) as exp18, only the recording set differs (82 vs 70).
#
# Usage: bash scripts/_run_exp19_wang_trial.sh <model> <device>
#   model: labram | cbramod | reve
#   device: cuda:N

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

EXP_ROOT="results/studies/exp19_wang_82rec_trial"
mkdir -p "$EXP_ROOT/logs"

# Match exp18 recipe (Wang reference: lr=1e-5 labram/cbramod, 3e-5 reve)
case "$MODEL" in
    labram)  NORM=zscore; LR=1e-5 ;;
    cbramod) NORM=none;   LR=1e-5 ;;
    reve)    NORM=none;   LR=3e-5 ;;
    *) echo "Unknown model $MODEL"; exit 2 ;;
esac

COMMON="--mode ft --folds 5 --epochs 50 --patience 15 --batch-size 32 \
  --loss focal --weight-decay 0.05 --grad-clip 1.0 --warmup-epochs 3 \
  --aug-overlap 0.75 --label dass \
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
        --run-id "studies/exp19_wang_82rec_trial/${run_sub}" \
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
        local ba=$($PY -c "import json; s=json.load(open('${run_dir}/summary.json')); print(s.get('subject_bal_acc') or s.get('bal_acc'))")
        echo "[$(date +%H:%M)] OK ${run_sub} (bal_acc=${ba})"
    else
        echo "[$(date +%H:%M)] FAIL ${run_sub} (no summary.json)"
    fi
    $PY -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
}

echo "=========================================="
echo "exp19 Wang 82-rec trial ${MODEL} 3-seed — $(date)"
echo "GPU: ${GPU}"
echo "=========================================="

run_one 42
run_one 123
run_one 2024

echo "=========================================="
echo "DONE exp19 ${MODEL} — $(date)"
echo "=========================================="
