#!/bin/bash
# exp24 — Diagnostic: does the LaBraM self.norm → Identity fix change
# Wang-protocol reproduction?
#
# Same 12 seeds as exp23, same hyperparameters (Wang-full protocol).
# Only difference: LaBraM model.py L194 now uses nn.Identity() for self.norm
# (matching original use_mean_pooling=True behavior).
#
# Usage: bash scripts/_run_exp24_labram_fixed.sh <gpu> <seed> [<seed> ...]

set -o pipefail
export PYTHONUNBUFFERED=1

GPU="$1"; shift
if [ -z "$GPU" ] || [ $# -eq 0 ]; then
    echo "Usage: $0 <cuda:N> <seed> [<seed> ...]"
    exit 2
fi

PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
cd /raid/jupyter-linjimmy1003.md10/UCSD_stress

EXP_ROOT="results/studies/exp24_labram_fixed"
mkdir -p "$EXP_ROOT/logs"

COMMON="--mode ft --split-mode wang --folds 5 --epochs 50 --patience 50 \
  --batch-size 32 --loss ce --val-metric win_acc --weight-decay 0.05 \
  --grad-clip 1.0 --warmup-epochs 3 \
  --aug-overlap 0.75 --eval-aug-overlap 0.75 \
  --layer-decay 0.65 --label dass \
  --csv data/comprehensive_labels_stress.csv --max-duration 400 \
  --extractor labram --norm zscore --lr 1e-5"

run_one() {
    local seed="$1"
    local run_sub="labram_s${seed}"
    local run_dir="${EXP_ROOT}/${run_sub}"

    if [ -f "${run_dir}/summary.json" ]; then
        echo "[$(date +%H:%M)] SKIP ${run_sub} (already complete)"
        return 0
    fi

    mkdir -p "$run_dir"
    echo "[$(date +%H:%M)] START ${run_sub} on ${GPU}"

    timeout 1800 $PY train_trial.py \
        --run-id "studies/exp24_labram_fixed/${run_sub}" \
        --device "$GPU" --seed "$seed" $COMMON \
        > "${EXP_ROOT}/logs/driver_${run_sub}.log" 2>&1
    local rc=$?

    if [ $rc -ne 0 ]; then
        echo "[$(date +%H:%M)] FAIL ${run_sub} (exit=$rc)"
    elif [ -f "${run_dir}/summary.json" ]; then
        local metrics=$($PY -c "
import json
s = json.load(open('${run_dir}/summary.json'))
print(f\"rec={s.get('bal_acc')}  win={s.get('win_bal_acc_mean')}\")
")
        echo "[$(date +%H:%M)] OK ${run_sub}: ${metrics}"
    fi
    $PY -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
}

echo "=========================================="
echo "exp24 LaBraM (fc_norm-only) diagnostic on ${GPU} — $(date)"
echo "Seeds: $@"
echo "=========================================="

for seed in "$@"; do
    run_one "$seed"
done

echo "=========================================="
echo "DONE ${GPU} — $(date)"
echo "=========================================="
