#!/bin/bash
# exp23 — LaBraM, strictest Wang-protocol reproduction.
# Changes over exp22:
#   - Eval-time augmentation: --eval-aug-overlap 0.75 (val+test increase class
#     gets 75% overlap aug, matching Wang's Table I sample counts).
# Carries over from exp22:
#   - Loss: CE (was focal)
#   - Val metric: win_acc (was rec_bal)
#   - LLRD 0.65, Wang single split, 15/2/2 + 50/6/7 counts
#
# Window counts should match Wang:
#   train ~6,630 (we get 6,497)
#   val   ~886   (we get 873)
#   test  ~846   (we get 922 — slightly high due to split-specific rec lengths)
#
# Usage: bash scripts/_run_exp23_wang_full.sh <gpu> <seed> [<seed> ...]

set -o pipefail
export PYTHONUNBUFFERED=1

GPU="$1"; shift
if [ -z "$GPU" ] || [ $# -eq 0 ]; then
    echo "Usage: $0 <cuda:N> <seed> [<seed> ...]"
    exit 2
fi

PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
cd /raid/jupyter-linjimmy1003.md10/UCSD_stress

EXP_ROOT="results/studies/exp23_wang_full"
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
        --run-id "studies/exp23_wang_full/${run_sub}" \
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
echo "exp23 LaBraM Wang-full-aug on ${GPU} — $(date)"
echo "Seeds: $@"
echo "=========================================="

for seed in "$@"; do
    run_one "$seed"
done

echo "=========================================="
echo "DONE ${GPU} — $(date)"
echo "=========================================="
