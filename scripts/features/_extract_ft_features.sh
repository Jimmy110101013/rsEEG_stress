#!/bin/bash
# Safe sequential FT feature extraction for variance analysis.
# Single process, no parallelism, timeouts, no retries.
#
# Usage: nohup bash scripts/_extract_ft_features.sh > results/ft_extract.log 2>&1 &

set -o pipefail
export PYTHONUNBUFFERED=1

PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
cd /raid/jupyter-linjimmy1003.md10/UCSD_stress
GPU=cuda:6

COMMON="--mode ft --folds 5 --epochs 50 --patience 15 --batch-size 32 \
  --loss focal --head-hidden 128 --weight-decay 0.05 --grad-clip 2.0 \
  --warmup-epochs 3 --aug-overlap 0.75 --save-features"

run_one() {
    local name="$1"
    local timeout_sec="$2"
    shift 2
    local run_dir="results/features_cache/ft_${name}"

    # Skip if already done
    if [ -f "${run_dir}/summary.json" ] && [ -f "${run_dir}/fold5_features.npz" ]; then
        echo "[$(date +%H:%M)] SKIP $name (already complete)"
        return 0
    fi

    mkdir -p "$run_dir"
    echo "[$(date +%H:%M)] START $name (timeout=${timeout_sec}s)"

    timeout "$timeout_sec" $PY train_ft.py \
        --run-id "features_cache/ft_${name}" \
        --device $GPU \
        "$@" $COMMON \
        > "${run_dir}/driver.log" 2>&1
    local exit_code=$?

    if [ $exit_code -eq 124 ]; then
        echo "[$(date +%H:%M)] TIMEOUT $name (${timeout_sec}s exceeded)"
    elif [ $exit_code -ne 0 ]; then
        echo "[$(date +%H:%M)] FAIL $name (exit=$exit_code)"
    elif [ -f "${run_dir}/summary.json" ]; then
        echo "[$(date +%H:%M)] OK $name"
    else
        echo "[$(date +%H:%M)] FAIL $name (no summary.json)"
    fi

    # Clean GPU memory between runs
    sleep 5
    $PY -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
    return 0
}

echo "=========================================="
echo "FT Feature Extraction — $(date)"
echo "GPU: $GPU"
echo "=========================================="

# HP configs:
#   Stress: use best HP from sweep (per-model)
#   Other datasets: use canonical (no sweep done on those)
#
# Best HP from Stress sweep:
#   LaBraM:  lr=1e-4, elrs=1.0
#   CBraMod: lr=1e-5, elrs=0.1
#   REVE:    lr=3e-5, elrs=0.1

# ── LaBraM (norm=none; extractor does /100 internally — 2026-04-26 alignment) ──
echo ""
echo "--- LaBraM ---"
run_one "labram_stress"  1800  --extractor labram --norm none --lr 1e-4 --encoder-lr-scale 1.0 --label dass --csv data/comprehensive_labels.csv --seed 42
run_one "labram_adftd"   2700  --extractor labram --norm none --lr 1e-5 --encoder-lr-scale 0.1 --dataset adftd --seed 42
run_one "labram_eegmat"  1800  --extractor labram --norm none --lr 1e-5 --encoder-lr-scale 0.1 --dataset eegmat --seed 42
run_one "labram_tdbrain" 10800 --extractor labram --norm none --lr 1e-5 --encoder-lr-scale 0.1 --dataset tdbrain --seed 42

# ── CBraMod (norm=none) ──
echo ""
echo "--- CBraMod ---"
run_one "cbramod_stress"  1800  --extractor cbramod --norm none --lr 1e-5 --encoder-lr-scale 0.1 --label dass --csv data/comprehensive_labels.csv --seed 42
run_one "cbramod_adftd"   2700  --extractor cbramod --norm none --lr 1e-5 --encoder-lr-scale 0.1 --dataset adftd --seed 42
run_one "cbramod_eegmat"  1800  --extractor cbramod --norm none --lr 1e-5 --encoder-lr-scale 0.1 --dataset eegmat --seed 42
run_one "cbramod_tdbrain" 10800 --extractor cbramod --norm none --lr 1e-5 --encoder-lr-scale 0.1 --dataset tdbrain --seed 42

# ── REVE (norm=none) ──
echo ""
echo "--- REVE ---"
run_one "reve_stress"  1800  --extractor reve --norm none --lr 3e-5 --encoder-lr-scale 0.1 --label dass --csv data/comprehensive_labels.csv --seed 42
run_one "reve_adftd"   2700  --extractor reve --norm none --lr 3e-5 --encoder-lr-scale 0.1 --dataset adftd --seed 42
run_one "reve_eegmat"  1800  --extractor reve --norm none --lr 3e-5 --encoder-lr-scale 0.1 --dataset eegmat --seed 42
run_one "reve_tdbrain" 10800 --extractor reve --norm none --lr 3e-5 --encoder-lr-scale 0.1 --dataset tdbrain --seed 42

echo ""
echo "=========================================="
echo "ALL DONE — $(date)"
echo "=========================================="

# Final status check
echo ""
echo "Completion status:"
for d in results/features_cache/ft_*/; do
    name=$(basename "$d")
    if [ -f "$d/summary.json" ] && [ -f "$d/fold5_features.npz" ]; then
        echo "  OK: $name"
    else
        echo "  MISSING: $name"
    fi
done
