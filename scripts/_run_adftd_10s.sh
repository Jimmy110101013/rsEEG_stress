#!/bin/bash
set -e
PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
cd /raid/jupyter-linjimmy1003.md10/UCSD_stress
DIR="studies/exp12_adftd_10s_multiseed"
COMMON="--mode ft --folds 5 --epochs 50 --patience 15 --batch-size 32 --loss focal --head-hidden 128 --weight-decay 0.05 --grad-clip 2.0 --warmup-epochs 3 --aug-overlap 0.75 --save-features --window-sec 10"
GPU=$1
shift

for SPEC in "$@"; do
    MODEL=$(echo $SPEC | cut -d: -f1)
    SEED=$(echo $SPEC | cut -d: -f2)

    if [ "$MODEL" = "labram" ]; then
        NORM="zscore"; LR="1e-5"
    elif [ "$MODEL" = "cbramod" ]; then
        NORM="none"; LR="1e-5"
    elif [ "$MODEL" = "reve" ]; then
        NORM="none"; LR="3e-5"
    fi

    RUN="${MODEL}_s${SEED}"
    RUNDIR="results/$DIR/$RUN"

    if [ -f "$RUNDIR/summary.json" ]; then
        echo "[$(date +%H:%M)] SKIP $RUN (done)"
        continue
    fi

    echo "[$(date +%H:%M)] START $RUN on $GPU"
    $PY train_ft.py --run-id "$DIR/$RUN" \
        --extractor $MODEL --norm $NORM --lr $LR --encoder-lr-scale 0.1 \
        --dataset adftd --seed $SEED --device $GPU $COMMON \
        2>&1
    echo "[$(date +%H:%M)] DONE $RUN"
done
echo "[$(date +%H:%M)] GPU $GPU ALL DONE"
