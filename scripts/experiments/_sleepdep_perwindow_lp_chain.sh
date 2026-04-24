#!/bin/bash
# SleepDep per-window LP chain — all 3 FMs sequentially (CPU bound, no GPU contention).
set -e
cd /raid/jupyter-linjimmy1003.md10/UCSD_stress
PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
LOGDIR=results/studies/perwindow_lp_all/logs
mkdir -p $LOGDIR
echo "=== $(date +%F_%T)  START SleepDep per-window LP chain ==="
for MODEL in labram cbramod reve; do
  echo "  [$(date +%T)] $MODEL × sleepdep"
  $PY train_lp.py \
      --extractor $MODEL --dataset sleepdep \
      > $LOGDIR/sleepdep_${MODEL}.log 2>&1
done
echo "=== $(date +%F_%T)  END SleepDep per-window LP chain ==="
echo "Results: results/studies/perwindow_lp_all/sleepdep/"
