#!/bin/bash
# FOOOF ablation full chain for SleepDep (GPU 7).
#
# Steps:
#   1. Extract per-window frozen cache (3 FMs × sleepdep) if missing
#   2. Run FOOOF fit + ablated-signal reconstruction for sleepdep (norm=none — SleepDep uses 19ch raw)
#   3. Extract FM features on ablated signals (3 FMs × 3 conditions = 9 runs)
#   4. Run subject + state probes → results/studies/fooof_ablation/sleepdep_probes.json
set -e
cd /raid/jupyter-linjimmy1003.md10/UCSD_stress
PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1
LOGDIR=results/studies/fooof_ablation/logs
mkdir -p $LOGDIR

echo "=== $(date +%F_%T)  START FOOOF SleepDep chain ==="

# STEP 1 — per-window frozen cache (skip if exists)
for MODEL in labram cbramod reve; do
  OUT=results/features_cache/frozen_${MODEL}_sleepdep_perwindow.npz
  if [ -f "$OUT" ]; then
    echo "  [$(date +%T)] skip per-window extract: $OUT exists"
  else
    echo "  [$(date +%T)] extract per-window: $MODEL × sleepdep"
    $PY scripts/features/extract_frozen_all_perwindow.py \
        --extractor $MODEL --dataset sleepdep --device cuda:0 \
        > $LOGDIR/extract_pw_${MODEL}.log 2>&1
  fi
done

# STEP 2 — FOOOF fit + signal reconstruction (norm=none: raw µV)
OUT_SIG=results/features_cache/fooof_ablation/sleepdep_norm_none.npz
if [ -f "$OUT_SIG" ]; then
  echo "  [$(date +%T)] skip FOOOF fit: $OUT_SIG exists"
else
  echo "  [$(date +%T)] FOOOF fit + ablation on sleepdep (norm=none)"
  $PY scripts/analysis/fooof_ablation.py --dataset sleepdep --norm none \
      > $LOGDIR/fooof_fit.log 2>&1
fi

# STEP 3 — FM feature extraction on ablated signals
for MODEL in labram cbramod reve; do
  SKIP=1
  for COND in aperiodic_removed periodic_removed both_removed; do
    OUT=results/features_cache/fooof_ablation/feat_${MODEL}_sleepdep_${COND}.npz
    if [ ! -f "$OUT" ]; then SKIP=0; fi
  done
  if [ $SKIP -eq 1 ]; then
    echo "  [$(date +%T)] skip extract-ablated: $MODEL all 3 conds exist"
  else
    echo "  [$(date +%T)] extract-ablated: $MODEL × sleepdep (3 conds)"
    $PY scripts/features/extract_fooof_ablated.py \
        --extractor $MODEL --dataset sleepdep --device cuda:0 \
        > $LOGDIR/extract_abl_${MODEL}.log 2>&1
  fi
done

# STEP 4 — probes (subject + state since sleepdep has NS/SD state label)
echo "  [$(date +%T)] probes: subject + state on sleepdep"
$PY scripts/experiments/fooof_ablation_probes.py --dataset sleepdep \
    > $LOGDIR/probes.log 2>&1

echo "=== $(date +%F_%T)  END FOOOF SleepDep chain ==="
echo "Results: results/studies/fooof_ablation/sleepdep_probes.json"
