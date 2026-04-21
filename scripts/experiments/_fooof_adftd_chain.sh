#!/bin/bash
# FOOOF ablation full chain for ADFTD (GPU 5, avoids GPU 7 non-FM queue).
#
# Per-window frozen features already cached (frozen_{fm}_adftd_perwindow.npz);
# Step 1 is therefore skipped here.
set -e
cd /raid/jupyter-linjimmy1003.md10/UCSD_stress
PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
export CUDA_VISIBLE_DEVICES=5
export PYTHONUNBUFFERED=1
LOGDIR=results/studies/fooof_ablation/logs
mkdir -p $LOGDIR

echo "=== $(date +%F_%T)  START FOOOF ADFTD chain ==="

# STEP 2 — FOOOF fit + signal reconstruction (norm=none: raw µV)
OUT_SIG=results/features_cache/fooof_ablation/adftd_norm_none.npz
if [ -f "$OUT_SIG" ]; then
  echo "  [$(date +%T)] skip FOOOF fit: $OUT_SIG exists"
else
  echo "  [$(date +%T)] FOOOF fit + ablation on adftd (norm=none)"
  $PY scripts/analysis/fooof_ablation.py --dataset adftd --norm none \
      > $LOGDIR/fooof_fit_adftd.log 2>&1
fi

# STEP 3 — FM feature extraction on ablated signals
for MODEL in labram cbramod reve; do
  SKIP=1
  for COND in aperiodic_removed periodic_removed both_removed; do
    OUT=results/features_cache/fooof_ablation/feat_${MODEL}_adftd_${COND}.npz
    if [ ! -f "$OUT" ]; then SKIP=0; fi
  done
  if [ $SKIP -eq 1 ]; then
    echo "  [$(date +%T)] skip extract-ablated: $MODEL all 3 conds exist"
  else
    echo "  [$(date +%T)] extract-ablated: $MODEL × adftd (3 conds)"
    $PY scripts/features/extract_fooof_ablated.py \
        --extractor $MODEL --dataset adftd --device cuda:0 \
        > $LOGDIR/extract_abl_${MODEL}_adftd.log 2>&1
  fi
done

# STEP 4 — probes (subject + state since ADFTD has AD/HC group label)
echo "  [$(date +%T)] probes: subject + state on adftd"
$PY scripts/experiments/fooof_ablation_probes.py --dataset adftd \
    > $LOGDIR/probes_adftd.log 2>&1

echo "=== $(date +%F_%T)  END FOOOF ADFTD chain ==="
echo "Results: results/studies/fooof_ablation/adftd_probes.json"
