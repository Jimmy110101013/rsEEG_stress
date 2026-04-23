#!/bin/bash
# FOOOF ablation full chain for ADFTD — split1 binary, per-FM window
# (2026-04-23 refresh, G-F11/F12). labram/cbramod at w=5s, reve at w=10s.
# FOOOF fit is run once per window; extract dispatches per-FM.
set -e
cd /raid/jupyter-linjimmy1003.md10/UCSD_stress
PY=/raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python
: "${GPU:=5}"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1
LOGDIR=results/studies/fooof_ablation/logs
mkdir -p $LOGDIR

echo "=== $(date +%F_%T)  START FOOOF ADFTD chain (split1, per-FM window, GPU=$GPU) ==="

# Step 1 — FOOOF fit + signal reconstruction, once per window
for W in 5 10; do
  OUT_SIG=results/features_cache/fooof_ablation/adftd_norm_none_w${W}.npz
  if [ -f "$OUT_SIG" ]; then
    echo "  [$(date +%T)] skip FOOOF fit w=${W}: $OUT_SIG exists"
  else
    echo "  [$(date +%T)] FOOOF fit + ablation on adftd split1 (norm=none, w=${W})"
    $PY scripts/analysis/fooof_ablation.py --dataset adftd --norm none --window-sec ${W} \
        > $LOGDIR/fooof_fit_adftd_w${W}.log 2>&1
  fi
done

# Step 2 — FM feature extraction on ablated signals (per-FM window via MODEL_WINDOW)
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

# Step 3 — probes (subject + state, ADFTD has AD/HC group label)
echo "  [$(date +%T)] probes: subject + state on adftd"
$PY scripts/experiments/fooof_ablation_probes.py --dataset adftd \
    > $LOGDIR/probes_adftd.log 2>&1

echo "=== $(date +%F_%T)  END FOOOF ADFTD chain ==="
echo "Results: results/studies/fooof_ablation/adftd_probes.json"
