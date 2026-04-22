#!/usr/bin/env bash
# GPU 3 sequential queue: EEGMAT LaBraM no-aug {s123,s2024}
# then CBraMod ADFTD no-aug {s42,s123,s2024}.
set -euo pipefail
cd "$(dirname "$0")/../.."
export GPU=3
FM=labram  DATASET=eegmat AUG= SEEDS="123 2024"  bash scripts/experiments/exp31_aug_audit_chain.sh
FM=cbramod DATASET=adftd  AUG= SEEDS="42 123 2024" bash scripts/experiments/exp31_aug_audit_chain.sh
