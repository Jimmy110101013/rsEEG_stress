#!/usr/bin/env bash
# GPU 4 queue: waits until the current LaBraM ADFTD no-aug audit on cuda:4
# finishes, then runs REVE ADFTD no-aug × 3 seeds.
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "[$(date +%H:%M:%S)] Waiting for any train_ft.py on cuda:4 to finish..."
while pgrep -af "train_ft.py.*cuda:4" >/dev/null 2>&1; do
  sleep 60
done
echo "[$(date +%H:%M:%S)] cuda:4 clear; launching REVE ADFTD no-aug queue."

export GPU=4
FM=reve DATASET=adftd AUG= SEEDS="42 123 2024" bash scripts/experiments/exp31_aug_audit_chain.sh
