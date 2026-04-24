#!/bin/bash
# Copy completed variance triangulation + temporal-block probe outputs
# to per-dataset final result folders.
#
# Source:
#   results/studies/exp33_temporal_block_probe/{cell}_probes.json
#   results/studies/exp32_variance_triangulation/{cell}_{permanova,cka}.json
# Dest:
#   results/final/{cell}/subject_probe_temporal_block/probes.json
#   results/final/{cell}/variance_triangulation/{permanova,cka}.json

set -e
cd "$(dirname "$0")/../.."

CELLS=(eegmat sleepdep stress adftd)

for DS in "${CELLS[@]}"; do
  SRC_PROBE=results/studies/exp33_temporal_block_probe/${DS}_probes.json
  SRC_PERM=results/studies/exp32_variance_triangulation/${DS}_permanova.json
  SRC_CKA=results/studies/exp32_variance_triangulation/${DS}_cka.json

  DEST_PROBE=results/final/${DS}/subject_probe_temporal_block
  DEST_VAR=results/final/${DS}/variance_triangulation

  mkdir -p $DEST_PROBE $DEST_VAR

  [ -f $SRC_PROBE ] && cp $SRC_PROBE $DEST_PROBE/probes.json && echo "-> $DEST_PROBE/probes.json"
  [ -f $SRC_PERM ]  && cp $SRC_PERM  $DEST_VAR/permanova.json && echo "-> $DEST_VAR/permanova.json"
  [ -f $SRC_CKA ]   && cp $SRC_CKA   $DEST_VAR/cka.json       && echo "-> $DEST_VAR/cka.json"
done

echo "SNAPSHOT DONE"
