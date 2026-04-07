#!/bin/bash
# Download OpenNeuro ds004504 (ADFTD - Alzheimer's/FTD EEG)
# Source: https://openneuro.org/datasets/ds004504/versions/1.0.7
# 88 subjects (36 AD, 23 FTD, 29 healthy), 19ch, 500Hz, eyes-closed resting-state
# ~5.4 GB total

DEST="data/adftd"
BASE="https://s3.amazonaws.com/openneuro.org/ds004504"

mkdir -p "$DEST"

# Top-level metadata
echo "Downloading metadata..."
wget -q -O "$DEST/participants.tsv" "$BASE/participants.tsv"
wget -q -O "$DEST/participants.json" "$BASE/participants.json"
wget -q -O "$DEST/dataset_description.json" "$BASE/dataset_description.json"

# All 88 subjects' EEG files (parallel, 8 concurrent)
# Note: subject IDs are zero-padded to 3 digits (sub-001 through sub-088)
echo "Downloading 88 subjects..."
for i in $(seq 1 88); do
  SUB=$(printf "sub-%03d" $i)
  mkdir -p "$DEST/$SUB/eeg"
  wget -q -O "$DEST/$SUB/eeg/${SUB}_task-eyesclosed_eeg.set" \
    "$BASE/$SUB/eeg/${SUB}_task-eyesclosed_eeg.set" &
  wget -q -O "$DEST/$SUB/eeg/${SUB}_task-eyesclosed_channels.tsv" \
    "$BASE/$SUB/eeg/${SUB}_task-eyesclosed_channels.tsv" &
  wget -q -O "$DEST/$SUB/eeg/${SUB}_task-eyesclosed_eeg.json" \
    "$BASE/$SUB/eeg/${SUB}_task-eyesclosed_eeg.json" &
  # Limit concurrency to 8
  [ $(jobs -r | wc -l) -ge 8 ] && wait -n
done
wait

echo "Done. $(du -sh $DEST)"
