#!/bin/bash
# TDBRAIN: Wait for download, unzip, inspect, and preprocess
set -e

DATA_DIR="/raid/jupyter-linjimmy1003.md10/UCSD_stress/data/tdbrain"
ZIP_FILE="$DATA_DIR/TDBRAIN-dataset-derivatives.zip"
LOG_FILE="$DATA_DIR/setup.log"
PASSWORD='!Bra1n$Rgr3at:)'

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

# ── Step 1: Wait for download to complete ──
log "Waiting for download to complete..."
prev_size=0
stable_count=0
while true; do
    curr_size=$(stat -c%s "$ZIP_FILE" 2>/dev/null || echo 0)
    if [ "$curr_size" -eq "$prev_size" ] && [ "$curr_size" -gt 0 ]; then
        stable_count=$((stable_count + 1))
    else
        stable_count=0
    fi
    if [ "$stable_count" -ge 6 ]; then
        log "Download complete: $(du -h "$ZIP_FILE" | cut -f1)"
        break
    fi
    prev_size=$curr_size
    sleep 10
done

# ── Step 2: Unzip with password ──
log "Unzipping derivatives (this may take a while)..."
cd "$DATA_DIR"
unzip -o -P "$PASSWORD" "$ZIP_FILE" -d "$DATA_DIR" >> "$LOG_FILE" 2>&1
log "Unzip complete."

# ── Step 3: Inspect structure ──
log "=== Directory structure (first 3 levels) ==="
find "$DATA_DIR" -maxdepth 4 -type d | head -50 | tee -a "$LOG_FILE"

log "=== Sample CSV files ==="
SAMPLE_CSV=$(find "$DATA_DIR" -name "*.csv" -type f | head -1)
if [ -n "$SAMPLE_CSV" ]; then
    log "First CSV found: $SAMPLE_CSV"
    log "=== CSV header (first line) ==="
    head -1 "$SAMPLE_CSV" | tee -a "$LOG_FILE"
    log "=== CSV shape (lines x cols) ==="
    LINES=$(wc -l < "$SAMPLE_CSV")
    COLS=$(head -1 "$SAMPLE_CSV" | tr ',' '\n' | wc -l)
    log "Lines: $LINES, Columns: $COLS"
    log "=== First 3 data rows ==="
    head -4 "$SAMPLE_CSV" | tee -a "$LOG_FILE"
else
    log "No CSV files found! Checking for other formats..."
    find "$DATA_DIR" -type f | head -20 | tee -a "$LOG_FILE"
fi

log "=== File count by extension ==="
find "$DATA_DIR" -type f | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -10 | tee -a "$LOG_FILE"

log "=== Total disk usage ==="
du -sh "$DATA_DIR" | tee -a "$LOG_FILE"

log "Setup complete. Ready for preprocessing."
