#!/bin/bash
# Download TUAB (Temple University Abnormal EEG Corpus) v3.0.1
#
# REQUIRES: TUH Corpus account registration
# 1. Go to https://isip.piconepress.com/projects/tuh_eeg/
# 2. Fill out the application PDF and email to help@nedcdata.org
# 3. Wait 24-48 hours for approval
# 4. After receiving SSH credentials, run this script
#
# Size: ~60 GB total (train + eval)
# For our experiment: we only need ~150 subjects (~2-3 GB)

DEST="data/tuab"
mkdir -p "$DEST"

echo "=== TUAB Download ==="
echo "This requires TUH Corpus SSH credentials."
echo "Apply at: https://isip.piconepress.com/projects/tuh_eeg/"
echo ""
echo "After approval, run:"
echo "  rsync -auxvL nedc@www.isip.piconepress.com:data/eeg/tuh_eeg_abnormal/v3.0.1/ $DEST/"
echo ""
echo "Or download a subset (train split only):"
echo "  rsync -auxvL nedc@www.isip.piconepress.com:data/eeg/tuh_eeg_abnormal/v3.0.1/edf/train/ $DEST/edf/train/"
