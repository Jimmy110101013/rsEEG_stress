"""Common 19-channel infrastructure for cross-dataset comparison.

The 19 standard 10-20 channels are the intersection across:
- UCSD Stress (30 channels)
- ADFTD / OpenNeuro ds004504 (19 channels)
- TUAB / Temple University Abnormal EEG (21+ channels)

All 19 channels are present in LaBraM's STANDARD_1020 position embedding table.
"""

import numpy as np

# `mne` is imported lazily inside select_channels() so this module can be
# imported on lightweight environments (e.g., HDF5-only loaders that never
# touch MNE Raw objects) without paying the scipy.spatial import cost.


# Standard 10-20 channels present in all datasets
COMMON_19 = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T3', 'C3', 'CZ', 'C4', 'T4',
    'T5', 'P3', 'PZ', 'P4', 'T6',
    'O1', 'O2',
]

# Aliases: newer 10-10 names → older 10-20 names used in COMMON_19
# Also handles case variations and TUAB "EEG FP1-REF" style prefixes
CHANNEL_ALIASES = {
    # 10-10 → 10-20 mapping
    'T7': 'T3',
    'T8': 'T4',
    'P7': 'T5',
    'P8': 'T6',
    # Case variations
    'Fp1': 'FP1', 'Fp2': 'FP2', 'FP1': 'FP1', 'FP2': 'FP2',
    'Fz': 'FZ', 'Cz': 'CZ', 'Pz': 'PZ', 'Oz': 'OZ',
    'F3': 'F3', 'F4': 'F4', 'F7': 'F7', 'F8': 'F8',
    'C3': 'C3', 'C4': 'C4',
    'P3': 'P3', 'P4': 'P4',
    'O1': 'O1', 'O2': 'O2',
    'T3': 'T3', 'T4': 'T4', 'T5': 'T5', 'T6': 'T6',
}


def normalize_channel_name(name: str) -> str:
    """Normalize a channel name to standard uppercase 10-20 format.

    Handles:
    - TUAB format: "EEG FP1-REF", "EEG FP1-LE" → "FP1"
    - Case variations: "Fp1" → "FP1", "Fz" → "FZ"
    - 10-10 aliases: "T7" → "T3", "P7" → "T5"
    """
    # Strip TUAB-style prefixes and suffixes
    name = name.strip()
    if name.startswith('EEG '):
        name = name[4:]
    for suffix in ['-REF', '-LE', '-AR', '-AVG']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    name = name.strip()

    # Try alias lookup first (handles case + 10-10 mapping)
    upper = name.upper()
    if name in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[name]
    if upper in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[upper]

    return upper


def select_channels(raw, target_channels: list[str] = None):
    """Pick and reorder channels from MNE Raw to match target channel list.

    Args:
        raw: MNE Raw object with any channel naming convention.
        target_channels: Desired channel list (default: COMMON_19).

    Returns:
        MNE Raw with only the target channels, in order.

    Raises:
        ValueError: If any target channel is missing from the recording.
    """
    import mne  # noqa: F401  (kept for type clarity, used by raw.copy())
    if target_channels is None:
        target_channels = COMMON_19

    # Build mapping: normalized name → original name in raw
    raw_ch_map = {}
    for ch in raw.ch_names:
        norm = normalize_channel_name(ch)
        raw_ch_map[norm] = ch

    # Find original names for each target channel
    pick_names = []
    missing = []
    for target in target_channels:
        target_norm = normalize_channel_name(target)
        if target_norm in raw_ch_map:
            pick_names.append(raw_ch_map[target_norm])
        else:
            missing.append(target)

    if missing:
        available = sorted(raw_ch_map.keys())
        raise ValueError(
            f"Missing channels: {missing}. "
            f"Available (normalized): {available}"
        )

    # Pick and reorder
    raw_picked = raw.copy().pick_channels(pick_names, ordered=True)

    # Rename to standard names
    rename_map = {orig: target for orig, target in zip(pick_names, target_channels)}
    raw_picked.rename_channels(rename_map)

    return raw_picked
