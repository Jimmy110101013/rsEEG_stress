"""LaBraM channel mapping — maps EEG channel names to position embedding indices.

LaBraM's position_embedding has 129 slots (128 channels + 1 CLS token at index 0).
The channel order follows the standard_1020 list from the original LaBraM repo.
"""

# Standard 10-20 extended channel order from LaBraM (120 channels)
# Source: github.com/935963004/LaBraM/blob/main/utils.py
STANDARD_1020 = [
    'FP1', 'FPZ', 'FP2',
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
]

# Our 30-channel montage (from data/*.set files)
OUR_CHANNELS = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3', 'FCZ',
    'FC4', 'FT8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'TP7', 'CP3', 'CPZ',
    'CP4', 'TP8', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'OZ', 'O2',
]


def get_input_chans(channel_names: list[str] = None) -> list[int]:
    """Convert channel names to LaBraM position embedding indices.

    Returns list starting with 0 (CLS token), followed by 1-indexed
    channel positions in STANDARD_1020.

    Source: github.com/935963004/LaBraM/blob/main/utils.py#L713
    """
    if channel_names is None:
        channel_names = OUR_CHANNELS

    input_chans = [0]  # CLS token
    for ch in channel_names:
        ch_upper = ch.upper()
        if ch_upper not in STANDARD_1020:
            raise ValueError(f"Channel '{ch}' not in LaBraM standard_1020 list")
        input_chans.append(STANDARD_1020.index(ch_upper) + 1)
    return input_chans
