"""EEGNet configuration (Lawhern et al. 2018).

Reference:
    Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M.,
    Hung, C. P., & Lance, B. J. (2018). EEGNet: a compact convolutional
    neural network for EEG-based brain-computer interfaces. Journal of
    Neural Engineering, 15(5), 056013.
    https://doi.org/10.1088/1741-2552/aace8c

Defaults follow the original paper:
    F1 = 8 (temporal filters)
    D  = 2 (depth multiplier)
    F2 = F1 * D = 16 (separable conv filters)
    kernel_length = half the sampling rate (temporal conv receptive field)
    dropout = 0.25

Window settings match LaBraM (`window_sec=5.0`, `sample_rate=200`) so
the 5-fold subject-level CV comparison is run on identical windows.
"""
from dataclasses import dataclass

from baseline.abstract.base_config import BaseModelConfig


@dataclass
class EEGNetModelConfig(BaseModelConfig):
    """EEGNet supervised-from-scratch baseline configuration."""

    model_name: str = "eegnet"

    # Match LaBraM's input windowing for a fair same-window comparison.
    sample_rate: int = 200
    n_channels: int = 30
    patch_size: int = 200     # unused by EEGNet but kept for BaseModelConfig contract
    window_sec: float = 5.0   # same 1000-sample window LaBraM sees

    # EEGNet block-1 hyperparameters (Lawhern 2018 defaults)
    F1: int = 8               # temporal filters
    D: int = 2                # depth multiplier for spatial depthwise conv
    F2: int = 16              # separable conv output channels (= F1 * D)
    kernel_length: int = 100  # temporal conv kernel (~= sample_rate/2 → 500 ms)

    # Block-1 and block-2 pooling strides (paper defaults)
    pool1: int = 4            # temporal avgpool after spatial depthwise
    pool2: int = 8            # temporal avgpool after separable conv

    dropout: float = 0.25
    dropout_type: str = "Dropout"  # 'Dropout' or 'SpatialDropout2D' (paper)

    # Effective embedding dimension after flatten:
    # T → T // pool1 → T // (pool1 * pool2) samples × F2 channels.
    # Computed dynamically in the extractor so it adapts if window_sec changes.
    embed_dim: int = 0  # filled in by extractor __init__
