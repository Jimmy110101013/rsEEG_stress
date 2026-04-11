"""EEG-Conformer configuration (Song et al. 2023).

Reference:
    Song, Y., Zheng, Q., Liu, B., & Gao, X. (2023). EEG Conformer:
    Convolutional Transformer for EEG Decoding and Visualization.
    IEEE Transactions on Neural Systems and Rehabilitation Engineering,
    31, 710-719. https://doi.org/10.1109/TNSRE.2022.3230250

Official implementation: https://github.com/eeyhsong/EEG-Conformer

Defaults follow the original paper's BCI-IV 2a setting:
    n_filters = 40            (shallow conv front-end width, reused as d_model)
    filter_time_length = 25
    pool_time_length = 75
    pool_time_stride = 15
    n_transformer_layers = 6
    n_heads = 10              (d_model=40 must be divisible by n_heads)
    forward_expansion = 4     (d_ff = forward_expansion * d_model = 160)
    transformer_dropout = 0.5
    embed_dropout = 0.5       (dropout after the AvgPool inside the CNN front-end)

Window settings match LaBraM / EEGNet (`window_sec=5.0`, `sample_rate=200`)
so the 5-fold subject-level CV comparison is run on identical windows.
"""
from dataclasses import dataclass

from baseline.abstract.base_config import BaseModelConfig


@dataclass
class EEGConformerModelConfig(BaseModelConfig):
    """EEG-Conformer supervised-from-scratch baseline configuration."""

    model_name: str = "eegconformer"

    # Match LaBraM's input windowing for a fair same-window comparison.
    sample_rate: int = 200
    n_channels: int = 30
    patch_size: int = 200
    window_sec: float = 5.0

    # Shallow conv front-end (shares spirit with ShallowConvNet)
    n_filters: int = 40           # == d_model for the transformer
    filter_time_length: int = 25
    pool_time_length: int = 75
    pool_time_stride: int = 15
    embed_dropout: float = 0.5    # dropout inside the conv front-end

    # Transformer encoder
    n_transformer_layers: int = 6
    n_heads: int = 10             # n_filters must be divisible by n_heads
    forward_expansion: int = 4    # d_ff = forward_expansion * d_model
    transformer_dropout: float = 0.5

    # Effective embedding dimension after flatten; computed dynamically.
    embed_dim: int = 0
