"""DeepConvNet configuration (Schirrmeister et al. 2017).

Reference:
    Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
    Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F.,
    Burgard, W., & Ball, T. (2017). Deep learning with convolutional
    neural networks for EEG decoding and visualization. Human Brain
    Mapping, 38(11), 5391-5420. https://doi.org/10.1002/hbm.23730

Defaults follow the original paper / Braindecode `Deep4Net`:
    block 1: Conv(1 -> 25, (1, 10)) + Conv(25 -> 25, (C, 1)) + BN + ELU + MaxPool((1,3),3)
    block 2: Conv(25 -> 50, (1, 10))  + BN + ELU + MaxPool((1,3),3)
    block 3: Conv(50 -> 100, (1, 10)) + BN + ELU + MaxPool((1,3),3)
    block 4: Conv(100 -> 200, (1, 10)) + BN + ELU + MaxPool((1,3),3)
    dropout = 0.5

Window settings match LaBraM / EEGNet (`window_sec=5.0`, `sample_rate=200`)
so the 5-fold subject-level CV comparison is run on identical windows.
"""
from dataclasses import dataclass

from baseline.abstract.base_config import BaseModelConfig


@dataclass
class DeepConvNetModelConfig(BaseModelConfig):
    """DeepConvNet supervised-from-scratch baseline configuration."""

    model_name: str = "deepconvnet"

    # Match LaBraM's input windowing for a fair same-window comparison.
    sample_rate: int = 200
    n_channels: int = 30
    patch_size: int = 200
    window_sec: float = 5.0

    # DeepConvNet block channel widths (Schirrmeister 2017 defaults)
    n_filters_1: int = 25
    n_filters_2: int = 50
    n_filters_3: int = 100
    n_filters_4: int = 200

    filter_time_length: int = 10   # temporal conv kernel length per block
    pool_time_length: int = 3      # temporal max-pool length per block
    pool_time_stride: int = 3      # temporal max-pool stride per block

    dropout: float = 0.5

    # Effective embedding dimension after flatten; computed dynamically.
    embed_dim: int = 0
