"""ShallowConvNet configuration (Schirrmeister et al. 2017).

Reference:
    Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
    Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F.,
    Burgard, W., & Ball, T. (2017). Deep learning with convolutional
    neural networks for EEG decoding and visualization. Human Brain
    Mapping, 38(11), 5391-5420. https://doi.org/10.1002/hbm.23730

Defaults follow the original paper / Braindecode reference implementation:
    n_filters_time = 40   (temporal conv output channels)
    filter_time_length = 25
    n_filters_spat = 40   (spatial conv output channels, depth tied to temporal)
    pool_time_length = 75
    pool_time_stride = 15
    dropout = 0.5

Window settings match LaBraM / EEGNet (`window_sec=5.0`, `sample_rate=200`)
so the 5-fold subject-level CV comparison is run on identical windows.
"""
from dataclasses import dataclass

from baseline.abstract.base_config import BaseModelConfig


@dataclass
class ShallowConvNetModelConfig(BaseModelConfig):
    """ShallowConvNet supervised-from-scratch baseline configuration."""

    model_name: str = "shallowconvnet"

    # Match LaBraM's input windowing for a fair same-window comparison.
    sample_rate: int = 200
    n_channels: int = 30
    patch_size: int = 200     # unused by ShallowConvNet but required by BaseModelConfig
    window_sec: float = 5.0   # same 1000-sample window LaBraM sees

    # ShallowConvNet block-1 hyperparameters (Schirrmeister 2017 defaults)
    n_filters_time: int = 40      # temporal conv output channels
    filter_time_length: int = 25  # temporal conv kernel length
    n_filters_spat: int = 40      # spatial conv output channels

    # Temporal pooling after the square activation (paper defaults)
    pool_time_length: int = 75
    pool_time_stride: int = 15

    dropout: float = 0.5

    # Effective embedding dimension after flatten; computed dynamically in
    # the extractor so it adapts if window_sec changes.
    embed_dim: int = 0  # filled in by extractor __init__
