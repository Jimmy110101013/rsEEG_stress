"""ShallowConvNet extractor — Schirrmeister et al. 2017 compact CNN.

Architecture (input (B, C, T), typically (B, 30, 1000) for 5 s @ 200 Hz):

    Conv2D(1 -> F_t, kernel=(1, filter_time_length))        # temporal filters
    Conv2D(F_t -> F_s, kernel=(C, 1), bias=False)           # spatial filters
    BatchNorm2d(F_s)
    Square activation (x^2)                                 # log-power front-end
    AvgPool2d(kernel=(1, pool_time_length), stride=(1, pool_time_stride))
    SafeLog activation (log(max(x, eps)))
    Dropout(dropout)

    Flatten -> (B, F_s * T_out)

`SafeLog` mirrors Braindecode's implementation: clamp the pooled signal
to a small positive floor before taking log so gradients are finite even
when the squared input averages to zero early in training.

Reference:
    Schirrmeister et al. 2017, Human Brain Mapping 38(11):5391-5420.
    Braindecode reference: https://braindecode.org (`ShallowFBCSPNet`).
"""

import torch
import torch.nn as nn
from torch import Tensor

from baseline.abstract.base_extractor import BaseExtractor
from baseline.abstract.factory import register_extractor
from .shallowconvnet_config import ShallowConvNetModelConfig


class _Square(nn.Module):
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return x * x


class _SafeLog(nn.Module):
    """Numerically stable log used by Braindecode's ShallowFBCSPNet."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return torch.log(torch.clamp(x, min=self.eps))


@register_extractor("shallowconvnet")
class ShallowConvNetExtractor(BaseExtractor):
    """ShallowConvNet (Schirrmeister 2017) wrapped as a BaseExtractor.

    Supervised-from-scratch baseline — no pretrained weights. Maps
    (B, C, T) -> (B, embed_dim) where embed_dim = F_s * T_out, with
    T_out determined dynamically by the temporal pool stride.
    """

    CONFIG_CLASS = ShallowConvNetModelConfig

    def __init__(self, config: ShallowConvNetModelConfig):
        super().__init__()
        self.config = config
        C = config.n_channels
        F_t = config.n_filters_time
        F_s = config.n_filters_spat
        k_time = config.filter_time_length
        pool_len = config.pool_time_length
        pool_stride = config.pool_time_stride
        dropout = config.dropout

        T = int(config.sample_rate * config.window_sec)

        # Block 1: temporal conv -> spatial conv -> BN -> square -> avgpool -> log
        self.block = nn.Sequential(
            # Temporal conv: (B, 1, C, T) -> (B, F_t, C, T - k_time + 1)
            nn.Conv2d(1, F_t, kernel_size=(1, k_time), bias=True),
            # Spatial conv: collapse the channel dim with a full-height kernel.
            nn.Conv2d(F_t, F_s, kernel_size=(C, 1), bias=False),
            nn.BatchNorm2d(F_s),
            _Square(),
            nn.AvgPool2d(kernel_size=(1, pool_len), stride=(1, pool_stride)),
            _SafeLog(),
            nn.Dropout(dropout),
        )

        # Determine the flattened embed dim by running a dummy forward once.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            h = self.block(dummy)
            flat_dim = int(h.reshape(1, -1).shape[1])
        self.embed_dim = flat_dim
        config.embed_dim = flat_dim

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[ShallowConvNet] F_t={F_t} F_s={F_s} k_time={k_time} "
              f"pool={pool_len}/{pool_stride} T={T} -> embed_dim={flat_dim} "
              f"({n_params:,} params)")

    def forward(self, x: Tensor) -> Tensor:
        """Extract features.

        Args:
            x: (B, C, T) raw EEG window.

        Returns:
            (B, embed_dim) flattened feature vector.
        """
        # Braindecode convention: (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self.block(x)
        return x.reshape(x.size(0), -1)
