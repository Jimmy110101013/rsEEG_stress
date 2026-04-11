"""DeepConvNet extractor — Schirrmeister et al. 2017 4-block CNN.

Architecture (input (B, C, T), typically (B, 30, 1000) for 5 s @ 200 Hz):

    Block 1 — temporal + spatial conv
        Conv2D(1 -> F1, kernel=(1, filter_time_length))
        Conv2D(F1 -> F1, kernel=(C, 1), bias=False)    # collapse channel dim
        BatchNorm2d(F1)
        ELU
        MaxPool2d((1, pool_time_length), stride=(1, pool_time_stride))
        Dropout

    Block 2..4 — stacked temporal convs with widening channels
        Conv2D(F_{k-1} -> F_k, kernel=(1, filter_time_length))
        BatchNorm2d(F_k)
        ELU
        MaxPool2d((1, pool_time_length), stride=(1, pool_time_stride))
        Dropout

    Flatten -> (B, F4 * T_out)

Reference:
    Schirrmeister et al. 2017, Human Brain Mapping 38(11):5391-5420.
    Braindecode reference: `Deep4Net` at https://braindecode.org
"""

import torch
import torch.nn as nn
from torch import Tensor

from baseline.abstract.base_extractor import BaseExtractor
from baseline.abstract.factory import register_extractor
from .deepconvnet_config import DeepConvNetModelConfig


def _temporal_block(in_ch: int, out_ch: int, k_time: int, pool_len: int,
                    pool_stride: int, dropout: float) -> nn.Sequential:
    """Standard Deep4Net block: Conv -> BN -> ELU -> MaxPool -> Dropout."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=(1, k_time), bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=(1, pool_len), stride=(1, pool_stride)),
        nn.Dropout(dropout),
    )


@register_extractor("deepconvnet")
class DeepConvNetExtractor(BaseExtractor):
    """DeepConvNet (Schirrmeister 2017) wrapped as a BaseExtractor.

    Supervised-from-scratch baseline — no pretrained weights. Maps
    (B, C, T) -> (B, embed_dim) where embed_dim = F4 * T_out, with
    T_out determined dynamically by the 4-stage max-pool stack.
    """

    CONFIG_CLASS = DeepConvNetModelConfig

    def __init__(self, config: DeepConvNetModelConfig):
        super().__init__()
        self.config = config
        C = config.n_channels
        F1, F2, F3, F4 = (
            config.n_filters_1, config.n_filters_2,
            config.n_filters_3, config.n_filters_4,
        )
        k_time = config.filter_time_length
        pool_len = config.pool_time_length
        pool_stride = config.pool_time_stride
        dropout = config.dropout

        T = int(config.sample_rate * config.window_sec)

        # Block 1 is special: it fuses temporal + spatial conv before pooling.
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, k_time), bias=True),
            nn.Conv2d(F1, F1, kernel_size=(C, 1), bias=False),
            nn.BatchNorm2d(F1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, pool_len), stride=(1, pool_stride)),
            nn.Dropout(dropout),
        )
        self.block2 = _temporal_block(F1, F2, k_time, pool_len, pool_stride, dropout)
        self.block3 = _temporal_block(F2, F3, k_time, pool_len, pool_stride, dropout)
        self.block4 = _temporal_block(F3, F4, k_time, pool_len, pool_stride, dropout)

        # Determine the flattened embed dim by running a dummy forward once.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            h = self.block1(dummy)
            h = self.block2(h)
            h = self.block3(h)
            h = self.block4(h)
            flat_dim = int(h.reshape(1, -1).shape[1])
        self.embed_dim = flat_dim
        config.embed_dim = flat_dim

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[DeepConvNet] F=({F1},{F2},{F3},{F4}) k_time={k_time} "
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
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.reshape(x.size(0), -1)

    def get_layer_groups(self) -> list[list[nn.Parameter]]:
        """Four groups so LLRD can taper learning rate from early to late."""
        return [
            list(self.block1.parameters()),
            list(self.block2.parameters()),
            list(self.block3.parameters()),
            list(self.block4.parameters()),
        ]
