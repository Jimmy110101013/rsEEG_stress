"""EEGNet extractor — Lawhern et al. 2018 compact CNN for EEG classification.

Architecture (input (B, C, T), typically (B, 30, 1000) for 5 s @ 200 Hz):

    Block 1 — temporal + spatial filtering
        Conv2D(1 → F1, kernel=(1, kernel_length), padding='same')
        BatchNorm2d
        DepthwiseConv2D(F1 → F1*D, kernel=(C, 1), groups=F1, bias=False)
            → spatial filtering: one (F1) temporal filter sees all C channels
              and is projected into D=2 spatial outputs
        BatchNorm2d
        ELU
        AvgPool2d((1, pool1))        # temporal downsampling by pool1=4
        Dropout(dropout)

    Block 2 — separable convolution
        SeparableConv2D(F1*D → F2, kernel=(1, 16))
            = DepthwiseConv2D(F1*D, (1, 16)) + PointwiseConv2D(F1*D → F2, (1, 1))
        BatchNorm2d
        ELU
        AvgPool2d((1, pool2))        # temporal downsampling by pool2=8
        Dropout(dropout)

    Flatten → (B, F2 * T_final) where T_final = T // (pool1 * pool2)

Reference:
    Lawhern et al. 2018, J. Neural Eng. 15(5):056013.
    Original tensorflow implementation at
    https://github.com/vlawhern/arl-eegmodels
"""

import torch
import torch.nn as nn
from torch import Tensor

from baseline.abstract.base_extractor import BaseExtractor
from baseline.abstract.factory import register_extractor
from .eegnet_config import EEGNetModelConfig


class _DepthwiseConv2d(nn.Conv2d):
    """Keras-style DepthwiseConv2D: groups = in_channels, out = in_channels * D."""

    def __init__(self, in_channels: int, depth_multiplier: int, kernel_size,
                 bias: bool = False, max_norm: float | None = None):
        super().__init__(
            in_channels=in_channels,
            out_channels=in_channels * depth_multiplier,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=bias,
        )
        self._max_norm = max_norm

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if self._max_norm is not None:
            # Max-norm constraint on the depthwise spatial filter (Lawhern eq. 1).
            with torch.no_grad():
                norms = self.weight.view(self.weight.size(0), -1).norm(
                    dim=1, keepdim=True
                )
                desired = torch.clamp(norms, max=self._max_norm)
                scale = desired / (norms + 1e-8)
                self.weight.mul_(scale.view(-1, 1, 1, 1))
        return super().forward(x)


class _SeparableConv2d(nn.Module):
    """Keras-style SeparableConv2D = DepthwiseConv2D + PointwiseConv2D."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, bias: bool = False):
        super().__init__()
        # Pad to 'same' along the time dimension only (channel dim is already 1 here).
        pad_t = kernel_size[1] // 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            groups=in_channels, bias=False, padding=(0, pad_t),
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


@register_extractor("eegnet")
class EEGNetExtractor(BaseExtractor):
    """EEGNet (Lawhern 2018) wrapped as a BaseExtractor.

    Supervised-from-scratch baseline — no pretrained weights. Maps
    (B, C, T) → (B, embed_dim) where embed_dim = F2 * (T // (pool1 * pool2)).
    The final linear classifier is *not* part of the extractor; the
    classifier head lives in `DecoupledStressModel`, matching how the
    FM extractors are used.
    """

    CONFIG_CLASS = EEGNetModelConfig

    def __init__(self, config: EEGNetModelConfig):
        super().__init__()
        self.config = config
        C = config.n_channels
        F1 = config.F1
        D = config.D
        F2 = config.F2
        kernel_length = config.kernel_length
        pool1 = config.pool1
        pool2 = config.pool2
        dropout = config.dropout

        T = int(config.sample_rate * config.window_sec)
        # PyTorch AvgPool2d handles non-divisible lengths via floor; we don't
        # require exact divisibility. The effective output time dim is
        # computed below via a dummy forward pass.

        # Block 1: temporal conv + spatial depthwise
        # 'same' padding along time = (kernel_length // 2)
        pad_t = kernel_length // 2
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, pad_t), bias=False),
            nn.BatchNorm2d(F1),
            _DepthwiseConv2d(F1, depth_multiplier=D, kernel_size=(C, 1), bias=False, max_norm=1.0),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool1)),
            nn.Dropout(dropout),
        )

        # Block 2: separable conv
        self.block2 = nn.Sequential(
            _SeparableConv2d(F1 * D, F2, kernel_size=(1, 16), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool2)),
            nn.Dropout(dropout),
        )

        # Determine the flattened embed dim by running a dummy forward once.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            h = self.block1(dummy)
            h = self.block2(h)
            flat_dim = int(h.reshape(1, -1).shape[1])
        self.embed_dim = flat_dim
        # Write it back into the config so downstream (train_ft.py) can read it.
        config.embed_dim = flat_dim

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[EEGNet] F1={F1} D={D} F2={F2} kernel={kernel_length} "
              f"T={T} → embed_dim={flat_dim}  ({n_params:,} params)")

    def forward(self, x: Tensor) -> Tensor:
        """Extract features.

        Args:
            x: (B, C, T) raw EEG window.

        Returns:
            (B, embed_dim) flattened feature vector.
        """
        # EEGNet Keras convention: (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.reshape(x.size(0), -1)
        return x

    def get_layer_groups(self) -> list[list[nn.Parameter]]:
        """EEGNet is shallow; two groups suffice for LLRD if ever needed."""
        return [list(self.block1.parameters()), list(self.block2.parameters())]
