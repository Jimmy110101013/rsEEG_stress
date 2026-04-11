"""EEG-Conformer extractor — Song et al. 2023 CNN + Transformer hybrid.

Architecture (input (B, C, T), typically (B, 30, 1000) for 5 s @ 200 Hz):

    PatchEmbedding (shallow conv front-end):
        Conv2D(1 -> F, kernel=(1, filter_time_length))       # temporal filters
        Conv2D(F -> F, kernel=(C, 1), bias=False)            # spatial filters
        BatchNorm2d(F)
        ELU
        AvgPool2d((1, pool_time_length), stride=(1, pool_time_stride))
        Dropout(embed_dropout)
        Conv2D(F -> F, kernel=(1, 1))                        # token projection
        -> rearrange to (B, T_tokens, F)

    TransformerEncoder (L layers, H heads, d_model=F, d_ff=4*F):
        Pre-norm MultiHeadSelfAttention + FeedForward residual blocks
        (nn.TransformerEncoder with batch_first=True, norm_first=True).

    Flatten -> (B, T_tokens * F)

The original paper's classifier head takes the flattened transformer
output through an MLP to produce logits. Here we stop at the flatten
step so the unified `DecoupledStressModel` head can consume the
features, matching how the FM extractors are wired.

Reference:
    Song et al. 2023, IEEE TNSRE 31:710-719.
    Official repo: https://github.com/eeyhsong/EEG-Conformer
"""

import torch
import torch.nn as nn
from torch import Tensor

from baseline.abstract.base_extractor import BaseExtractor
from baseline.abstract.factory import register_extractor
from .eegconformer_config import EEGConformerModelConfig


class _PatchEmbedding(nn.Module):
    """Shallow conv front-end that produces transformer tokens."""

    def __init__(self, n_channels: int, n_filters: int,
                 filter_time_length: int, pool_time_length: int,
                 pool_time_stride: int, dropout: float):
        super().__init__()
        self.temporal = nn.Conv2d(1, n_filters, kernel_size=(1, filter_time_length),
                                  bias=True)
        self.spatial = nn.Conv2d(n_filters, n_filters, kernel_size=(n_channels, 1),
                                 bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.act = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_time_length),
                                 stride=(1, pool_time_stride))
        self.drop = nn.Dropout(dropout)
        # 1x1 projection; kept separate from the temporal/spatial convs so
        # it can be reinitialised or frozen independently if needed.
        self.proj = nn.Conv2d(n_filters, n_filters, kernel_size=(1, 1), bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """(B, 1, C, T) -> (B, T_tokens, F)."""
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.drop(x)
        x = self.proj(x)
        # x: (B, F, 1, T_tokens) -> (B, T_tokens, F)
        x = x.squeeze(2).transpose(1, 2)
        return x


@register_extractor("eegconformer")
class EEGConformerExtractor(BaseExtractor):
    """EEG-Conformer (Song 2023) wrapped as a BaseExtractor.

    Supervised-from-scratch baseline — no pretrained weights. Maps
    (B, C, T) -> (B, embed_dim) where embed_dim = n_filters * T_tokens
    and T_tokens is determined dynamically by the conv front-end.
    """

    CONFIG_CLASS = EEGConformerModelConfig

    def __init__(self, config: EEGConformerModelConfig):
        super().__init__()
        self.config = config
        C = config.n_channels
        F = config.n_filters
        T = int(config.sample_rate * config.window_sec)

        if F % config.n_heads != 0:
            raise ValueError(
                f"n_filters ({F}) must be divisible by n_heads "
                f"({config.n_heads}) for the transformer encoder."
            )

        self.patch_embed = _PatchEmbedding(
            n_channels=C,
            n_filters=F,
            filter_time_length=config.filter_time_length,
            pool_time_length=config.pool_time_length,
            pool_time_stride=config.pool_time_stride,
            dropout=config.embed_dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=F,
            nhead=config.n_heads,
            dim_feedforward=F * config.forward_expansion,
            dropout=config.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_transformer_layers
        )

        # Determine the flattened embed dim by running a dummy forward once.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            tokens = self.patch_embed(dummy)
            h = self.transformer(tokens)
            flat_dim = int(h.reshape(1, -1).shape[1])
            self._n_tokens = int(tokens.shape[1])
        self.embed_dim = flat_dim
        config.embed_dim = flat_dim

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[EEG-Conformer] F={F} L={config.n_transformer_layers} "
              f"H={config.n_heads} T={T} tokens={self._n_tokens} "
              f"-> embed_dim={flat_dim} ({n_params:,} params)")

    def forward(self, x: Tensor) -> Tensor:
        """Extract features.

        Args:
            x: (B, C, T) raw EEG window.

        Returns:
            (B, embed_dim) flattened feature vector.
        """
        # (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(1)
        tokens = self.patch_embed(x)           # (B, T_tokens, F)
        enc = self.transformer(tokens)         # (B, T_tokens, F)
        return enc.reshape(enc.size(0), -1)    # (B, T_tokens * F)

    def get_layer_groups(self) -> list[list[nn.Parameter]]:
        """Two groups: conv front-end and transformer encoder."""
        return [
            list(self.patch_embed.parameters()),
            list(self.transformer.parameters()),
        ]
