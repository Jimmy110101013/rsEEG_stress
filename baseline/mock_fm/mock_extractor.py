import torch.nn as nn
from torch import Tensor

from baseline.abstract.base_extractor import BaseExtractor
from baseline.abstract.factory import register_extractor
from .mock_config import MockModelConfig


@register_extractor("mock_fm")
class MockExtractor(BaseExtractor):
    """Dummy FM that projects flattened EEG to embed_dim via a linear layer.

    Ensures gradients flow for MVP plumbing verification.
    """

    CONFIG_CLASS = MockModelConfig

    def __init__(self, config: MockModelConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        input_dim = config.n_channels * config.n_samples  # 30 * 2000 = 60000
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, self.embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, T) -> (B, embed_dim)"""
        return self.proj(x)
