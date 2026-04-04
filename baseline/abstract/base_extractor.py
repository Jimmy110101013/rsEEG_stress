from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseExtractor(nn.Module, ABC):
    """Unified interface for all EEG Foundation Model wrappers.

    Every FM extractor must map (B, C, T) -> (B, embed_dim).
    """

    embed_dim: int

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Extract features from raw EEG epochs.

        Args:
            x: (B, C, T) — batch of single-epoch EEG windows.

        Returns:
            (B, embed_dim) feature vectors.
        """
        ...

    def get_layer_groups(self) -> list[list[nn.Parameter]]:
        """Return parameter groups ordered from early to late layers.

        Used for LLRD (layer-wise learning rate decay).
        Default: single group with all parameters.
        Override in subclasses for model-specific layer ordering.
        """
        return [list(self.parameters())]
