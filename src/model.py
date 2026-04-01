import torch
import torch.nn as nn
from torch import Tensor

from baseline.abstract.base_extractor import BaseExtractor


class DecoupledStressModel(nn.Module):
    """Decoupled Ordinal Multi-Task Learning model for EEG stress classification.

    Architecture:
        (B, M, C, T) → extractor → (B, M, embed_dim) → masked avg pool
            → (B, embed_dim)
            ├── Branch A: Linear → (B, 2)  baseline classification
            └── Branch B: Linear → (B, 1)  stress severity regression
    """

    def __init__(self, extractor: BaseExtractor, embed_dim: int = 512):
        super().__init__()
        self.extractor = extractor
        self.embed_dim = embed_dim

        self.pool = nn.AdaptiveAvgPool1d(1)

        # Branch A: Baseline Stress Classification (normal vs increase)
        self.head_cls = nn.Linear(embed_dim, 2)
        # Branch B: State Stress Ordinal Regression
        self.head_reg = nn.Linear(embed_dim, 1)

    def forward(
        self, x: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, M, C, T) — padded epoch sequences
            mask: (B, M) bool — True for valid epochs

        Returns:
            cls_logits: (B, 2)
            reg_output: (B, 1)
        """
        B, M, C, T = x.shape

        # Flatten batch and epoch dims for extractor
        x_flat = x.reshape(B * M, C, T)              # (B*M, C, T)
        feats = self.extractor(x_flat)                # (B*M, embed_dim)
        feats = feats.reshape(B, M, self.embed_dim)   # (B, M, embed_dim)

        # Masked average pooling over epochs
        mask_expanded = mask.unsqueeze(-1).float()     # (B, M, 1)
        feats_masked = feats * mask_expanded           # zero out padding
        n_valid = mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
        pooled = feats_masked.sum(dim=1) / n_valid     # (B, embed_dim)

        cls_logits = self.head_cls(pooled)  # (B, 2)
        reg_output = self.head_reg(pooled)  # (B, 1)

        return cls_logits, reg_output

    def freeze_backbone(self):
        for param in self.extractor.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.extractor.parameters():
            param.requires_grad = True
