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

    def __init__(self, extractor: BaseExtractor, embed_dim: int = 512,
                 dropout: float = 0.0, head_hidden: int = 0):
        super().__init__()
        self.extractor = extractor
        self.embed_dim = embed_dim

        # Branch A: Baseline Stress Classification (normal vs increase)
        # Branch B: State Stress Ordinal Regression
        if head_hidden > 0:
            self.head_cls = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(embed_dim, head_hidden),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(head_hidden, 2),
            )
            self.head_reg = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(embed_dim, head_hidden),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1),
            )
        elif dropout > 0.0:
            self.head_cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, 2))
            self.head_reg = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, 1))
        else:
            self.head_cls = nn.Linear(embed_dim, 2)
            self.head_reg = nn.Linear(embed_dim, 1)

    def extract_pooled(self, x: Tensor, mask: Tensor) -> Tensor:
        """Extract features and pool over epochs.

        Args:
            x: (B, M, C, T) — padded epoch sequences
            mask: (B, M) bool — True for valid epochs

        Returns:
            pooled: (B, embed_dim)
        """
        B, M, C, T = x.shape
        x_flat = x.reshape(B * M, C, T)
        feats = self.extractor(x_flat)
        feats = feats.reshape(B, M, self.embed_dim)

        mask_expanded = mask.unsqueeze(-1).float()
        feats_masked = feats * mask_expanded
        n_valid = mask_expanded.sum(dim=1).clamp(min=1)
        return feats_masked.sum(dim=1) / n_valid

    def classify(self, pooled: Tensor) -> tuple[Tensor, Tensor]:
        """Run classification and regression heads on pooled features."""
        return self.head_cls(pooled), self.head_reg(pooled)

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
        pooled = self.extract_pooled(x, mask)
        return self.classify(pooled)

    def freeze_backbone(self, unfreeze_cls_query: bool = False):
        for param in self.extractor.parameters():
            param.requires_grad = False
        if unfreeze_cls_query and hasattr(self.extractor, 'reve'):
            self.extractor.reve.cls_query_token.requires_grad = True

    def unfreeze_backbone(self):
        for param in self.extractor.parameters():
            param.requires_grad = True
