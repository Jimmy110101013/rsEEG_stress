import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function

from baseline.abstract.base_extractor import BaseExtractor


class GradientReversalFn(Function):
    """Gradient Reversal Layer (Ganin et al., JMLR 2016).
    Forward: identity. Backward: negate gradients scaled by lambda.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversal(nn.Module):
    def forward(self, x, lambda_=1.0):
        return GradientReversalFn.apply(x, lambda_)


class DecoupledStressModel(nn.Module):
    """Decoupled Ordinal Multi-Task Learning model for EEG stress classification.

    Architecture:
        (B, M, C, T) → extractor → (B, M, embed_dim) → masked avg pool
            → (B, embed_dim)
            ├── Branch A: Linear → (B, 2)  baseline classification
            └── Branch B: Linear → (B, 1)  stress severity regression
    """

    def __init__(self, extractor: BaseExtractor, embed_dim: int = 512,
                 dropout: float = 0.0, head_hidden: int = 0,
                 n_subjects: int = 0):
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

        # Subject-adversarial head (opt-in: n_subjects > 0)
        if n_subjects > 0:
            self.grl = GradientReversal()
            self.head_subj = nn.Sequential(
                nn.Linear(embed_dim, 128), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(128, n_subjects),
            )

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

    def classify_subject(self, features: Tensor, lambda_adv: float = 1.0) -> Tensor | None:
        """Subject classification through GRL for adversarial training.
        Returns None if no adversarial head is configured."""
        if not hasattr(self, 'head_subj'):
            return None
        return self.head_subj(self.grl(features, lambda_adv))

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

    def predict_windows(self, x: Tensor, mask: Tensor):
        """Predict at both window-level and subject-level (single forward pass).

        Returns:
            window_logits: (B, M, 2) — per-window classification
            cls_logits: (B, 2) — subject-level classification (pooled)
            reg_output: (B, 1) — subject-level regression (pooled)
        """
        B, M, C, T = x.shape
        x_flat = x.reshape(B * M, C, T)
        feats = self.extractor(x_flat)
        feats = feats.reshape(B, M, self.embed_dim)

        # Window-level: classify each window independently
        window_logits = self.head_cls(feats)  # (B, M, 2)

        # Subject-level: masked average pool then classify
        mask_exp = mask.unsqueeze(-1).float()
        pooled = (feats * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1)
        cls_logits, reg_output = self.classify(pooled)

        return window_logits, cls_logits, reg_output

    def freeze_backbone(self, unfreeze_cls_query: bool = False):
        for param in self.extractor.parameters():
            param.requires_grad = False
        if unfreeze_cls_query and hasattr(self.extractor, 'reve'):
            self.extractor.reve.cls_query_token.requires_grad = True

    def unfreeze_backbone(self):
        for param in self.extractor.parameters():
            param.requires_grad = True
