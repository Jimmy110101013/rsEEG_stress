import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default 2.0).
        alpha: Per-class weights tensor of shape (num_classes,), or None for uniform.
    """

    def __init__(self, gamma: float = 2.0, alpha: Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()

        focal_weight = (1.0 - probs).pow(self.gamma)
        if self.alpha is not None:
            focal_weight = focal_weight * self.alpha.unsqueeze(0)
        loss = -(focal_weight * log_probs * targets_one_hot).sum(dim=1).mean()
        return loss


class PairwiseRankingLoss(nn.Module):
    """Pairwise ranking loss for ordinal regression.

    For all pairs (i, j) in a batch where score_i != score_j,
    enforces that the model output preserves the same ordering.
    Invariant to absolute score values — only cares about relative rank.
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(self, reg_output: Tensor, stress_scores: Tensor) -> Tensor:
        pred = reg_output.squeeze(-1)  # (B,)
        n = pred.size(0)
        if n < 2:
            return pred.new_tensor(0.0)

        i, j = torch.triu_indices(n, n, offset=1, device=pred.device)
        target = torch.sign(stress_scores[i] - stress_scores[j])

        # Filter out ties (same score)
        valid = target != 0
        if not valid.any():
            return pred.new_tensor(0.0)

        return F.margin_ranking_loss(
            pred[i[valid]], pred[j[valid]], target[valid], margin=self.margin,
        )


class MTLLoss(nn.Module):
    """Multi-Task Loss combining classification and regression branches.

    Branch A: Configurable classification loss (FocalLoss or CE)
    Branch B: Configurable regression loss (PairwiseRankingLoss or MSE)
    Total = alpha * loss_A + beta * loss_B
    """

    def __init__(self, cls_criterion: nn.Module, reg_criterion: nn.Module,
                 alpha: float = 1.0, beta: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.cls_criterion = cls_criterion
        self.reg_criterion = reg_criterion

    def forward(
        self,
        cls_logits: Tensor,
        reg_output: Tensor,
        baseline_labels: Tensor,
        stress_scores: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            cls_logits: (B, 2) from Branch A
            reg_output: (B, 1) from Branch B
            baseline_labels: (B,) long — 0 or 1
            stress_scores: (B,) float — normalized stress score

        Returns:
            total_loss, loss_a, loss_b
        """
        loss_a = self.cls_criterion(cls_logits, baseline_labels)
        loss_b = self.reg_criterion(reg_output, stress_scores)
        total = self.alpha * loss_a + self.beta * loss_b
        return total, loss_a, loss_b
