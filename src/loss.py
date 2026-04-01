import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced classification.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()

        focal_weight = (1.0 - probs).pow(self.gamma)
        loss = -(focal_weight * log_probs * targets_one_hot).sum(dim=1).mean()
        return loss


class MTLLoss(nn.Module):
    """Multi-Task Loss combining classification and regression branches.

    Branch A: CrossEntropyLoss (baseline stress classification)
    Branch B: MSELoss (stress severity regression, placeholder for CORAL)
    Total = alpha * loss_A + beta * loss_B
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

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
        loss_a = self.ce_loss(cls_logits, baseline_labels)
        loss_b = self.mse_loss(reg_output.squeeze(-1), stress_scores)
        total = self.alpha * loss_a + self.beta * loss_b
        return total, loss_a, loss_b
