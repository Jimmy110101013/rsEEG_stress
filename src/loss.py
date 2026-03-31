import torch
import torch.nn as nn
from torch import Tensor


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
