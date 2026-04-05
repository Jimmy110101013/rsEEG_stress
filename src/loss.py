import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def adv_lambda_schedule(epoch: int, max_epochs: int, max_lambda: float = 0.1) -> float:
    """DANN-style progressive lambda ramp (Ganin et al., JMLR 2016).
    Sigmoid schedule: 0 → max_lambda over training."""
    p = epoch / max_epochs
    return float(max_lambda * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0))


class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default 2.0).
        alpha: Per-class weights tensor of shape (num_classes,), or None for uniform.
    """

    def __init__(self, gamma: float = 2.0, alpha: Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("alpha", alpha)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        n_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=n_classes).float()

        if self.label_smoothing > 0:
            targets_one_hot = (1.0 - self.label_smoothing) * targets_one_hot + \
                              self.label_smoothing / n_classes

        focal_weight = (1.0 - probs).pow(self.gamma)
        if self.alpha is not None:
            focal_weight = focal_weight * self.alpha.unsqueeze(0)
        loss = -(focal_weight * log_probs * targets_one_hot).sum(dim=1).mean()
        return loss


class PairwiseRankingLoss(nn.Module):
    """Within-subject pairwise ranking loss with score-gap-weighted margins.

    For each within-subject pair (i, j), enforces:
        pred_i - pred_j ≥ |score_i - score_j|   (if score_i > score_j)

    The margin equals the actual score difference, so the model must reproduce
    both the ranking AND the magnitude of stress differences. Noise pairs
    (tiny score gaps like 63 vs 65) get near-zero margins and don't dominate.

    When patient_ids is None, falls back to all-pairs (cross-subject).
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.fixed_margin = margin  # additional fixed margin on top of score gap

    def forward(self, reg_output: Tensor, stress_scores: Tensor,
                patient_ids: Tensor | None = None) -> Tensor:
        pred = reg_output.squeeze(-1)  # (B,)
        n = pred.size(0)
        if n < 2:
            return pred.new_tensor(0.0)

        i, j = torch.triu_indices(n, n, offset=1, device=pred.device)

        # Within-subject filtering: only keep pairs from the same patient
        if patient_ids is not None:
            same_subj = patient_ids[i] == patient_ids[j]
            i, j = i[same_subj], j[same_subj]
            if len(i) == 0:
                return pred.new_tensor(0.0)

        score_diff = stress_scores[i] - stress_scores[j]
        target = torch.sign(score_diff)

        # Filter out ties (same score)
        valid = target != 0
        if not valid.any():
            return pred.new_tensor(0.0)

        i, j, target = i[valid], j[valid], target[valid]
        # Per-pair margin = actual score gap (scores already in [0,1])
        margins = torch.abs(stress_scores[i] - stress_scores[j]) + self.fixed_margin

        # loss = max(0, margin_ij - target_ij * (pred_i - pred_j))
        pred_diff = pred[i] - pred[j]
        loss = torch.clamp(margins - target * pred_diff, min=0)
        return loss.mean()


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
