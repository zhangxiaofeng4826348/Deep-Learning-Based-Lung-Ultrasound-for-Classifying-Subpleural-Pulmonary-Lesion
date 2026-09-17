"""Loss functions for SLLA-UNet."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross entropy loss used in SimCLR."""

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.shape != z2.shape:
            raise ValueError(f"z1 and z2 must have the same shape, got {z1.shape} and {z2.shape}")
        batch_size = z1.size(0)
        if batch_size < 2:
            raise ValueError("NT-Xent loss requires batch_size >= 2")

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)
        similarity = torch.matmul(z, z.T) / self.temperature

        self_mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        similarity = similarity.masked_fill(self_mask, -1e9)

        positive_logits = torch.cat([
            torch.diag(similarity, batch_size),
            torch.diag(similarity, -batch_size),
        ], dim=0)
        denominator = torch.exp(similarity).sum(dim=1)
        loss = -torch.log(torch.exp(positive_logits) / denominator.clamp_min(1e-12))
        return loss.mean()


def effective_number_weights(class_counts: Sequence[int], beta: float = 0.999) -> torch.Tensor:
    """Class-balanced weights based on the effective number of samples."""
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    if torch.any(counts <= 0):
        raise ValueError("all class counts must be positive")
    effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), counts)
    weights = (1.0 - beta) / effective_num.clamp_min(1e-12)
    return weights / weights.sum() * len(class_counts)


class ClassBalancedFocalLoss(nn.Module):
    """Class-balanced focal loss for binary or multiclass classification."""

    def __init__(self, class_counts: Sequence[int], gamma: float = 1.0, beta: float = 0.999, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("class_weight", effective_number_weights(class_counts, beta=beta))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        log_pt = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        weights = self.class_weight.to(logits.device)[targets]
        loss = -weights * (1.0 - pt).clamp_min(1e-6).pow(self.gamma) * log_pt
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        numerator = 2.0 * (probs * targets).sum(dim=(1, 2, 3)) + self.eps
        denominator = probs.pow(2).sum(dim=(1, 2, 3)) + targets.pow(2).sum(dim=(1, 2, 3)) + self.eps
        return (1.0 - numerator / denominator).mean()


class CombinedLoss(nn.Module):
    """Joint loss: L_total = lambda_cls * L_cls + lambda_seg * L_seg."""

    def __init__(
        self,
        class_counts: Sequence[int] = (1, 1),
        lambda_cls: float = 1.0,
        lambda_seg: float = 1.0,
        focal_gamma: float = 1.0,
        cb_beta: float = 0.999,
    ) -> None:
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_seg = lambda_seg
        self.cls_loss = ClassBalancedFocalLoss(class_counts=class_counts, gamma=focal_gamma, beta=cb_beta)
        self.seg_loss = DiceLoss()

    def forward(
        self,
        cls_logits: torch.Tensor,
        cls_targets: torch.Tensor,
        seg_logits: torch.Tensor,
        seg_targets: torch.Tensor,
    ) -> torch.Tensor:
        cls = self.cls_loss(cls_logits, cls_targets)
        seg = self.seg_loss(seg_logits, seg_targets)
        return self.lambda_cls * cls + self.lambda_seg * seg
