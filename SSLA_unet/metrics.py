"""Evaluation metrics for SLLA-UNet."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


@dataclass
class ClassificationMetrics:
    auc: float
    acc: float
    sen: float
    spc: float
    pre: float
    f1: float


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> ClassificationMetrics:
    """Compute binary classification metrics using probability >= threshold as malignant."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    acc = float(accuracy_score(y_true, y_pred))
    sen = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    spc = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    pre = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    return ClassificationMetrics(auc=auc, acc=acc, sen=sen, spc=spc, pre=pre, f1=f1)


def dice_iou(seg_prob, seg_mask, threshold: float = 0.5, eps: float = 1e-6) -> Dict[str, float]:
    """Compute mean Dice and mean IoU for binary segmentation."""
    pred = (np.asarray(seg_prob) >= threshold).astype(np.float32)
    target = (np.asarray(seg_mask) >= 0.5).astype(np.float32)

    if pred.ndim == 3:
        pred = pred[:, None, :, :]
    if target.ndim == 3:
        target = target[:, None, :, :]

    intersection = (pred * target).sum(axis=(1, 2, 3))
    pred_sum = pred.sum(axis=(1, 2, 3))
    target_sum = target.sum(axis=(1, 2, 3))
    union = pred_sum + target_sum - intersection

    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return {"mDice": float(np.mean(dice)), "mIoU": float(np.mean(iou))}


def bootstrap_auc_ci(y_true, y_prob, n_bootstraps: int = 1000, seed: int = 2026, alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for AUC."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not scores:
        return float("nan"), float("nan")
    return tuple(float(x) for x in np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
