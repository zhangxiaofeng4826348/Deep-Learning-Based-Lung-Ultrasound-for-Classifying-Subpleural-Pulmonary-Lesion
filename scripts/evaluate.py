#!/usr/bin/env python
"""Evaluate SLLA-UNet on a labeled dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from slla_unet.datasets import FineTuneDataset
from slla_unet.metrics import bootstrap_auc_ci, classification_metrics, dice_iou
from slla_unet.models import build_model
from slla_unet.utils import device_from_arg, load_model_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SLLA-UNet")
    parser.add_argument("--csv", required=True, help="CSV with columns image,label,mask")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="outputs/evaluation")
    parser.add_argument("--variant", default="full", choices=["full", "non-ssl", "non-ms", "non-swin", "non-ms-swin"])
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = device_from_arg(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = FineTuneDataset(args.csv, root_dir=args.root_dir, input_size=args.input_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(args.variant, input_size=args.input_size).to(device)
    load_model_weights(model, args.checkpoint, device=device, strict=False)
    model.eval()

    labels, probs, seg_probs, masks, paths = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            seg_logits, cls_logits, _, _, _ = model(images)
            probs_batch = torch.softmax(cls_logits, dim=1)[:, 1]
            labels.append(batch["label"].cpu().numpy())
            probs.append(probs_batch.cpu().numpy())
            seg_probs.append(torch.sigmoid(seg_logits).cpu().numpy())
            masks.append(batch["mask"].cpu().numpy())
            paths.extend(batch["image_path"])

    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probs)
    seg_prob = np.concatenate(seg_probs)
    seg_mask = np.concatenate(masks)

    cls_metrics = classification_metrics(y_true, y_prob, threshold=args.threshold)
    seg_metrics = dice_iou(seg_prob, seg_mask, threshold=0.5)
    ci_low, ci_high = bootstrap_auc_ci(y_true, y_prob, n_bootstraps=args.bootstrap)
    metrics = asdict(cls_metrics)
    metrics.update(seg_metrics)
    metrics["auc_ci95"] = [ci_low, ci_high]
    metrics["threshold"] = args.threshold

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_table = pd.DataFrame({"image_path": paths, "label": y_true, "prob_malignant": y_prob, "pred_label": (y_prob >= args.threshold).astype(int)})
    pred_table.to_csv(out_dir / "predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
