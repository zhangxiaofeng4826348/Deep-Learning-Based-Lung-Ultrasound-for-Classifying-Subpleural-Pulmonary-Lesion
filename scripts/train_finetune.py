#!/usr/bin/env python
"""Supervised fine-tuning for SLLA-UNet."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from slla_unet.datasets import FineTuneDataset
from slla_unet.losses import CombinedLoss
from slla_unet.metrics import classification_metrics, dice_iou
from slla_unet.models import build_model
from slla_unet.optimizers import build_finetune_optimizer, build_scheduler
from slla_unet.utils import device_from_arg, load_model_weights, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="SLLA-UNet supervised fine-tuning")
    parser.add_argument("--train-csv", required=True, help="CSV with columns image,label,mask")
    parser.add_argument("--val-csv", required=True, help="CSV with columns image,label,mask")
    parser.add_argument("--root-dir", default=".", help="Root folder for relative image/mask paths")
    parser.add_argument("--out-dir", default="outputs/finetune")
    parser.add_argument("--variant", default="full", choices=["full", "non-ssl", "non-ms", "non-swin", "non-ms-swin"])
    parser.add_argument("--ssl-checkpoint", default=None, help="Optional SSL checkpoint")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-encoder", type=float, default=5e-6)
    parser.add_argument("--lr-decoder", type=float, default=1e-5)
    parser.add_argument("--lr-classifier", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--lambda-seg", type=float, default=1.0)
    parser.add_argument("--class-counts", type=int, nargs=2, default=[210, 484], metavar=("N_BENIGN", "N_MALIGNANT"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    losses = []
    all_labels = []
    all_probs = []
    all_seg_probs = []
    all_masks = []

    with torch.set_grad_enabled(train):
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)
            seg_logits, cls_logits, _, _, _ = model(images)
            loss = criterion(cls_logits, labels, seg_logits, masks)
            if train:
                loss.backward()
                optimizer.step()

            losses.append(loss.item() * images.size(0))
            probs = torch.softmax(cls_logits, dim=1)[:, 1]
            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
            all_seg_probs.append(torch.sigmoid(seg_logits).detach().cpu().numpy())
            all_masks.append(masks.detach().cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    seg_prob = np.concatenate(all_seg_probs)
    seg_mask = np.concatenate(all_masks)
    cls = classification_metrics(y_true, y_prob, threshold=0.5)
    seg = dice_iou(seg_prob, seg_mask, threshold=0.5)
    return sum(losses) / len(loader.dataset), cls, seg


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = device_from_arg(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = FineTuneDataset(args.train_csv, root_dir=args.root_dir, input_size=args.input_size)
    val_dataset = FineTuneDataset(args.val_csv, root_dir=args.root_dir, input_size=args.input_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.variant, input_size=args.input_size).to(device)
    if args.ssl_checkpoint and args.variant != "non-ssl":
        load_model_weights(model, args.ssl_checkpoint, device=device, strict=False)
        print(f"Loaded SSL checkpoint: {args.ssl_checkpoint}")

    criterion = CombinedLoss(class_counts=args.class_counts, lambda_cls=args.lambda_cls, lambda_seg=args.lambda_seg)
    optimizer = build_finetune_optimizer(
        model,
        lr_encoder=args.lr_encoder,
        lr_decoder=args.lr_decoder,
        lr_classifier=args.lr_classifier,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, epochs=args.epochs, min_lr=args.min_lr)

    best_val_loss = float("inf")
    best_val_auc = -float("inf")
    for epoch in range(args.epochs):
        train_loss, train_cls, train_seg = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss, val_cls, val_seg = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, AUC {train_cls.auc:.4f}, mDice {train_seg['mDice']:.4f} | "
            f"val loss {val_loss:.4f}, AUC {val_cls.auc:.4f}, ACC {val_cls.acc:.4f}, mDice {val_seg['mDice']:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({"model": model.state_dict(), "epoch": epoch + 1, "val_loss": best_val_loss}, out_dir / "best_loss.pth")
        if not np.isnan(val_cls.auc) and val_cls.auc > best_val_auc:
            best_val_auc = val_cls.auc
            save_checkpoint({"model": model.state_dict(), "epoch": epoch + 1, "val_auc": best_val_auc}, out_dir / "best_auc.pth")

    save_checkpoint({"model": model.state_dict(), "epoch": args.epochs}, out_dir / "final.pth")


if __name__ == "__main__":
    main()
