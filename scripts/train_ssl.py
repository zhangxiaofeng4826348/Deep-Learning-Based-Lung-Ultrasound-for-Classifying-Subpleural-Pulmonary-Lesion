#!/usr/bin/env python
"""Self-supervised contrastive pretraining for SLLA-UNet."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from slla_unet.datasets import SSLDataset
from slla_unet.losses import NTXentLoss
from slla_unet.models import ProjectionHead, build_model, classifier_input_dim
from slla_unet.optimizers import build_scheduler, build_ssl_optimizer
from slla_unet.utils import AverageMeter, device_from_arg, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="SLLA-UNet SSL pretraining")
    parser.add_argument("--image-dir", required=True, help="Folder containing unlabeled ultrasound images")
    parser.add_argument("--out-dir", default="outputs/ssl", help="Output directory")
    parser.add_argument("--variant", default="full", choices=["full", "non-swin", "non-ms", "non-ms-swin"], help="Model variant")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = device_from_arg(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform1 = T.Compose([
        T.RandomResizedCrop(args.input_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.ToTensor(),
    ])
    transform2 = T.Compose([
        T.RandomResizedCrop(args.input_size, scale=(0.8, 1.0)),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
    ])

    dataset = SSLDataset(args.image_dir, transform1=transform1, transform2=transform2)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    model = build_model(args.variant, input_size=args.input_size).to(device)
    projection_head = ProjectionHead(classifier_input_dim(model), proj_dim=args.proj_dim).to(device)
    optimizer = build_ssl_optimizer(model, projection_head, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, epochs=args.epochs, min_lr=1e-6)
    criterion = NTXentLoss(temperature=args.temperature)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_loss = float("inf")
    losses = []

    for epoch in range(args.epochs):
        model.train()
        projection_head.train()
        meter = AverageMeter()
        for batch in loader:
            img1 = batch["image1"].to(device)
            img2 = batch["image2"].to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                _, _, feat1, _, _ = model(img1)
                _, _, feat2, _, _ = model(img2)
                z1 = projection_head(feat1)
                z2 = projection_head(feat2)
                loss = criterion(z1, z2)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(projection_head.parameters()), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            meter.update(loss.item(), img1.size(0))

        scheduler.step()
        losses.append(meter.avg)
        print(f"Epoch {epoch + 1:03d}/{args.epochs} | SSL loss: {meter.avg:.4f}")

        if meter.avg < best_loss:
            best_loss = meter.avg
            save_checkpoint({"model": model.state_dict(), "projection_head": projection_head.state_dict(), "epoch": epoch + 1, "loss": best_loss}, out_dir / "ssl_best.pth")

    save_checkpoint({"model": model.state_dict(), "projection_head": projection_head.state_dict(), "loss_curve": losses}, out_dir / "ssl_final.pth")

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("NT-Xent loss")
    plt.tight_layout()
    plt.savefig(out_dir / "ssl_loss_curve.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
