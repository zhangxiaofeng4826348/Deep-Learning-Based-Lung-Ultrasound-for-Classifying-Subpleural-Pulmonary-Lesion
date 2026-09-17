#!/usr/bin/env python
"""Create a synthetic toy dataset to verify that the training scripts run.

This does not represent clinical data and must not be used for scientific evaluation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(description="Create a synthetic SLLA-UNet toy dataset")
    parser.add_argument("--out-dir", default="examples/toy_data")
    parser.add_argument("--n-train", type=int, default=8)
    parser.add_argument("--n-val", type=int, default=4)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def create_case(path_img: Path, path_mask: Path, label: int, size: int, rng: np.random.Generator) -> None:
    image = np.zeros((size, size), dtype=np.uint8)
    image += rng.normal(35, 10, image.shape).clip(-20, 30).astype(np.uint8)
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    cx, cy = rng.integers(size // 3, 2 * size // 3, size=2)
    rx, ry = rng.integers(size // 8, size // 4, size=2)
    if label == 1:
        # Irregular polygon-like lesion
        points = []
        for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            jitter = rng.uniform(0.7, 1.3)
            x = cx + int(rx * jitter * np.cos(angle))
            y = cy + int(ry * jitter * np.sin(angle))
            points.append((x, y))
        draw.polygon(points, fill=255)
    else:
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    mask_arr = np.asarray(mask) > 0
    image[mask_arr] = rng.integers(110, 190)
    image = np.clip(image + rng.normal(0, 7, image.shape), 0, 255).astype(np.uint8)
    rgb = np.stack([image, image, image], axis=-1)
    Image.fromarray(rgb).save(path_img)
    mask.save(path_mask)


def write_split(split: str, n: int, out_dir: Path, size: int, rng: np.random.Generator) -> None:
    img_dir = out_dir / split / "images"
    mask_dir = out_dir / split / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        label = i % 2
        img_rel = f"{split}/images/{split}_{i:03d}.png"
        mask_rel = f"{split}/masks/{split}_{i:03d}_mask.png"
        create_case(out_dir / img_rel, out_dir / mask_rel, label, size, rng)
        rows.append({"image": img_rel, "mask": mask_rel, "label": label})
    with open(out_dir / f"{split}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "mask", "label"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)
    write_split("train", args.n_train, out_dir, args.size, rng)
    write_split("val", args.n_val, out_dir, args.size, rng)
    print(f"Synthetic toy dataset written to {out_dir}")


if __name__ == "__main__":
    main()
