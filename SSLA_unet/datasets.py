"""Dataset utilities for SLLA-UNet.

Expected supervised CSV columns:
    image,label,mask

- image: relative or absolute path to an RGB/B-mode image.
- label: 0 for benign, 1 for malignant.
- mask: optional relative or absolute path to a binary lesion mask. If mask is missing,
  a zero mask is returned so that inference/classification utilities can still run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"image folder does not exist: {folder}")
    files = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise FileNotFoundError(f"no image files found in: {folder}")
    return files


def default_image_transform(input_size: int = 224) -> T.Compose:
    return T.Compose([T.Resize((input_size, input_size)), T.ToTensor()])


def default_mask_transform(input_size: int = 224) -> T.Compose:
    return T.Compose([T.Resize((input_size, input_size), interpolation=T.InterpolationMode.NEAREST), T.ToTensor()])


class SSLDataset(Dataset):
    """Unlabeled image dataset returning two augmented views per image."""

    def __init__(self, image_folder: str | Path, transform1: Callable, transform2: Callable) -> None:
        self.image_paths = list_images(image_folder)
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        return {
            "image1": self.transform1(image),
            "image2": self.transform2(image),
            "path": str(path),
        }


class FineTuneDataset(Dataset):
    """Supervised dataset for joint segmentation and classification."""

    def __init__(
        self,
        csv_path: str | Path,
        root_dir: str | Path = ".",
        input_size: int = 224,
        image_col: str = "image",
        label_col: str = "label",
        mask_col: str = "mask",
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file does not exist: {self.csv_path}")
        self.root_dir = Path(root_dir)
        self.image_col = image_col
        self.label_col = label_col
        self.mask_col = mask_col
        self.image_transform = default_image_transform(input_size)
        self.mask_transform = default_mask_transform(input_size)

        table = pd.read_csv(self.csv_path)
        required = {image_col, label_col}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        self.table = table.reset_index(drop=True)
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.table)

    def _resolve(self, value: str) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else self.root_dir / path

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        row = self.table.iloc[index]
        image_path = self._resolve(row[self.image_col])
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)

        label = torch.tensor(int(row[self.label_col]), dtype=torch.long)

        mask_tensor: torch.Tensor
        if self.mask_col in self.table.columns and pd.notna(row.get(self.mask_col, None)) and str(row[self.mask_col]).strip():
            mask_path = self._resolve(row[self.mask_col])
            if not mask_path.exists():
                raise FileNotFoundError(f"mask not found: {mask_path}")
            mask = Image.open(mask_path).convert("L")
            mask_tensor = (self.mask_transform(mask) > 0.5).float()
        else:
            mask_tensor = torch.zeros((1, self.input_size, self.input_size), dtype=torch.float32)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": label,
            "image_path": str(image_path),
        }
