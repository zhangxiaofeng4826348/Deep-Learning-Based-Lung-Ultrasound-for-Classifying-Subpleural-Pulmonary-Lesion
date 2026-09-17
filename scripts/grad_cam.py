#!/usr/bin/env python
"""Generate Grad-CAM visualization for one image."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from slla_unet.models import build_model
from slla_unet.utils import device_from_arg, load_model_weights


class GradCAM:
    def __init__(self, model, target_layer: str = "RS") -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        if target_layer not in {"R1", "R2", "R3", "R4", "RS"}:
            raise ValueError("target_layer must be one of R1, R2, R3, R4, RS")
        module = getattr(model, target_layer)
        module.register_forward_hook(self._forward_hook)
        module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        _, cls_logits, _, _, _ = self.model(x)
        score = cls_logits[:, class_idx].sum()
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        return cam.detach().cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM for SLLA-UNet")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="outputs/grad_cam.png")
    parser.add_argument("--variant", default="full", choices=["full", "non-ssl", "non-ms", "non-swin", "non-ms-swin"])
    parser.add_argument("--target-layer", default="RS", choices=["R1", "R2", "R3", "R4", "RS"])
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = device_from_arg(args.device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(args.variant, input_size=args.input_size).to(device)
    load_model_weights(model, args.checkpoint, device=device, strict=False)
    model.eval()

    image = Image.open(args.image).convert("RGB").resize((args.input_size, args.input_size))
    image_arr = np.asarray(image)
    x = T.ToTensor()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        _, cls_logits, _, _, _ = model(x)
        prob = torch.softmax(cls_logits, dim=1)[0, 1].item()

    cam = GradCAM(model, target_layer=args.target_layer)(x, class_idx=1)

    plt.figure(figsize=(4, 4))
    plt.imshow(image_arr, cmap="gray")
    plt.imshow(cam, alpha=0.45)
    plt.axis("off")
    plt.title(f"Malignancy probability: {prob:.3f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Grad-CAM to {out_path}")


if __name__ == "__main__":
    main()
