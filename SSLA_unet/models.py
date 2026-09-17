"""Model definitions for SLLA-UNet.

The full SLLA-UNet model combines:
1) a U-Net-like segmentation branch,
2) multi-scale encoder feature aggregation, and
3) a Swin Transformer branch for global contextual features.

Input images are expected to be 224 x 224 RGB tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.swin_transformer import swin_t


class ResidualBlock(nn.Module):
    """Two-convolution residual block used in the convolutional encoder/decoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.shortcut(x))


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for full and ablated SLLA-UNet variants."""

    num_classes: int = 2
    input_size: int = 224
    use_multiscale_features: bool = True
    use_swin: bool = True
    dropout: float = 0.3


class SLLAUNet(nn.Module):
    """Joint segmentation-classification model for SPL analysis.

    Full model configuration:
        SLLAUNet(ModelConfig(use_multiscale_features=True, use_swin=True))

    Ablation configurations:
        non-MS:       use_multiscale_features=False, use_swin=True
        non-Swin:     use_multiscale_features=True,  use_swin=False
        non-MS-Swin:  use_multiscale_features=False, use_swin=False
        non-SSL:      same architecture as the full model, initialized randomly
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.R1 = ResidualBlock(3, 32)
        self.R2 = ResidualBlock(32, 64)
        self.R3 = ResidualBlock(64, 128)
        self.R4 = ResidualBlock(128, 256)
        self.RS = ResidualBlock(256, 512)

        # Optional Swin Transformer branch
        self.swin_out_dim = 768 if self.config.use_swin else 0
        if self.config.use_swin:
            self.swin = swin_t(weights=None)
            self.swin.head = nn.Identity()

        # Decoder / segmentation head
        self.U1 = nn.Conv2d(512, 256, kernel_size=1)
        self.U2 = nn.Conv2d(256, 128, kernel_size=1)
        self.U3 = nn.Conv2d(128, 64, kernel_size=1)
        self.U4 = nn.Conv2d(64, 32, kernel_size=1)
        self.CD1 = ResidualBlock(512, 256)
        self.CD2 = ResidualBlock(256, 128)
        self.CD3 = ResidualBlock(128, 64)
        self.CD4 = ResidualBlock(64, 32)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.seg_head = nn.Conv2d(32, 1, kernel_size=1)

        # Classification feature dimensionality
        self.gap = nn.AdaptiveAvgPool2d(1)
        if self.config.use_multiscale_features:
            conv_dim = 32 + 64 + 128 + 256 + 512
        else:
            conv_dim = 512
        cls_in_dim = conv_dim + self.swin_out_dim

        self.cls_head = nn.Sequential(
            nn.Linear(cls_in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(512, self.config.num_classes),
        )

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r1 = self.R1(x)
        r2 = self.R2(self.pool(r1))
        r3 = self.R3(self.pool(r2))
        r4 = self.R4(self.pool(r3))
        rs = self.RS(self.pool(r4))
        return r1, r2, r3, r4, rs

    def _decode(self, r1: torch.Tensor, r2: torch.Tensor, r3: torch.Tensor, r4: torch.Tensor, rs: torch.Tensor) -> torch.Tensor:
        d1 = self.CD1(torch.cat([self.up(self.U1(rs)), r4], dim=1))
        d2 = self.CD2(torch.cat([self.U2(self.up(d1)), r3], dim=1))
        d3 = self.CD3(torch.cat([self.U3(self.up(d2)), r2], dim=1))
        d4 = self.CD4(torch.cat([self.U4(self.up(d3)), r1], dim=1))
        return self.seg_head(d4)

    def _classification_features(
        self,
        x: torch.Tensor,
        r1: torch.Tensor,
        r2: torch.Tensor,
        r3: torch.Tensor,
        r4: torch.Tensor,
        rs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.config.use_multiscale_features:
            conv_feats = torch.cat([self.gap(f).flatten(1) for f in [r1, r2, r3, r4, rs]], dim=1)
        else:
            conv_feats = self.gap(rs).flatten(1)

        swin_feat = None
        if self.config.use_swin:
            if x.shape[-2:] != (self.config.input_size, self.config.input_size):
                x_swin = F.interpolate(
                    x,
                    size=(self.config.input_size, self.config.input_size),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                x_swin = x
            swin_feat = self.swin(x_swin)
            return torch.cat([conv_feats, swin_feat], dim=1), swin_feat
        return conv_feats, swin_feat

    def forward(self, x: torch.Tensor):
        r1, r2, r3, r4, rs = self._encode(x)
        seg_logits = self._decode(r1, r2, r3, r4, rs)
        fused_feats, swin_feat = self._classification_features(x, r1, r2, r3, r4, rs)
        cls_logits = self.cls_head(fused_feats)
        cam_features: Dict[str, torch.Tensor] = {"R1": r1, "R2": r2, "R3": r3, "R4": r4, "RS": rs, "Fused": fused_feats}
        return seg_logits, cls_logits, fused_feats, swin_feat, cam_features


class ProjectionHead(nn.Module):
    """Projection head used for SimCLR-style self-supervised pretraining."""

    def __init__(self, in_dim: int, proj_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(variant: str = "full", input_size: int = 224, num_classes: int = 2) -> SLLAUNet:
    """Factory for the full SLLA-UNet and ablation variants."""
    variant = variant.lower().replace("_", "-")
    if variant in {"full", "slla-unet", "non-ssl"}:
        config = ModelConfig(num_classes=num_classes, input_size=input_size, use_multiscale_features=True, use_swin=True)
    elif variant in {"non-ms", "no-ms"}:
        config = ModelConfig(num_classes=num_classes, input_size=input_size, use_multiscale_features=False, use_swin=True)
    elif variant in {"non-swin", "no-swin"}:
        config = ModelConfig(num_classes=num_classes, input_size=input_size, use_multiscale_features=True, use_swin=False)
    elif variant in {"non-ms-swin", "no-ms-swin"}:
        config = ModelConfig(num_classes=num_classes, input_size=input_size, use_multiscale_features=False, use_swin=False)
    else:
        raise ValueError(f"Unknown model variant: {variant}")
    return SLLAUNet(config)


def classifier_input_dim(model: SLLAUNet) -> int:
    """Return the dimension of the feature vector before the classification head."""
    return model.cls_head[0].in_features
