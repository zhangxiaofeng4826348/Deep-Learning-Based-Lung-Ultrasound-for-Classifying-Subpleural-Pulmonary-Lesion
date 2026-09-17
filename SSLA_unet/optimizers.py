"""Optimizer and scheduler builders."""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


def _trainable_params(modules) -> list:
    params = []
    for module in modules:
        params.extend([p for p in module.parameters() if p.requires_grad])
    return params


def build_ssl_optimizer(model, projection_head, lr: float = 1e-4, weight_decay: float = 1e-4):
    params = list(model.parameters()) + list(projection_head.parameters())
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def build_finetune_optimizer(
    model,
    lr_encoder: float = 5e-6,
    lr_decoder: float = 1e-5,
    lr_classifier: float = 1e-5,
    weight_decay: float = 1e-5,
    freeze_encoder: bool = False,
    freeze_swin: bool = False,
):
    encoder_modules = [model.R1, model.R2, model.R3, model.R4, model.RS]
    if freeze_encoder:
        for module in encoder_modules:
            for p in module.parameters():
                p.requires_grad = False

    if hasattr(model, "swin"):
        if freeze_swin:
            for p in model.swin.parameters():
                p.requires_grad = False
        encoder_modules.append(model.swin)

    decoder_modules = [model.U1, model.U2, model.U3, model.U4, model.CD1, model.CD2, model.CD3, model.CD4, model.seg_head]

    param_groups = []
    encoder_params = _trainable_params(encoder_modules)
    decoder_params = _trainable_params(decoder_modules)
    classifier_params = [p for p in model.cls_head.parameters() if p.requires_grad]

    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": lr_encoder})
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": lr_decoder})
    if classifier_params:
        param_groups.append({"params": classifier_params, "lr": lr_classifier})

    if not param_groups:
        raise ValueError("No trainable parameters were found")
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def build_scheduler(optimizer, epochs: int, min_lr: float = 1e-6):
    return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
