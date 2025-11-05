"""Loss functions for AutoPainter."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
from torchvision import models


class GANLoss(nn.Module):
    def __init__(self, gan_mode: str = "lsgan") -> None:
        super().__init__()
        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))
        self.gan_mode = gan_mode
        if gan_mode not in {"lsgan", "vanilla", "hinge"}:
            raise ValueError(f"Unsupported gan_mode: {gan_mode}")
        self.loss = nn.MSELoss() if gan_mode == "lsgan" else nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        return (self.real_label if target_is_real else self.fake_label).expand_as(prediction)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if self.gan_mode == "hinge":
            if target_is_real:
                return torch.relu(1.0 - prediction).mean()
            return torch.relu(1.0 + prediction).mean()

        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)


class PerceptualLoss(nn.Module):
    def __init__(self, layer: str = "relu3_3") -> None:
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features
        self.layer_name_map = {
            "relu1_2": 3,
            "relu2_2": 8,
            "relu3_3": 15,
            "relu4_3": 22,
        }
        if layer not in self.layer_name_map:
            raise ValueError(f"Unsupported perceptual layer {layer}")
        cutoff = self.layer_name_map[layer]
        self.features = nn.Sequential(*list(vgg.children())[: cutoff + 1])
        for param in self.features.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.criterion = nn.L1Loss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert range [-1,1] to [0,1]
        prediction_norm = (prediction + 1.0) / 2.0
        target_norm = (target + 1.0) / 2.0
        prediction_norm = (prediction_norm - self.mean) / self.std
        target_norm = (target_norm - self.mean) / self.std
        pred_features = self.features(prediction_norm)
        target_features = self.features(target_norm)
        return self.criterion(pred_features, target_features)

