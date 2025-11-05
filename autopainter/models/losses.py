from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import torch
from torch import nn
from torchvision import models


class AdversarialLoss(nn.Module):
    def __init__(self, mode: str = "bce") -> None:
        super().__init__()
        self.mode = mode
        if mode == "bce":
            self.loss = nn.BCEWithLogitsLoss()
        elif mode == "lsgan":
            self.loss = nn.MSELoss()
        elif mode == "hinge":
            self.loss = None
        else:
            raise ValueError("Unsupported adversarial mode")

    def forward(self, logits: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if self.mode == "hinge":
            if target_is_real:
                return torch.relu(1.0 - logits).mean()
            return torch.relu(1.0 + logits).mean()

        target = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
        return self.loss(logits, target)


class PerceptualFeatureExtractor(nn.Module):
    def __init__(self, layers: Iterable[str], backbone: str = "vgg19") -> None:
        super().__init__()
        if backbone == "vgg19":
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        elif backbone == "vgg16":
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        else:
            raise ValueError("Unsupported backbone")

        self.layers = set(layers)
        self.selected_layers = layers
        self.vgg = nn.Sequential()
        self.layer_map = {}

        idx = 0
        layer_names = {}
        block = 1
        conv = 0
        relu = 0
        pool = 0
        for layer in vgg:
            if isinstance(layer, nn.Conv2d):
                conv += 1
                name = f"conv_{block}_{conv}"
            elif isinstance(layer, nn.ReLU):
                relu += 1
                name = f"relu_{block}_{relu}"
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                pool += 1
                name = f"pool_{block}_{pool}"
                block += 1
                conv = 0
                relu = 0
            else:
                name = f"layer_{idx}"

            self.vgg.add_module(name, layer)
            layer_names[idx] = name
            idx += 1
            for sel in self.layers:
                if sel == name:
                    self.layer_map[sel] = len(self.vgg) - 1
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        outputs: OrderedDict[str, torch.Tensor] = OrderedDict()
        for idx, (name, layer) in enumerate(self.vgg._modules.items()):
            x = layer(x)
            if name in self.layers:
                outputs[name] = x
        return outputs


class PerceptualLoss(nn.Module):
    def __init__(self, layers: Iterable[str], backbone: str = "vgg19") -> None:
        super().__init__()
        self.extractor = PerceptualFeatureExtractor(layers, backbone)
        self.criterion = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = (pred + 1.0) / 2.0
        target_norm = (target + 1.0) / 2.0
        features_pred = self.extractor(pred_norm)
        features_target = self.extractor(target_norm)
        loss = 0.0
        for key in features_pred:
            loss = loss + self.criterion(features_pred[key], features_target[key])
        return loss


class ReconstructionLoss(nn.Module):
    def __init__(self, mode: str = "l1") -> None:
        super().__init__()
        if mode == "l1":
            self.loss = nn.L1Loss()
        elif mode == "l2":
            self.loss = nn.MSELoss()
        else:
            raise ValueError("Unsupported reconstruction mode")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


def feature_matching_loss(real_features: list[torch.Tensor], fake_features: list[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    l1 = nn.L1Loss()
    for real, fake in zip(real_features, fake_features):
        loss = loss + l1(real.detach(), fake)
    return loss
