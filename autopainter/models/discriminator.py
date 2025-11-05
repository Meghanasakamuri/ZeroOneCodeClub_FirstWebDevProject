from __future__ import annotations

import torch
from torch import nn


class PatchDiscriminator(nn.Module):
    """Multi-layer PatchGAN discriminator supporting pixel discriminators."""

    def __init__(self, in_channels: int = 4, base_channels: int = 64, mode: str = "patch") -> None:
        super().__init__()
        if mode not in {"patch", "pixel"}:
            raise ValueError("mode must be 'patch' or 'pixel'")
        self.mode = mode

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(in_channels, base_channels, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        n_layers = 3
        for n in range(1, n_layers + 1):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            stride = 1 if n == n_layers else 2
            sequence += [
                nn.Conv2d(base_channels * nf_mult_prev, base_channels * nf_mult, kernel_size=kw, stride=stride, padding=padw, bias=False),
                nn.BatchNorm2d(base_channels * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        sequence += [
            nn.Conv2d(base_channels * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
        ]

        if self.mode == "pixel":
            sequence.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.model = nn.Sequential(*sequence)

    def forward(
        self,
        input_image: torch.Tensor,
        target_image: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        x = torch.cat([input_image, target_image], dim=1)
        features: list[torch.Tensor] = []
        out = x
        for layer in self.model:
            out = layer(out)
            if return_features and isinstance(layer, nn.LeakyReLU):
                features.append(out)

        logits = out
        if self.mode == "pixel":
            logits = logits.view(logits.size(0), -1)

        if return_features:
            return logits, features
        return logits
