from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not normalize)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    """Pix2Pix UNet generator supporting style conditioning."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        base_channels: int = 64,
        num_styles: int | None = None,
        style_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_styles = num_styles
        self.style_dim = style_dim if num_styles else 0
        conditioned_in_channels = in_channels + (self.style_dim if num_styles else 0)

        if num_styles:
            self.style_embedding = nn.Embedding(num_styles, style_dim)
        else:
            self.style_embedding = None

        self.down1 = DownsampleBlock(conditioned_in_channels, base_channels, normalize=False)
        self.down2 = DownsampleBlock(base_channels, base_channels * 2)
        self.down3 = DownsampleBlock(base_channels * 2, base_channels * 4)
        self.down4 = DownsampleBlock(base_channels * 4, base_channels * 8)
        self.down5 = DownsampleBlock(base_channels * 8, base_channels * 8)
        self.down6 = DownsampleBlock(base_channels * 8, base_channels * 8)
        self.down7 = DownsampleBlock(base_channels * 8, base_channels * 8)
        self.down8 = DownsampleBlock(base_channels * 8, base_channels * 8, normalize=False)

        self.up1 = UpsampleBlock(base_channels * 8, base_channels * 8, dropout=0.5)
        self.up2 = UpsampleBlock(base_channels * 16, base_channels * 8, dropout=0.5)
        self.up3 = UpsampleBlock(base_channels * 16, base_channels * 8, dropout=0.5)
        self.up4 = UpsampleBlock(base_channels * 16, base_channels * 8)
        self.up5 = UpsampleBlock(base_channels * 16, base_channels * 4)
        self.up6 = UpsampleBlock(base_channels * 8, base_channels * 2)
        self.up7 = UpsampleBlock(base_channels * 4, base_channels)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, style_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.num_styles:
            if style_ids is None:
                raise ValueError("style_ids must be provided when num_styles is set")
            style = self.style_embedding(style_ids)
            style = style.unsqueeze(-1).unsqueeze(-1)
            style = style.expand(-1, -1, x.size(2), x.size(3))
            conditioned = torch.cat([x, style], dim=1)
        else:
            conditioned = x

        d1 = self.down1(conditioned)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bottleneck = self.down8(d7)

        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))
        output = self.up8(torch.cat([u7, d1], dim=1))
        return output
