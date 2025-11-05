"""Model architectures for AutoPainter."""

from __future__ import annotations

import functools
from typing import Callable, Optional

import torch
from torch import nn

from .config import ModelConfig
from .utils import configure_logging, count_parameters


def get_norm_layer(norm_type: str) -> Callable[[int], nn.Module]:
    norm_type = norm_type.lower()
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    def init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise ValueError(f"Unsupported init_type: {init_type}")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: Optional[int] = None,
        submodule: Optional[nn.Module] = None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.outermost = outermost
        use_bias = False

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            assert submodule is not None, "Outermost block requires a submodule"
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            upnorm = norm_layer(outer_nc)
            model = [downrelu, downconv, uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            upnorm = norm_layer(outer_nc)
            assert submodule is not None, "Intermediate block requires a submodule"
            downnorm = norm_layer(inner_nc)
            model = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm]
            if use_dropout:
                model += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)


class UNetGenerator(nn.Module):
    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        num_downs: int = 8,
        ngf: int = 64,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()

        unet_block = UnetSkipConnectionBlock(
            outer_nc=ngf * 8,
            inner_nc=ngf * 8,
            input_nc=None,
            submodule=None,
            innermost=True,
            norm_layer=norm_layer,
        )
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        unet_block = UnetSkipConnectionBlock(
            ngf * 4,
            ngf * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2,
            ngf * 4,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
        )
        unet_block = UnetSkipConnectionBlock(
            ngf,
            ngf * 2,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
        )
        self.model = UnetSkipConnectionBlock(
            outer_nc=output_nc,
            inner_nc=ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )

    def forward(self, sketch: torch.Tensor) -> torch.Tensor:
        return self.model(sketch)


class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc: int, ndf: int = 64, norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d) -> None:
        super().__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        self.model = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)


def expand_style_channel(sketch: torch.Tensor, style_vec: Optional[torch.Tensor]) -> torch.Tensor:
    if style_vec is None or style_vec.numel() == 0:
        return sketch
    b, _, h, w = sketch.shape
    style_map = style_vec.view(b, -1, 1, 1).expand(b, -1, h, w)
    return torch.cat([sketch, style_map], dim=1)


def build_generator(config: ModelConfig, num_styles: int = 0) -> UNetGenerator:
    configure_logging()
    style_dim = config.style_condition_dim if num_styles > 1 else 0
    total_in_channels = config.input_channels + style_dim
    norm_layer = get_norm_layer(config.norm_layer)
    net = UNetGenerator(
        input_nc=total_in_channels,
        output_nc=config.output_channels,
        num_downs=8,
        ngf=config.generator_filters,
        norm_layer=norm_layer,
        use_dropout=config.use_dropout,
    )
    init_weights(net, config.init_type, config.init_gain)
    return net


def build_discriminator(config: ModelConfig, num_styles: int = 0) -> nn.Module:
    style_dim = config.style_condition_dim if num_styles > 1 else 0
    total_input_nc = config.input_channels + config.output_channels + style_dim
    norm_layer = get_norm_layer(config.norm_layer)
    if config.discriminator_type == "pixel":
        net = PixelDiscriminator(total_input_nc, config.discriminator_filters, norm_layer)
    else:
        net = PatchDiscriminator(total_input_nc, config.discriminator_filters, 3, norm_layer)
    init_weights(net, config.init_type, config.init_gain)
    return net

