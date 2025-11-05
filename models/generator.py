"""
U-Net Generator for Pix2Pix
Converts grayscale sketches to colored paintings
"""
import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """
        Args:
            input_nc: Number of input channels (1 for grayscale)
            output_nc: Number of output channels (3 for RGB)
            num_downs: Number of downsampling layers
            ngf: Number of generator filters in first conv layer
            norm_layer: Normalization layer
            use_dropout: Whether to use dropout
        """
        super(UNetGenerator, self).__init__()
        
        # Build U-Net
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, 
                                             norm_layer=norm_layer, innermost=True)
        
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, 
                                                 submodule=unet_block, norm_layer=norm_layer, 
                                                 use_dropout=use_dropout)
        
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, 
                                             submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, 
                                             submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, 
                                             submodule=unet_block, norm_layer=norm_layer)
        
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, 
                                            submodule=unet_block, outermost=True, 
                                            norm_layer=norm_layer)

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, 
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        
        if input_nc is None:
            input_nc = outer_nc
        
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                           stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                      kernel_size=4, stride=2,
                                      padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                      kernel_size=4, stride=2,
                                      padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                      kernel_size=4, stride=2,
                                      padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
