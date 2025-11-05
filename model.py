"""
Pix2Pix GAN Model Architecture for AutoPainter
Includes Generator (U-Net) and Discriminator (PatchGAN/PixelGAN)
"""
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and activation"""
    def __init__(self, in_channels, out_channels, down=True, use_activation=True, use_dropout=False, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if use_activation else nn.Identity(),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    """
    U-Net Generator for Pix2Pix
    Converts grayscale sketches to colored paintings
    """
    def __init__(self, in_channels=1, out_channels=3, features=64):
        super().__init__()
        
        # Encoder (Downsampling)
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        
        self.down1 = ConvBlock(features, features * 2, down=True, use_activation=True, kernel_size=4, stride=2, padding=1)
        self.down2 = ConvBlock(features * 2, features * 4, down=True, use_activation=True, kernel_size=4, stride=2, padding=1)
        self.down3 = ConvBlock(features * 4, features * 8, down=True, use_activation=True, kernel_size=4, stride=2, padding=1)
        self.down4 = ConvBlock(features * 8, features * 8, down=True, use_activation=True, kernel_size=4, stride=2, padding=1)
        self.down5 = ConvBlock(features * 8, features * 8, down=True, use_activation=True, kernel_size=4, stride=2, padding=1)
        self.down6 = ConvBlock(features * 8, features * 8, down=True, use_activation=True, kernel_size=4, stride=2, padding=1)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1),
            nn.ReLU()
        )
        
        # Decoder (Upsampling)
        self.up1 = ConvBlock(features * 8, features * 8, down=False, use_activation=True, use_dropout=True, kernel_size=4, stride=2, padding=1)
        self.up2 = ConvBlock(features * 8 * 2, features * 8, down=False, use_activation=True, use_dropout=True, kernel_size=4, stride=2, padding=1)
        self.up3 = ConvBlock(features * 8 * 2, features * 8, down=False, use_activation=True, use_dropout=True, kernel_size=4, stride=2, padding=1)
        self.up4 = ConvBlock(features * 8 * 2, features * 8, down=False, use_activation=True, use_dropout=False, kernel_size=4, stride=2, padding=1)
        self.up5 = ConvBlock(features * 8 * 2, features * 4, down=False, use_activation=True, use_dropout=False, kernel_size=4, stride=2, padding=1)
        self.up6 = ConvBlock(features * 4 * 2, features * 2, down=False, use_activation=True, use_dropout=False, kernel_size=4, stride=2, padding=1)
        self.up7 = ConvBlock(features * 2 * 2, features, down=False, use_activation=True, use_dropout=False, kernel_size=4, stride=2, padding=1)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # Encoder with skip connections
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        
        bottleneck = self.bottleneck(d7)
        
        # Decoder with skip connections
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        
        return self.final_up(torch.cat([up7, d1], 1))


class Discriminator(nn.Module):
    """
    PatchGAN or PixelGAN Discriminator
    RESEARCH: Compares patch-wise vs pixel-wise adversarial loss
    """
    def __init__(self, in_channels=4, features=64, disc_type="patchgan"):
        """
        Args:
            in_channels: 4 (grayscale sketch + RGB painting)
            features: Number of base features
            disc_type: "patchgan" for patch-wise or "pixel" for pixel-wise
        """
        super().__init__()
        self.disc_type = disc_type
        
        if disc_type == "patchgan":
            # PatchGAN: 70x70 receptive field
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
                nn.LeakyReLU(0.2),
                
                self._block(features, features * 2, 4, 2, 1),
                self._block(features * 2, features * 4, 4, 2, 1),
                self._block(features * 4, features * 8, 4, 1, 1),
                
                nn.Conv2d(features * 8, 1, 4, 1, 1, padding_mode="reflect"),
            )
        else:
            # PixelGAN: 1x1 receptive field
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, features, 1, 1, 0),
                nn.LeakyReLU(0.2),
                
                self._block(features, features * 2, 1, 1, 0),
                
                nn.Conv2d(features * 2, 1, 1, 1, 0),
            )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, y):
        # Concatenate sketch (x) and painting (y)
        x = torch.cat([x, y], dim=1)
        return self.model(x)


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG16 features
    RESEARCH: Compare L1 vs Perceptual loss for artistic style transfer
    """
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        # Use features from multiple layers for perceptual loss
        self.feature_layers = nn.ModuleList([
            vgg.features[:4],   # relu1_2
            vgg.features[:9],   # relu2_2
            vgg.features[:16],  # relu3_3
            vgg.features[:23],  # relu4_3
        ])
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        loss = 0.0
        x = pred
        y = target
        
        for layer in self.feature_layers:
            x = layer(x)
            y = layer(y)
            loss += nn.functional.l1_loss(x, y)
        
        return loss


def test_model():
    """Test function to verify model architectures"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test Generator
    gen = Generator(in_channels=1, out_channels=3, features=64).to(device)
    x = torch.randn(1, 1, 256, 256).to(device)
    y = gen(x)
    print(f"Generator Input: {x.shape} -> Output: {y.shape}")
    
    # Test PatchGAN Discriminator
    disc_patch = Discriminator(in_channels=4, features=64, disc_type="patchgan").to(device)
    pred = disc_patch(x, y)
    print(f"PatchGAN Discriminator Output: {pred.shape}")
    
    # Test PixelGAN Discriminator
    disc_pixel = Discriminator(in_channels=4, features=64, disc_type="pixel").to(device)
    pred = disc_pixel(x, y)
    print(f"PixelGAN Discriminator Output: {pred.shape}")
    
    # Count parameters
    gen_params = sum(p.numel() for p in gen.parameters())
    disc_params = sum(p.numel() for p in disc_patch.parameters())
    print(f"\nGenerator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")


if __name__ == "__main__":
    test_model()
