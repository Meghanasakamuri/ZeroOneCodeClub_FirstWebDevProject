"""
Utility functions for AutoPainter
Includes loss functions, visualization, and checkpointing
"""
import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from config import Config


def save_checkpoint(gen, disc, opt_gen, opt_disc, epoch, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': gen.state_dict(),
        'discriminator_state_dict': disc.state_dict(),
        'optimizer_gen_state_dict': opt_gen.state_dict(),
        'optimizer_disc_state_dict': opt_disc.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(gen, disc, opt_gen, opt_disc, filename, device):
    """Load model checkpoint"""
    checkpoint = torch.load(filename, map_location=device)
    gen.load_state_dict(checkpoint['generator_state_dict'])
    disc.load_state_dict(checkpoint['discriminator_state_dict'])
    opt_gen.load_state_dict(checkpoint['optimizer_gen_state_dict'])
    opt_disc.load_state_dict(checkpoint['optimizer_disc_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded: {filename} (Epoch {epoch})")
    return epoch


def save_sample_images(gen, val_loader, epoch, folder, device, num_samples=5):
    """Save sample predictions during training"""
    gen.eval()
    os.makedirs(folder, exist_ok=True)
    
    with torch.no_grad():
        for idx, (sketch, real_painting) in enumerate(val_loader):
            if idx >= num_samples:
                break
            
            sketch = sketch.to(device)
            real_painting = real_painting.to(device)
            
            fake_painting = gen(sketch)
            
            # Denormalize for visualization
            sketch_vis = sketch * 0.5 + 0.5
            fake_vis = fake_painting * 0.5 + 0.5
            real_vis = real_painting * 0.5 + 0.5
            
            # Save individual images
            save_image(sketch_vis, os.path.join(folder, f"epoch{epoch}_sample{idx}_sketch.png"))
            save_image(fake_vis, os.path.join(folder, f"epoch{epoch}_sample{idx}_fake.png"))
            save_image(real_vis, os.path.join(folder, f"epoch{epoch}_sample{idx}_real.png"))
            
            # Create comparison grid
            comparison = torch.cat([sketch_vis.repeat(1, 3, 1, 1), fake_vis, real_vis], dim=3)
            save_image(comparison, os.path.join(folder, f"epoch{epoch}_sample{idx}_comparison.png"))
    
    gen.train()


def plot_losses(losses_dict, save_path):
    """Plot training losses"""
    plt.figure(figsize=(12, 6))
    
    for name, values in losses_dict.items():
        plt.plot(values, label=name)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


class LossManager:
    """
    Manages different loss functions for research comparisons
    RESEARCH QUESTIONS:
    - L1 vs Perceptual Loss
    - Pixel-wise vs Patch-wise Adversarial Loss
    """
    def __init__(self, loss_type="l1", device="cuda"):
        self.loss_type = loss_type
        self.device = device
        
        # Reconstruction losses
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        
        # Perceptual loss (VGG-based)
        if "perceptual" in loss_type or loss_type == "combined":
            from model import VGGPerceptualLoss
            self.perceptual_loss = VGGPerceptualLoss().to(device)
        else:
            self.perceptual_loss = None
        
        # GAN loss
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def generator_loss(self, fake_painting, real_painting, disc_fake):
        """
        Calculate generator loss
        
        Components:
        1. Adversarial loss (fool discriminator)
        2. Reconstruction loss (L1, Perceptual, or Combined)
        """
        # Adversarial loss
        adv_loss = self.bce_loss(disc_fake, torch.ones_like(disc_fake))
        
        # Reconstruction loss
        if self.loss_type == "l1":
            recon_loss = self.l1_loss(fake_painting, real_painting) * Config.L1_LAMBDA
            loss_dict = {
                'gen_adv': adv_loss.item(),
                'gen_l1': recon_loss.item()
            }
        
        elif self.loss_type == "perceptual":
            recon_loss = self.perceptual_loss(fake_painting, real_painting) * Config.PERCEPTUAL_LAMBDA
            loss_dict = {
                'gen_adv': adv_loss.item(),
                'gen_perceptual': recon_loss.item()
            }
        
        elif self.loss_type == "combined":
            l1 = self.l1_loss(fake_painting, real_painting) * Config.L1_LAMBDA
            perceptual = self.perceptual_loss(fake_painting, real_painting) * Config.PERCEPTUAL_LAMBDA
            recon_loss = l1 + perceptual
            loss_dict = {
                'gen_adv': adv_loss.item(),
                'gen_l1': l1.item(),
                'gen_perceptual': perceptual.item()
            }
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        total_loss = adv_loss + recon_loss
        loss_dict['gen_total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def discriminator_loss(self, disc_real, disc_fake):
        """
        Calculate discriminator loss
        
        Real images -> 1 (real)
        Fake images -> 0 (fake)
        """
        real_loss = self.bce_loss(disc_real, torch.ones_like(disc_real))
        fake_loss = self.bce_loss(disc_fake, torch.zeros_like(disc_fake))
        
        total_loss = (real_loss + fake_loss) / 2
        
        loss_dict = {
            'disc_real': real_loss.item(),
            'disc_fake': fake_loss.item(),
            'disc_total': total_loss.item()
        }
        
        return total_loss, loss_dict


def initialize_weights(model):
    """Initialize model weights (Xavier/He initialization)"""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


def denormalize(tensor):
    """Denormalize image tensor from [-1, 1] to [0, 1]"""
    return tensor * 0.5 + 0.5


def create_experiment_folder():
    """Create folder structure for current experiment"""
    exp_name = Config.get_experiment_name()
    
    folders = {
        'checkpoint': os.path.join(Config.CHECKPOINT_DIR, exp_name),
        'samples': os.path.join(Config.LOG_DIR, exp_name, 'samples'),
        'plots': os.path.join(Config.LOG_DIR, exp_name, 'plots')
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    return folders


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MetricsTracker:
    """Track and log training metrics"""
    def __init__(self):
        self.metrics = {}
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_average(self, key, last_n=100):
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        values = self.metrics[key][-last_n:]
        return sum(values) / len(values)
    
    def save(self, filepath):
        torch.save(self.metrics, filepath)
    
    def load(self, filepath):
        self.metrics = torch.load(filepath)


if __name__ == "__main__":
    print("Utils module loaded successfully!")
    print(f"Device: {Config.DEVICE}")
    print(f"Experiment name: {Config.get_experiment_name()}")
