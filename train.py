"""
Training script for AutoPainter (Pix2Pix GAN)
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from models import UNetGenerator, PatchGANDiscriminator
from utils.losses import GANLoss, PerceptualLoss
from utils.utils import save_checkpoint, tensor_to_image, init_weights
from config import Config


def train():
    config = Config()
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs('data/train/sketches', exist_ok=True)
    os.makedirs('data/train/paintings', exist_ok=True)
    
    # Initialize models
    generator = UNetGenerator(
        input_nc=config.input_nc,
        output_nc=config.output_nc,
        ngf=config.ngf
    ).to(config.device)
    
    discriminator = PatchGANDiscriminator(
        input_nc=config.input_nc + config.output_nc,
        ndf=config.ndf,
        n_layers=config.n_layers_d
    ).to(config.device)
    
    # Initialize weights
    init_weights(generator)
    init_weights(discriminator)
    
    # Loss functions
    criterion_gan = GANLoss(config.gan_mode).to(config.device)
    criterion_l1 = nn.L1Loss()
    if config.use_perceptual_loss:
        criterion_perceptual = PerceptualLoss().to(config.device)
    
    # Optimizers
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=config.lr_g,
        betas=(config.beta1, config.beta2)
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.lr_d,
        betas=(config.beta1, config.beta2)
    )
    
    # Load dataset
    try:
        from utils.dataset import get_dataloader
        train_loader = get_dataloader(
            config.train_sketch_dir,
            config.train_painting_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            image_size=config.image_size,
            mode='train'
        )
        print(f"Loaded {len(train_loader.dataset)} training images")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating sample dataset structure...")
        print("Please place your training images in:")
        print(f"  - {config.train_sketch_dir}/")
        print(f"  - {config.train_painting_dir}/")
        return
    
    # Training loop
    print("Starting training...")
    print(f"Device: {config.device}")
    print(f"Using perceptual loss: {config.use_perceptual_loss}")
    print(f"Using patch-wise loss: {config.use_patch_loss}")
    
    for epoch in range(config.num_epochs):
        generator.train()
        discriminator.train()
        
        for i, (sketches, paintings) in enumerate(train_loader):
            sketches = sketches.to(config.device)
            paintings = paintings.to(config.device)
            
            # ========== Train Discriminator ==========
            optimizer_d.zero_grad()
            
            # Real pairs
            real_pairs = torch.cat([sketches, paintings], dim=1)
            pred_real = discriminator(real_pairs)
            loss_d_real = criterion_gan(pred_real, True)
            
            # Fake pairs
            with torch.no_grad():
                fake_paintings = generator(sketches)
            fake_pairs = torch.cat([sketches, fake_paintings.detach()], dim=1)
            pred_fake = discriminator(fake_pairs)
            loss_d_fake = criterion_gan(pred_fake, False)
            
            # Discriminator loss
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            optimizer_d.step()
            
            # ========== Train Generator ==========
            optimizer_g.zero_grad()
            
            # Generate fake paintings
            fake_paintings = generator(sketches)
            fake_pairs = torch.cat([sketches, fake_paintings], dim=1)
            
            # GAN loss
            pred_fake = discriminator(fake_pairs)
            loss_g_gan = criterion_gan(pred_fake, True)
            
            # L1 loss
            loss_g_l1 = criterion_l1(fake_paintings, paintings) * config.lambda_l1
            
            # Perceptual loss
            if config.use_perceptual_loss:
                loss_g_perceptual = criterion_perceptual(fake_paintings, paintings) * config.lambda_perceptual
            else:
                loss_g_perceptual = torch.tensor(0.0).to(config.device)
            
            # Total generator loss
            loss_g = loss_g_gan + loss_g_l1 + loss_g_perceptual
            loss_g.backward()
            optimizer_g.step()
            
            # Print progress
            if (i + 1) % config.print_interval == 0:
                print(f"Epoch [{epoch+1}/{config.num_epochs}], "
                      f"Step [{i+1}/{len(train_loader)}], "
                      f"Loss_D: {loss_d.item():.4f}, "
                      f"Loss_G: {loss_g.item():.4f}, "
                      f"Loss_G_GAN: {loss_g_gan.item():.4f}, "
                      f"Loss_G_L1: {loss_g_l1.item():.4f}, "
                      f"Loss_G_Perceptual: {loss_g_perceptual.item():.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f'checkpoint_epoch_{epoch+1}.pth'
            )
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer_g.state_dict(),
                'config': config.__dict__,
            }, checkpoint_path)
            
            # Save sample output
            generator.eval()
            with torch.no_grad():
                sample_sketch = sketches[0:1]
                sample_painting = paintings[0:1]
                sample_output = generator(sample_sketch)
                
                # Save images
                output_path = os.path.join(config.output_dir, f'epoch_{epoch+1}_output.png')
                save_image(sample_output, output_path)
                print(f"Saved sample output to {output_path}")
    
    # Save final model
    final_path = os.path.join(config.checkpoint_dir, 'final_model.pth')
    save_checkpoint({
        'epoch': config.num_epochs,
        'model_state_dict': generator.state_dict(),
        'optimizer_state_dict': optimizer_g.state_dict(),
        'config': config.__dict__,
    }, final_path)
    
    print("Training completed!")


def save_image(tensor, path):
    """Save tensor as image"""
    image = tensor_to_image(tensor)
    image.save(path)


if __name__ == '__main__':
    train()
