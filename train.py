"""
Training script for AutoPainter Pix2Pix GAN

This script implements:
- Full training loop with Generator and Discriminator
- Support for multiple loss types (L1, Perceptual, Combined)
- Support for PatchGAN and PixelGAN discriminators
- Checkpoint saving and loading
- Sample image generation
- Metrics tracking and visualization

RESEARCH QUESTIONS ADDRESSED:
1. Effect of L1 vs Perceptual Loss (Config.LOSS_TYPE)
2. Amount of paired data needed (Config.TRAIN_SIZE)
3. Multiple artistic styles (Config.STYLE_MODE)
4. Pixel-wise vs Patch-wise adversarial loss (Config.DISC_TYPE)
"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from model import Generator, Discriminator
from dataset import get_dataloaders
from utils import (
    save_checkpoint, load_checkpoint, save_sample_images,
    initialize_weights, LossManager, MetricsTracker,
    create_experiment_folder, count_parameters
)


def train_epoch(gen, disc, train_loader, opt_gen, opt_disc, loss_manager, device, epoch):
    """Train for one epoch"""
    gen.train()
    disc.train()
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch}")
    epoch_metrics = {
        'gen_total': [],
        'disc_total': [],
        'gen_adv': [],
        'gen_recon': []
    }
    
    for batch_idx, (sketch, real_painting) in enumerate(loop):
        sketch = sketch.to(device)
        real_painting = real_painting.to(device)
        
        # ==================== Train Discriminator ====================
        fake_painting = gen(sketch)
        
        # Discriminator on real and fake
        disc_real = disc(sketch, real_painting)
        disc_fake = disc(sketch, fake_painting.detach())
        
        disc_loss, disc_loss_dict = loss_manager.discriminator_loss(disc_real, disc_fake)
        
        opt_disc.zero_grad()
        disc_loss.backward()
        opt_disc.step()
        
        # ==================== Train Generator ====================
        disc_fake = disc(sketch, fake_painting)
        
        gen_loss, gen_loss_dict = loss_manager.generator_loss(
            fake_painting, real_painting, disc_fake
        )
        
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()
        
        # ==================== Logging ====================
        epoch_metrics['gen_total'].append(gen_loss_dict['gen_total'])
        epoch_metrics['disc_total'].append(disc_loss_dict['disc_total'])
        epoch_metrics['gen_adv'].append(gen_loss_dict['gen_adv'])
        
        # Update progress bar
        loop.set_postfix(
            G_loss=f"{gen_loss_dict['gen_total']:.4f}",
            D_loss=f"{disc_loss_dict['disc_total']:.4f}"
        )
    
    # Return average metrics for epoch
    return {k: sum(v) / len(v) for k, v in epoch_metrics.items()}


def validate(gen, val_loader, device):
    """Validation phase"""
    gen.eval()
    total_l1_loss = 0
    
    with torch.no_grad():
        for sketch, real_painting in val_loader:
            sketch = sketch.to(device)
            real_painting = real_painting.to(device)
            
            fake_painting = gen(sketch)
            l1_loss = torch.abs(fake_painting - real_painting).mean()
            total_l1_loss += l1_loss.item()
    
    avg_l1_loss = total_l1_loss / len(val_loader)
    return avg_l1_loss


def main():
    """Main training function"""
    print("=" * 70)
    print("AutoPainter Training - Pix2Pix GAN for Sketch-to-Painting")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Loss Type: {Config.LOSS_TYPE}")
    print(f"  Discriminator Type: {Config.DISC_TYPE}")
    print(f"  Training Size: {Config.TRAIN_SIZE if Config.TRAIN_SIZE else 'All data'}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Epochs: {Config.NUM_EPOCHS}")
    print(f"  Image Size: {Config.IMAGE_SIZE}")
    print(f"  Experiment: {Config.get_experiment_name()}")
    print("=" * 70)
    
    # Create folders
    folders = create_experiment_folder()
    
    # Initialize models
    print("\nInitializing models...")
    gen = Generator(
        in_channels=Config.INPUT_CHANNELS,
        out_channels=Config.OUTPUT_CHANNELS,
        features=Config.GEN_FEATURES
    ).to(Config.DEVICE)
    
    disc = Discriminator(
        in_channels=Config.INPUT_CHANNELS + Config.OUTPUT_CHANNELS,
        features=Config.DISC_FEATURES,
        disc_type=Config.DISC_TYPE
    ).to(Config.DEVICE)
    
    # Initialize weights
    initialize_weights(gen)
    initialize_weights(disc)
    
    print(f"Generator parameters: {count_parameters(gen):,}")
    print(f"Discriminator parameters: {count_parameters(disc):,}")
    
    # Optimizers
    opt_gen = optim.Adam(
        gen.parameters(),
        lr=Config.LEARNING_RATE,
        betas=(Config.BETA1, Config.BETA2)
    )
    
    opt_disc = optim.Adam(
        disc.parameters(),
        lr=Config.LEARNING_RATE,
        betas=(Config.BETA1, Config.BETA2)
    )
    
    # Loss manager
    loss_manager = LossManager(loss_type=Config.LOSS_TYPE, device=Config.DEVICE)
    
    # Load checkpoint if specified
    start_epoch = 0
    if Config.LOAD_CHECKPOINT and Config.CHECKPOINT_PATH and os.path.exists(Config.CHECKPOINT_PATH):
        start_epoch = load_checkpoint(gen, disc, opt_gen, opt_disc, Config.CHECKPOINT_PATH, Config.DEVICE)
        start_epoch += 1
    
    # Data loaders
    print("\nLoading dataset...")
    try:
        train_loader, val_loader = get_dataloaders(dataset_type="separate")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please run download_dataset.py first to prepare the data.")
        return
    
    if len(train_loader.dataset) == 0:
        print("ERROR: No training data found!")
        print(f"Please ensure data exists in: {Config.DATA_DIR}")
        return
    
    # Metrics tracker
    metrics = MetricsTracker()
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70 + "\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        # Train
        epoch_metrics = train_epoch(
            gen, disc, train_loader, opt_gen, opt_disc,
            loss_manager, Config.DEVICE, epoch
        )
        
        # Validate
        val_loss = validate(gen, val_loader, Config.DEVICE)
        
        # Update metrics
        metrics.update(
            gen_loss=epoch_metrics['gen_total'],
            disc_loss=epoch_metrics['disc_total'],
            val_loss=val_loss
        )
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Gen Loss: {epoch_metrics['gen_total']:.4f}")
        print(f"  Disc Loss: {epoch_metrics['disc_total']:.4f}")
        print(f"  Val L1 Loss: {val_loss:.4f}")
        
        # Save sample images
        if epoch % Config.SAVE_INTERVAL == 0 or epoch == Config.NUM_EPOCHS - 1:
            save_sample_images(
                gen, val_loader, epoch,
                folders['samples'], Config.DEVICE
            )
        
        # Save checkpoint
        if epoch % Config.SAVE_INTERVAL == 0 or epoch == Config.NUM_EPOCHS - 1:
            checkpoint_path = os.path.join(
                folders['checkpoint'],
                f"checkpoint_epoch_{epoch}.pth"
            )
            save_checkpoint(gen, disc, opt_gen, opt_disc, epoch, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(folders['checkpoint'], "best_model.pth")
            save_checkpoint(gen, disc, opt_gen, opt_disc, epoch, best_path)
            print(f"  New best model saved! (Val Loss: {val_loss:.4f})")
        
        # Plot losses
        if (epoch + 1) % 5 == 0:
            plt.figure(figsize=(10, 5))
            plt.plot(metrics.metrics['gen_loss'], label='Generator Loss')
            plt.plot(metrics.metrics['disc_loss'], label='Discriminator Loss')
            plt.plot(metrics.metrics['val_loss'], label='Validation L1 Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(folders['plots'], 'training_losses.png'))
            plt.close()
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {folders['checkpoint']}")
    print(f"Sample images saved in: {folders['samples']}")
    print(f"Plots saved in: {folders['plots']}")


if __name__ == "__main__":
    main()
