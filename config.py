"""
Configuration file for AutoPainter
"""
import torch

class Config:
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data paths
    train_sketch_dir = 'data/train/sketches'
    train_painting_dir = 'data/train/paintings'
    val_sketch_dir = 'data/val/sketches'
    val_painting_dir = 'data/val/paintings'
    
    # Model parameters
    input_nc = 1  # Grayscale input
    output_nc = 3  # RGB output
    ngf = 64  # Generator filters
    ndf = 64  # Discriminator filters
    n_layers_d = 3  # Discriminator layers
    
    # Training parameters
    batch_size = 1
    num_epochs = 200
    lr_g = 0.0002  # Generator learning rate
    lr_d = 0.0002  # Discriminator learning rate
    beta1 = 0.5
    beta2 = 0.999
    
    # Loss weights
    lambda_l1 = 100.0  # L1 loss weight
    lambda_perceptual = 10.0  # Perceptual loss weight
    use_perceptual_loss = True  # Set to False to use only L1 loss
    use_patch_loss = True  # Use patch-wise adversarial loss
    
    # Image parameters
    image_size = 256
    
    # Training settings
    num_workers = 4
    save_interval = 10  # Save checkpoint every N epochs
    print_interval = 100  # Print loss every N iterations
    
    # Paths
    checkpoint_dir = 'checkpoints'
    output_dir = 'outputs'
    pretrained_model_path = 'checkpoints/pretrained_model.pth'
    
    # GAN mode
    gan_mode = 'lsgan'  # 'lsgan' or 'vanilla'
