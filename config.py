"""
Configuration file for AutoPainter Pix2Pix GAN
"""
import torch

class Config:
    # Model Architecture
    INPUT_CHANNELS = 1  # Grayscale input (sketch)
    OUTPUT_CHANNELS = 3  # RGB output (painting)
    GEN_FEATURES = 64
    DISC_FEATURES = 64
    
    # Training Hyperparameters
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 16
    NUM_EPOCHS = 200
    BETA1 = 0.5
    BETA2 = 0.999
    
    # Loss Configuration - RESEARCH QUESTION: L1 vs Perceptual Loss
    LOSS_TYPE = "l1"  # Options: "l1", "perceptual", "combined"
    L1_LAMBDA = 100  # Weight for L1 loss
    PERCEPTUAL_LAMBDA = 10  # Weight for perceptual loss
    
    # Discriminator Loss Type - RESEARCH QUESTION: Pixel-wise vs Patch-wise
    DISC_TYPE = "patchgan"  # Options: "patchgan" (patch-wise), "pixel" (pixel-wise)
    
    # Image Settings
    IMAGE_SIZE = 256
    LOAD_SIZE = 286  # For random cropping during augmentation
    
    # Dataset - RESEARCH QUESTION: How much paired data is needed?
    DATA_DIR = "data/monet_sketches"
    TRAIN_SIZE = None  # None = use all data, or specify number (e.g., 100, 500, 1000)
    
    # Style Settings - RESEARCH QUESTION: Multiple artistic styles
    STYLE_MODE = "single"  # Options: "single", "multi"
    STYLES = ["monet"]  # Available: ["monet", "oil", "vangogh", "cezanne"]
    USE_STYLE_CONDITIONING = False  # If True, uses conditional GAN for multi-style
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoints
    CHECKPOINT_DIR = "checkpoints"
    SAVE_INTERVAL = 10  # Save every N epochs
    LOAD_CHECKPOINT = False
    CHECKPOINT_PATH = None
    
    # Logging
    LOG_DIR = "logs"
    SAMPLE_INTERVAL = 100  # Save sample images every N iterations
    
    # Data Augmentation
    USE_AUGMENTATION = True
    HORIZONTAL_FLIP = True
    
    # Pretrained Model
    USE_PRETRAINED = True
    PRETRAINED_PATH = "pretrained/pix2pix_monet.pth"
    
    @staticmethod
    def get_experiment_name():
        """Generate experiment name based on configuration"""
        name_parts = [
            f"loss_{Config.LOSS_TYPE}",
            f"disc_{Config.DISC_TYPE}",
            f"data_{Config.TRAIN_SIZE if Config.TRAIN_SIZE else 'all'}",
            f"style_{Config.STYLE_MODE}"
        ]
        return "_".join(name_parts)
