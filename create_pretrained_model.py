"""
Create a pretrained model checkpoint for inference
This script initializes a model with random weights and saves it as a pretrained model.
For actual training, you should train the model first using train.py
"""
import os
import torch
from models import UNetGenerator
from utils.utils import save_checkpoint, init_weights
from config import Config


def create_pretrained_model():
    """Create a pretrained model checkpoint"""
    config = Config()
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Initialize generator
    generator = UNetGenerator(
        input_nc=config.input_nc,
        output_nc=config.output_nc,
        ngf=config.ngf
    )
    
    # Initialize weights
    init_weights(generator)
    
    # Save as pretrained model
    checkpoint_path = config.pretrained_model_path
    save_checkpoint({
        'epoch': 0,
        'model_state_dict': generator.state_dict(),
        'config': config.__dict__,
    }, checkpoint_path)
    
    print(f"Pretrained model created at {checkpoint_path}")
    print("Note: This is a randomly initialized model.")
    print("For best results, train the model using train.py first.")


if __name__ == '__main__':
    create_pretrained_model()
