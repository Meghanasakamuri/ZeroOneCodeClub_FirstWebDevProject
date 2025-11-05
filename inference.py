"""
Inference script for AutoPainter
Converts grayscale sketches to colored paintings
"""
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import argparse

from models import UNetGenerator
from utils.utils import load_checkpoint, tensor_to_image, save_image
from config import Config


def load_model(checkpoint_path, device):
    """Load pretrained generator model"""
    config = Config()
    
    generator = UNetGenerator(
        input_nc=config.input_nc,
        output_nc=config.output_nc,
        ngf=config.ngf
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['model_state_dict'])
        else:
            generator.load_state_dict(checkpoint)
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model")
    
    generator.eval()
    return generator


def preprocess_image(image_path, image_size=256):
    """Preprocess input grayscale image"""
    image = Image.open(image_path).convert('L')
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    tensor = transform(image)
    # Normalize to [-1, 1]
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0)  # Add batch dimension


def convert_sketch_to_painting(sketch_path, output_path, checkpoint_path=None, device=None):
    """
    Convert a grayscale sketch to a colored painting
    
    Args:
        sketch_path: Path to input grayscale sketch
        output_path: Path to save output painting
        checkpoint_path: Path to model checkpoint (default: pretrained_model.pth)
        device: Device to run on (default: auto-detect)
    """
    config = Config()
    
    if device is None:
        device = config.device
    
    if checkpoint_path is None:
        checkpoint_path = config.pretrained_model_path
    
    # Load model
    generator = load_model(checkpoint_path, device)
    
    # Preprocess input
    sketch_tensor = preprocess_image(sketch_path, config.image_size)
    sketch_tensor = sketch_tensor.to(device)
    
    # Generate painting
    with torch.no_grad():
        painting_tensor = generator(sketch_tensor)
    
    # Save output
    save_image(painting_tensor, output_path)
    print(f"Generated painting saved to {output_path}")
    
    return painting_tensor


def main():
    parser = argparse.ArgumentParser(description='Convert sketch to painting')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input grayscale sketch image')
    parser.add_argument('--output', '-o', type=str, default='output_painting.png',
                       help='Path to save output painting')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                       help='Path to model checkpoint (default: pretrained_model.pth)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, default: auto)')
    
    args = parser.parse_args()
    
    convert_sketch_to_painting(
        args.input,
        args.output,
        args.checkpoint,
        args.device
    )


if __name__ == '__main__':
    main()
