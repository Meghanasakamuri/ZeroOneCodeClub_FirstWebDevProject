"""
Inference script for AutoPainter
Convert grayscale sketches to Monet-style colored paintings
"""

import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from model import Generator


def load_model(checkpoint_path, device):
    """Load trained generator model"""
    gen = Generator(
        in_channels=Config.INPUT_CHANNELS,
        out_channels=Config.OUTPUT_CHANNELS,
        features=Config.GEN_FEATURES
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        gen.load_state_dict(checkpoint['generator_state_dict'])
        print(f"Model loaded from: {checkpoint_path}")
        print(f"Trained for {checkpoint['epoch']} epochs")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model (will produce random results)")
    
    gen.eval()
    return gen


def preprocess_image(image_path, size=256):
    """
    Preprocess input sketch image
    Converts to grayscale and normalizes
    """
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    
    transform = transforms.Compose([
        transforms.Resize((size, size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return transform(image).unsqueeze(0)  # Add batch dimension


def postprocess_image(tensor):
    """
    Postprocess output painting
    Denormalizes from [-1, 1] to [0, 1]
    """
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = tensor * 0.5 + 0.5  # Denormalize
    tensor = torch.clamp(tensor, 0, 1)
    return tensor


def sketch_to_painting(gen, sketch_path, output_path, device, show_result=True):
    """
    Convert a single sketch to painting
    
    Args:
        gen: Trained generator model
        sketch_path: Path to input sketch image
        output_path: Path to save output painting
        device: Device to run inference on
        show_result: Whether to display the result
    """
    # Load and preprocess sketch
    sketch = preprocess_image(sketch_path, Config.IMAGE_SIZE).to(device)
    
    # Generate painting
    with torch.no_grad():
        fake_painting = gen(sketch)
    
    # Postprocess
    sketch_vis = postprocess_image(sketch.cpu())
    painting_vis = postprocess_image(fake_painting.cpu())
    
    # Save output
    save_image(painting_vis, output_path)
    print(f"Painting saved to: {output_path}")
    
    # Display result
    if show_result:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Sketch
        axes[0].imshow(sketch_vis.squeeze(), cmap='gray')
        axes[0].set_title('Input Sketch')
        axes[0].axis('off')
        
        # Painting
        painting_np = painting_vis.permute(1, 2, 0).numpy()
        axes[1].imshow(painting_np)
        axes[1].set_title('Generated Monet Painting')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path.replace('.png', '_comparison.png'))
        plt.show()
    
    return painting_vis


def batch_inference(gen, input_dir, output_dir, device):
    """
    Process multiple sketches in a directory
    
    Args:
        gen: Trained generator model
        input_dir: Directory containing sketch images
        output_dir: Directory to save output paintings
        device: Device to run inference on
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    print(f"Processing {len(image_files)} images...")
    
    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, f"painting_{image_file}")
        
        try:
            sketch_to_painting(gen, input_path, output_path, device, show_result=False)
            print(f"✓ Processed: {image_file}")
        except Exception as e:
            print(f"✗ Failed to process {image_file}: {e}")
    
    print(f"\nBatch processing complete! Outputs saved in: {output_dir}")


def create_sample_sketch():
    """Create a simple sample sketch for testing"""
    from PIL import ImageDraw
    
    # Create a simple cat sketch
    img = Image.new('L', (256, 256), color=255)
    draw = ImageDraw.Draw(img)
    
    # Simple cat outline
    # Head (circle)
    draw.ellipse([80, 80, 176, 176], outline=0, width=3)
    
    # Ears
    draw.polygon([(90, 90), (80, 50), (110, 80)], outline=0, width=2)
    draw.polygon([(166, 90), (176, 50), (146, 80)], outline=0, width=2)
    
    # Eyes
    draw.ellipse([100, 110, 115, 125], outline=0, width=2)
    draw.ellipse([141, 110, 156, 125], outline=0, width=2)
    
    # Nose
    draw.polygon([(128, 135), (120, 145), (136, 145)], fill=0)
    
    # Mouth
    draw.arc([110, 140, 146, 160], 0, 180, fill=0, width=2)
    
    # Whiskers
    draw.line([(80, 130), (50, 125)], fill=0, width=2)
    draw.line([(80, 140), (50, 140)], fill=0, width=2)
    draw.line([(176, 130), (206, 125)], fill=0, width=2)
    draw.line([(176, 140), (206, 140)], fill=0, width=2)
    
    output_path = "sample_sketch.png"
    img.save(output_path)
    print(f"Sample cat sketch created: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='AutoPainter: Sketch to Painting Inference')
    parser.add_argument('--input', type=str, help='Input sketch image path or directory')
    parser.add_argument('--output', type=str, default='output_painting.png', 
                       help='Output painting path or directory')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--batch', action='store_true', 
                       help='Process all images in input directory')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create a sample sketch for testing')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). Auto-detect if not specified')
    
    args = parser.parse_args()
    
    # Determine device
    device = args.device if args.device else Config.DEVICE
    print(f"Using device: {device}")
    
    # Create sample sketch if requested
    if args.create_sample:
        sample_path = create_sample_sketch()
        if not args.input:
            args.input = sample_path
    
    # Check input
    if not args.input:
        print("Error: Please provide --input path or use --create-sample")
        return
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    gen = load_model(args.checkpoint, device)
    
    # Run inference
    if args.batch:
        # Batch processing
        if not os.path.isdir(args.input):
            print("Error: --batch mode requires input to be a directory")
            return
        batch_inference(gen, args.input, args.output, device)
    else:
        # Single image processing
        if not os.path.isfile(args.input):
            print(f"Error: Input file not found: {args.input}")
            return
        sketch_to_painting(gen, args.input, args.output, device, show_result=True)


if __name__ == "__main__":
    main()
