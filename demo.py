"""
Demo script for AutoPainter
Creates a sample grayscale image and converts it to a painting
"""
import torch
import numpy as np
from PIL import Image, ImageDraw
import os

from inference import convert_sketch_to_painting
from create_pretrained_model import create_pretrained_model


def create_sample_sketch(output_path='sample_sketch.png'):
    """Create a sample grayscale sketch for testing"""
    # Create a simple cat sketch
    img = Image.new('L', (256, 256), color=255)  # White background
    draw = ImageDraw.Draw(img)
    
    # Draw a simple cat outline
    # Head
    draw.ellipse([80, 60, 176, 130], outline=0, width=2)
    
    # Ears
    draw.polygon([(100, 70), (110, 50), (120, 70)], outline=0, fill=0)
    draw.polygon([(136, 70), (146, 50), (156, 70)], outline=0, fill=0)
    
    # Eyes
    draw.ellipse([110, 85, 125, 100], outline=0, fill=0)
    draw.ellipse([131, 85, 146, 100], outline=0, fill=0)
    
    # Nose
    draw.polygon([(128, 105), (135, 115), (121, 115)], outline=0, fill=0)
    
    # Mouth
    draw.line([(128, 115), (128, 125)], fill=0, width=2)
    draw.line([(128, 120), (118, 125)], fill=0, width=2)
    draw.line([(128, 120), (138, 125)], fill=0, width=2)
    
    # Body
    draw.ellipse([90, 130, 166, 200], outline=0, width=2)
    
    # Legs
    draw.rectangle([100, 200, 110, 230], outline=0, fill=0)
    draw.rectangle([146, 200, 156, 230], outline=0, fill=0)
    
    # Tail
    draw.arc([166, 160, 200, 200], start=0, end=90, fill=0, width=3)
    
    img.save(output_path)
    print(f"Sample sketch created: {output_path}")
    return output_path


def run_demo():
    """Run a complete demo"""
    print("=" * 60)
    print("AutoPainter Demo")
    print("=" * 60)
    
    # Check if pretrained model exists
    from config import Config
    config = Config()
    
    if not os.path.exists(config.pretrained_model_path):
        print("\nPretrained model not found. Creating one...")
        create_pretrained_model()
    
    # Create sample sketch
    print("\nCreating sample sketch...")
    sketch_path = create_sample_sketch()
    
    # Convert to painting
    print("\nConverting sketch to painting...")
    output_path = 'demo_output.png'
    
    try:
        convert_sketch_to_painting(
            sketch_path=sketch_path,
            output_path=output_path,
            checkpoint_path=config.pretrained_model_path
        )
        print(f"\nDemo completed! Check {output_path} for the result.")
        print("\nNote: The pretrained model uses random weights.")
        print("For better results, train the model first using train.py")
    except Exception as e:
        print(f"\nError during conversion: {e}")
        print("Make sure you have installed all dependencies:")
        print("  pip install -r requirements.txt")


if __name__ == '__main__':
    run_demo()
