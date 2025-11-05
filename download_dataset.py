"""
Dataset preparation script for AutoPainter

This script downloads and prepares datasets for training:
1. Monet paintings dataset
2. Automatically creates sketch versions using edge detection

For production use, you can also use:
- Custom paired datasets
- Pre-existing Pix2Pix datasets (edges2shoes, facades, etc.)
"""

import os
import urllib.request
import zipfile
import shutil
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import gdown


def create_directory_structure(base_dir):
    """Create directory structure for dataset"""
    dirs = [
        os.path.join(base_dir, "train", "paintings"),
        os.path.join(base_dir, "train", "sketches"),
        os.path.join(base_dir, "val", "paintings"),
        os.path.join(base_dir, "val", "sketches"),
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    print(f"Created directory structure in {base_dir}")


def image_to_sketch(image_path, method='canny'):
    """
    Convert an image to a sketch using edge detection
    
    Args:
        image_path: Path to input image
        method: Edge detection method ('canny', 'hed', or 'contour')
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if method == 'canny':
        # Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Dilate edges slightly for better visibility
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Invert (white background, black lines)
        sketch = 255 - edges
    
    elif method == 'contour':
        # Enhanced contour detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on white background
        sketch = np.ones_like(gray) * 255
        cv2.drawContours(sketch, contours, -1, (0, 0, 0), 2)
    
    else:  # simplified method
        # Simple thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, sketch = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    return sketch


def download_monet_dataset():
    """
    Download Monet paintings dataset
    Uses a publicly available dataset
    """
    print("Downloading Monet paintings dataset...")
    
    # You can use various sources. Here's an example using Kaggle dataset
    # For this example, we'll provide instructions for manual download
    
    print("\n" + "="*70)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("\nOption 1: Use Monet2Photo Dataset (Recommended)")
    print("  1. Visit: https://www.kaggle.com/datasets/balraj98/monet2photo")
    print("  2. Download the dataset")
    print("  3. Extract to a 'downloads' folder")
    print("\nOption 2: Use your own paintings")
    print("  1. Place painting images in: data/monet_sketches/train/paintings/")
    print("  2. Place painting images in: data/monet_sketches/val/paintings/")
    print("  3. This script will auto-generate sketches")
    print("\nOption 3: Download sample dataset programmatically")
    print("  Using WikiArt or other public sources...")
    print("="*70)
    
    # Try to download a small sample dataset
    sample_url = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip"
    
    try:
        print(f"\nAttempting to download from: {sample_url}")
        download_path = "monet2photo.zip"
        
        # Download with progress bar
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded / total_size * 100, 100)
            print(f"\rDownloading: {percent:.1f}%", end='')
        
        urllib.request.urlretrieve(sample_url, download_path, show_progress)
        print("\n✓ Download complete!")
        
        # Extract
        print("Extracting dataset...")
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall("downloads")
        
        os.remove(download_path)
        print("✓ Extraction complete!")
        
        return "downloads/monet2photo"
        
    except Exception as e:
        print(f"\n✗ Automatic download failed: {e}")
        print("Please download manually using the instructions above.")
        return None


def process_monet_dataset(source_dir, target_dir, split_ratio=0.9):
    """
    Process Monet dataset to create sketch-painting pairs
    
    Args:
        source_dir: Directory containing original Monet paintings
        target_dir: Output directory for processed dataset
        split_ratio: Train/val split ratio
    """
    print(f"\nProcessing dataset from {source_dir}...")
    
    # Find all image files
    if os.path.exists(os.path.join(source_dir, "trainA")):
        # CycleGAN format
        source_paintings = os.path.join(source_dir, "trainA")
    elif os.path.exists(os.path.join(source_dir, "monet")):
        source_paintings = os.path.join(source_dir, "monet")
    else:
        source_paintings = source_dir
    
    if not os.path.exists(source_paintings):
        print(f"Error: Cannot find paintings in {source_dir}")
        return False
    
    image_files = [f for f in os.listdir(source_paintings) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print(f"Error: No images found in {source_paintings}")
        return False
    
    print(f"Found {len(image_files)} images")
    
    # Split into train/val
    split_idx = int(len(image_files) * split_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Process training set
    print("\nProcessing training set...")
    for filename in tqdm(train_files):
        src_path = os.path.join(source_paintings, filename)
        
        # Copy painting
        dst_painting = os.path.join(target_dir, "train", "paintings", filename)
        shutil.copy2(src_path, dst_painting)
        
        # Generate sketch
        sketch = image_to_sketch(src_path, method='canny')
        if sketch is not None:
            dst_sketch = os.path.join(target_dir, "train", "sketches", filename)
            cv2.imwrite(dst_sketch, sketch)
    
    # Process validation set
    print("Processing validation set...")
    for filename in tqdm(val_files):
        src_path = os.path.join(source_paintings, filename)
        
        # Copy painting
        dst_painting = os.path.join(target_dir, "val", "paintings", filename)
        shutil.copy2(src_path, dst_painting)
        
        # Generate sketch
        sketch = image_to_sketch(src_path, method='canny')
        if sketch is not None:
            dst_sketch = os.path.join(target_dir, "val", "sketches", filename)
            cv2.imwrite(dst_sketch, sketch)
    
    print(f"\n✓ Dataset processing complete!")
    print(f"  Training pairs: {len(train_files)}")
    print(f"  Validation pairs: {len(val_files)}")
    print(f"  Output directory: {target_dir}")
    
    return True


def create_sample_dataset(target_dir, num_samples=100):
    """
    Create a small synthetic dataset for testing
    Useful when you don't have real data yet
    """
    print(f"\nCreating synthetic sample dataset ({num_samples} samples)...")
    
    from PIL import ImageDraw
    import random
    
    for split in ['train', 'val']:
        n = num_samples if split == 'train' else num_samples // 10
        
        for i in tqdm(range(n), desc=f"Creating {split} set"):
            # Create random abstract painting
            img = Image.new('RGB', (256, 256), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # Random shapes with random colors
            for _ in range(random.randint(5, 15)):
                color = (random.randint(50, 255), 
                        random.randint(50, 255), 
                        random.randint(50, 255))
                
                shape_type = random.choice(['circle', 'rectangle', 'line'])
                
                if shape_type == 'circle':
                    x, y = random.randint(0, 200), random.randint(0, 200)
                    r = random.randint(10, 50)
                    draw.ellipse([x, y, x+r, y+r], fill=color, outline=color)
                
                elif shape_type == 'rectangle':
                    x1, y1 = random.randint(0, 200), random.randint(0, 200)
                    x2, y2 = x1 + random.randint(20, 80), y1 + random.randint(20, 80)
                    draw.rectangle([x1, y1, x2, y2], fill=color, outline=color)
                
                else:  # line
                    x1, y1 = random.randint(0, 256), random.randint(0, 256)
                    x2, y2 = random.randint(0, 256), random.randint(0, 256)
                    draw.line([x1, y1, x2, y2], fill=color, width=random.randint(2, 8))
            
            # Save painting
            painting_path = os.path.join(target_dir, split, "paintings", f"sample_{i:04d}.png")
            img.save(painting_path)
            
            # Create sketch
            sketch = image_to_sketch(painting_path, method='canny')
            sketch_path = os.path.join(target_dir, split, "sketches", f"sample_{i:04d}.png")
            cv2.imwrite(sketch_path, sketch)
    
    print("✓ Sample dataset created successfully!")


def main():
    """Main function for dataset preparation"""
    print("="*70)
    print("AutoPainter Dataset Preparation")
    print("="*70)
    
    target_dir = "data/monet_sketches"
    
    # Create directory structure
    create_directory_structure(target_dir)
    
    print("\nDataset Preparation Options:")
    print("1. Download Monet dataset (requires internet)")
    print("2. Process existing dataset from custom directory")
    print("3. Create synthetic sample dataset (for testing)")
    
    choice = input("\nEnter choice (1/2/3) or press Enter for option 3: ").strip() or "3"
    
    if choice == "1":
        # Download and process
        source_dir = download_monet_dataset()
        if source_dir and os.path.exists(source_dir):
            process_monet_dataset(source_dir, target_dir)
        else:
            print("\nFalling back to synthetic dataset...")
            create_sample_dataset(target_dir, num_samples=200)
    
    elif choice == "2":
        # Process existing directory
        source_dir = input("Enter path to directory containing paintings: ").strip()
        if os.path.exists(source_dir):
            process_monet_dataset(source_dir, target_dir)
        else:
            print(f"Error: Directory not found: {source_dir}")
    
    else:
        # Create synthetic dataset
        num_samples = input("Enter number of samples (default 100): ").strip()
        num_samples = int(num_samples) if num_samples else 100
        create_sample_dataset(target_dir, num_samples=num_samples)
    
    print("\n" + "="*70)
    print("Dataset Preparation Complete!")
    print("="*70)
    print(f"Dataset location: {target_dir}")
    print("\nYou can now run: python train.py")


if __name__ == "__main__":
    main()
