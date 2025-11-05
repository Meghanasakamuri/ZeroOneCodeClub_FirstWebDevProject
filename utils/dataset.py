"""
Dataset utilities for loading paired sketch-painting images
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class SketchPaintingDataset(Dataset):
    def __init__(self, sketch_dir, painting_dir, transform=None, mode='train'):
        """
        Args:
            sketch_dir: Directory containing grayscale sketch images
            painting_dir: Directory containing colored painting images
            transform: Optional transform to apply
            mode: 'train' or 'test'
        """
        self.sketch_dir = sketch_dir
        self.painting_dir = painting_dir
        self.transform = transform
        self.mode = mode
        
        # Get list of image files
        self.sketch_files = sorted([f for f in os.listdir(sketch_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.painting_files = sorted([f for f in os.listdir(painting_dir) 
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Ensure same number of files
        assert len(self.sketch_files) == len(self.painting_files), \
            "Number of sketch and painting images must match"
        
    def __len__(self):
        return len(self.sketch_files)
    
    def __getitem__(self, idx):
        sketch_path = os.path.join(self.sketch_dir, self.sketch_files[idx])
        painting_path = os.path.join(self.painting_dir, self.painting_files[idx])
        
        # Load images
        sketch = Image.open(sketch_path).convert('L')  # Convert to grayscale
        painting = Image.open(painting_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            sketch = self.transform(sketch)
            painting = self.transform(painting)
        else:
            # Default transforms
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
            sketch = transform(sketch)
            # For painting, normalize to [-1, 1]
            transform_painting = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            painting = transform_painting(painting)
        
        # Normalize sketch to [-1, 1]
        sketch = (sketch - 0.5) / 0.5
        
        return sketch, painting


def get_dataloader(sketch_dir, painting_dir, batch_size=1, num_workers=4, 
                   image_size=256, mode='train'):
    """
    Create DataLoader for sketch-painting dataset
    """
    # Create custom transforms
    transform_sketch = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    transform_painting = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create dataset with proper transforms
    class TransformedDataset(SketchPaintingDataset):
        def __getitem__(self, idx):
            sketch_path = os.path.join(self.sketch_dir, self.sketch_files[idx])
            painting_path = os.path.join(self.painting_dir, self.painting_files[idx])
            
            sketch = Image.open(sketch_path).convert('L')
            painting = Image.open(painting_path).convert('RGB')
            
            # Apply transforms
            sketch = transform_sketch(sketch)
            sketch = (sketch - 0.5) / 0.5  # Normalize to [-1, 1]
            painting = transform_painting(painting)
            
            return sketch, painting
    
    dataset = TransformedDataset(
        sketch_dir=sketch_dir,
        painting_dir=painting_dir,
        mode=mode
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def download_sample_dataset():
    """
    Downloads a sample dataset or creates directories structure
    For actual training, you should use a proper dataset like:
    - Edges2Shoes, Edges2Handbags (from Pix2Pix paper)
    - Custom sketch-painting pairs
    """
    print("For training, please prepare paired sketch-painting images.")
    print("Place grayscale sketches in 'data/train/sketches'")
    print("Place colored paintings in 'data/train/paintings'")
    print("\nYou can use datasets like:")
    print("- https://github.com/phillipi/pix2pix")
    print("- Custom sketch-painting pairs")
