"""
Dataset handling for AutoPainter
Loads paired sketch-painting images for training
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from config import Config


class SketchToPaintingDataset(Dataset):
    """
    Dataset for paired sketch-to-painting images
    Supports grayscale sketch input and RGB painting output
    """
    def __init__(self, root_dir, mode="train", max_size=None):
        """
        Args:
            root_dir: Root directory containing 'sketches' and 'paintings' folders
            mode: 'train' or 'val'
            max_size: Maximum number of samples to use (for data ablation studies)
        """
        self.root_dir = root_dir
        self.mode = mode
        self.sketch_dir = os.path.join(root_dir, mode, "sketches")
        self.painting_dir = os.path.join(root_dir, mode, "paintings")
        
        # Get list of files
        if os.path.exists(self.sketch_dir):
            self.image_files = sorted(os.listdir(self.sketch_dir))
        else:
            print(f"Warning: {self.sketch_dir} does not exist. Using empty dataset.")
            self.image_files = []
        
        # Limit dataset size for experiments
        if max_size is not None and max_size < len(self.image_files):
            self.image_files = self.image_files[:max_size]
        
        # Transforms
        if mode == "train" and Config.USE_AUGMENTATION:
            self.transform_sketch = transforms.Compose([
                transforms.Resize((Config.LOAD_SIZE, Config.LOAD_SIZE), Image.BICUBIC),
                transforms.RandomCrop(Config.IMAGE_SIZE),
                transforms.RandomHorizontalFlip(p=0.5 if Config.HORIZONTAL_FLIP else 0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale
            ])
            
            self.transform_painting = transforms.Compose([
                transforms.Resize((Config.LOAD_SIZE, Config.LOAD_SIZE), Image.BICUBIC),
                transforms.RandomCrop(Config.IMAGE_SIZE),
                transforms.RandomHorizontalFlip(p=0.5 if Config.HORIZONTAL_FLIP else 0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # RGB
            ])
        else:
            self.transform_sketch = transforms.Compose([
                transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            
            self.transform_painting = transforms.Compose([
                transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load sketch (grayscale)
        sketch_path = os.path.join(self.sketch_dir, self.image_files[idx])
        sketch = Image.open(sketch_path).convert('L')  # Convert to grayscale
        
        # Load painting (RGB)
        painting_path = os.path.join(self.painting_dir, self.image_files[idx])
        painting = Image.open(painting_path).convert('RGB')
        
        # Apply same random seed for paired augmentation
        if self.mode == "train" and Config.USE_AUGMENTATION:
            seed = np.random.randint(2147483647)
            
            torch.manual_seed(seed)
            sketch = self.transform_sketch(sketch)
            
            torch.manual_seed(seed)
            painting = self.transform_painting(painting)
        else:
            sketch = self.transform_sketch(sketch)
            painting = self.transform_painting(painting)
        
        return sketch, painting


class CombinedImageDataset(Dataset):
    """
    Alternative dataset format where sketch and painting are combined side-by-side
    Common format for Pix2Pix datasets (e.g., edges2shoes, facades)
    """
    def __init__(self, root_dir, mode="train", max_size=None):
        self.root_dir = root_dir
        self.mode = mode
        self.image_dir = os.path.join(root_dir, mode)
        
        if os.path.exists(self.image_dir):
            self.image_files = sorted([f for f in os.listdir(self.image_dir) 
                                      if f.endswith(('.jpg', '.png', '.jpeg'))])
        else:
            print(f"Warning: {self.image_dir} does not exist. Using empty dataset.")
            self.image_files = []
        
        if max_size is not None and max_size < len(self.image_files):
            self.image_files = self.image_files[:max_size]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        w, h = image.size
        w_half = w // 2
        
        # Split image: left half is painting (target), right half is sketch (input)
        painting = image.crop((0, 0, w_half, h))
        sketch = image.crop((w_half, 0, w, h))
        
        # Convert sketch to grayscale
        sketch = sketch.convert('L')
        
        # Apply transforms
        if self.mode == "train" and Config.USE_AUGMENTATION:
            transform_sketch = transforms.Compose([
                transforms.Resize((Config.LOAD_SIZE, Config.LOAD_SIZE), Image.BICUBIC),
                transforms.RandomCrop(Config.IMAGE_SIZE),
                transforms.RandomHorizontalFlip(p=0.5 if Config.HORIZONTAL_FLIP else 0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            
            transform_painting = transforms.Compose([
                transforms.Resize((Config.LOAD_SIZE, Config.LOAD_SIZE), Image.BICUBIC),
                transforms.RandomCrop(Config.IMAGE_SIZE),
                transforms.RandomHorizontalFlip(p=0.5 if Config.HORIZONTAL_FLIP else 0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            # Same seed for paired augmentation
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            sketch = transform_sketch(sketch)
            torch.manual_seed(seed)
            painting = transform_painting(painting)
        else:
            sketch = transforms.Compose([
                transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])(sketch)
            
            painting = transforms.Compose([
                transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])(painting)
        
        return sketch, painting


def get_dataloaders(dataset_type="separate"):
    """
    Create train and validation dataloaders
    
    Args:
        dataset_type: "separate" for separate sketch/painting folders,
                     "combined" for side-by-side images
    """
    if dataset_type == "separate":
        train_dataset = SketchToPaintingDataset(
            Config.DATA_DIR, 
            mode="train", 
            max_size=Config.TRAIN_SIZE
        )
        val_dataset = SketchToPaintingDataset(
            Config.DATA_DIR, 
            mode="val", 
            max_size=None
        )
    else:
        train_dataset = CombinedImageDataset(
            Config.DATA_DIR, 
            mode="train", 
            max_size=Config.TRAIN_SIZE
        )
        val_dataset = CombinedImageDataset(
            Config.DATA_DIR, 
            mode="val", 
            max_size=None
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing dataset loading...")
    print(f"Data directory: {Config.DATA_DIR}")
    
    try:
        train_loader, val_loader = get_dataloaders(dataset_type="separate")
        print(f"Train dataset size: {len(train_loader.dataset)}")
        print(f"Val dataset size: {len(val_loader.dataset)}")
        
        if len(train_loader.dataset) > 0:
            sketch, painting = next(iter(train_loader))
            print(f"Sketch shape: {sketch.shape} (min: {sketch.min():.2f}, max: {sketch.max():.2f})")
            print(f"Painting shape: {painting.shape} (min: {painting.min():.2f}, max: {painting.max():.2f})")
    except Exception as e:
        print(f"Dataset test failed (expected if data not downloaded yet): {e}")
