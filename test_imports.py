"""
Test script to verify all imports work correctly
Run this before training to ensure everything is set up properly
"""
import sys

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    
    try:
        import torch
        import torchvision
        from PIL import Image
        import numpy as np
        print("✓ Basic imports successful")
    except ImportError as e:
        print(f"✗ Basic imports failed: {e}")
        return False
    
    try:
        from models import UNetGenerator, PatchGANDiscriminator
        print("✓ Model imports successful")
    except ImportError as e:
        print(f"✗ Model imports failed: {e}")
        return False
    
    try:
        from utils import SketchPaintingDataset, get_dataloader, PerceptualLoss
        from utils import save_checkpoint, load_checkpoint, tensor_to_image
        print("✓ Utility imports successful")
    except ImportError as e:
        print(f"✗ Utility imports failed: {e}")
        return False
    
    try:
        from config import Config
        print("✓ Config import successful")
    except ImportError as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    try:
        # Test model instantiation
        config = Config()
        generator = UNetGenerator(
            input_nc=config.input_nc,
            output_nc=config.output_nc,
            ngf=config.ngf
        )
        discriminator = PatchGANDiscriminator(
            input_nc=config.input_nc + config.output_nc,
            ndf=config.ndf,
            n_layers=config.n_layers_d
        )
        print("✓ Model instantiation successful")
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        return False
    
    print("\n" + "="*50)
    print("All tests passed! System is ready to use.")
    print("="*50)
    return True


if __name__ == '__main__':
    success = test_imports()
    sys.exit(0 if success else 1)
