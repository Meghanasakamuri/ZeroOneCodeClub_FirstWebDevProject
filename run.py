#!/usr/bin/env python
"""
AutoPainter Quick Run Script

This is a simple helper script to run common tasks.
Perfect for getting started quickly!
"""

import os
import sys
import subprocess


def print_banner():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║                    🎨 AutoPainter 🎨                          ║
    ║            Sketch-to-Painting with Pix2Pix GAN               ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


def check_dependencies():
    """Check if required packages are installed"""
    try:
        import torch
        import torchvision
        import PIL
        import cv2
        print("✓ All dependencies installed")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease run: pip install -r requirements.txt")
        return False


def check_dataset():
    """Check if dataset exists"""
    data_dir = "data/monet_sketches/train/sketches"
    if os.path.exists(data_dir):
        num_files = len([f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        print(f"✓ Dataset found: {num_files} training sketches")
        return True
    else:
        print("✗ Dataset not found")
        print("\nPlease run: python download_dataset.py")
        return False


def setup():
    """Initial setup"""
    print("\n" + "="*70)
    print("SETUP")
    print("="*70)
    
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        return False
    
    print("\n2. Checking dataset...")
    has_data = check_dataset()
    
    if not has_data:
        print("\nWould you like to create a sample dataset now?")
        choice = input("This will create 100 synthetic images (y/n): ").strip().lower()
        if choice == 'y':
            print("\nCreating sample dataset...")
            result = subprocess.run([sys.executable, "download_dataset.py"], 
                                   input="3\n100\n", text=True, capture_output=True)
            if result.returncode == 0:
                print("✓ Sample dataset created!")
            else:
                print("✗ Failed to create dataset")
                print(result.stderr)
                return False
        else:
            print("\nPlease run 'python download_dataset.py' manually")
            return False
    
    print("\n" + "="*70)
    print("Setup complete! Ready to train or infer.")
    print("="*70)
    return True


def quick_train():
    """Quick training with reduced settings"""
    print("\n" + "="*70)
    print("QUICK TRAINING")
    print("="*70)
    print("\nThis will train for 10 epochs with a small dataset.")
    print("Perfect for testing the pipeline!")
    print("\nEstimated time:")
    print("  - GPU: ~5-10 minutes")
    print("  - CPU: ~1-2 hours")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        return
    
    # Create a quick config
    print("\nStarting training...")
    
    # Modify config for quick training
    quick_config = """
# Quick training configuration
import sys
sys.path.insert(0, '.')
from config import Config

Config.NUM_EPOCHS = 10
Config.BATCH_SIZE = 8
Config.TRAIN_SIZE = 50
Config.SAVE_INTERVAL = 5

print("Quick training mode activated!")
print(f"  Epochs: {Config.NUM_EPOCHS}")
print(f"  Batch size: {Config.BATCH_SIZE}")
print(f"  Training samples: {Config.TRAIN_SIZE}")
"""
    
    with open("quick_config.py", "w") as f:
        f.write(quick_config)
    
    # Run training
    result = subprocess.run([sys.executable, "-c", 
                           "exec(open('quick_config.py').read()); from train import main; main()"])
    
    if result.returncode == 0:
        print("\n✓ Quick training complete!")
        print("Check checkpoints/ for the trained model")
        print("Run inference with: python run.py --infer")
    else:
        print("\n✗ Training failed")


def full_train():
    """Full training with default settings"""
    print("\n" + "="*70)
    print("FULL TRAINING")
    print("="*70)
    print("\nThis will train with default settings (200 epochs).")
    print("\nEstimated time:")
    print("  - GPU (RTX 3060): ~2-4 hours")
    print("  - GPU (A100): ~30-60 minutes")
    print("  - CPU: Not recommended (20+ hours)")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        return
    
    print("\nStarting full training...")
    result = subprocess.run([sys.executable, "train.py"])
    
    if result.returncode == 0:
        print("\n✓ Training complete!")
        print("Check checkpoints/ for the trained model")
    else:
        print("\n✗ Training failed")


def infer():
    """Run inference"""
    print("\n" + "="*70)
    print("INFERENCE")
    print("="*70)
    
    # Check for checkpoint
    checkpoint_dir = "checkpoints"
    checkpoints = []
    
    if os.path.exists(checkpoint_dir):
        for root, dirs, files in os.walk(checkpoint_dir):
            checkpoints.extend([os.path.join(root, f) for f in files if f.endswith('.pth')])
    
    if not checkpoints:
        print("\n✗ No trained model found!")
        print("Please train a model first:")
        print("  - Quick train: python run.py --quick-train")
        print("  - Full train: python run.py --full-train")
        return
    
    print(f"\nFound {len(checkpoints)} checkpoint(s)")
    
    # Find best model
    best_model = None
    for cp in checkpoints:
        if "best_model" in cp:
            best_model = cp
            break
    
    if not best_model:
        best_model = checkpoints[0]
    
    print(f"Using checkpoint: {best_model}")
    
    # Options
    print("\nInference Options:")
    print("  1. Create sample sketch and convert")
    print("  2. Convert your own sketch")
    print("  3. Batch process folder")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        print("\nCreating sample sketch and generating painting...")
        result = subprocess.run([
            sys.executable, "inference.py",
            "--create-sample",
            "--checkpoint", best_model
        ])
    
    elif choice == "2":
        input_path = input("Enter sketch image path: ").strip()
        output_path = input("Enter output path (default: output_painting.png): ").strip() or "output_painting.png"
        
        print(f"\nGenerating painting from {input_path}...")
        result = subprocess.run([
            sys.executable, "inference.py",
            "--input", input_path,
            "--output", output_path,
            "--checkpoint", best_model
        ])
    
    elif choice == "3":
        input_dir = input("Enter input directory: ").strip()
        output_dir = input("Enter output directory: ").strip()
        
        print(f"\nBatch processing from {input_dir}...")
        result = subprocess.run([
            sys.executable, "inference.py",
            "--input", input_dir,
            "--output", output_dir,
            "--batch",
            "--checkpoint", best_model
        ])
    
    else:
        print("Invalid choice")


def run_experiments():
    """Run research experiments"""
    print("\n" + "="*70)
    print("RESEARCH EXPERIMENTS")
    print("="*70)
    
    subprocess.run([sys.executable, "experiments.py"])


def show_help():
    """Show help information"""
    print("\n" + "="*70)
    print("USAGE")
    print("="*70)
    print("""
Usage: python run.py [option]

Options:
  --setup          Initial setup (check dependencies, create dataset)
  --quick-train    Quick training (10 epochs, small dataset)
  --full-train     Full training (200 epochs, all data)
  --infer          Run inference on sketch images
  --experiments    Run research experiments
  --help           Show this help message
  
No options: Interactive menu

Examples:
  python run.py                    # Interactive menu
  python run.py --setup            # Check setup
  python run.py --quick-train      # Quick test
  python run.py --infer            # Generate paintings

For detailed documentation, see:
  - QUICKSTART.md
  - README_AUTOPAINTER.md
    """)


def main():
    """Main interactive menu"""
    print_banner()
    
    # Parse command line args
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ["--help", "-h"]:
            show_help()
        elif arg == "--setup":
            setup()
        elif arg == "--quick-train":
            if setup():
                quick_train()
        elif arg == "--full-train":
            if setup():
                full_train()
        elif arg == "--infer":
            infer()
        elif arg == "--experiments":
            run_experiments()
        else:
            print(f"Unknown option: {arg}")
            show_help()
        return
    
    # Interactive menu
    while True:
        print("\n" + "="*70)
        print("MAIN MENU")
        print("="*70)
        print("  1. Setup (check dependencies and dataset)")
        print("  2. Quick Train (10 epochs, ~5-10 min on GPU)")
        print("  3. Full Train (200 epochs, ~2-4 hours on GPU)")
        print("  4. Inference (generate paintings from sketches)")
        print("  5. Run Experiments (research questions)")
        print("  6. Help")
        print("  0. Exit")
        
        choice = input("\nEnter choice: ").strip()
        
        if choice == "1":
            setup()
        elif choice == "2":
            if setup():
                quick_train()
        elif choice == "3":
            if setup():
                full_train()
        elif choice == "4":
            infer()
        elif choice == "5":
            run_experiments()
        elif choice == "6":
            show_help()
        elif choice == "0":
            print("\nGoodbye! 👋")
            break
        else:
            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    main()
