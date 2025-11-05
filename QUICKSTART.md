# AutoPainter Quick Start Guide 🚀

Get your sketch-to-painting system running in 5 minutes!

## Installation (1 minute)

```bash
# Navigate to project
cd /workspace

# Install dependencies
pip install -r requirements.txt
```

## Prepare Data (2 minutes)

```bash
# Create sample dataset for quick testing
python download_dataset.py

# When prompted, select option 3 (synthetic dataset)
# This creates 100 training samples automatically
```

## Train Model (Choose One)

### Option A: Quick Test (5 minutes on GPU)

```bash
# Edit config.py - set these values:
# NUM_EPOCHS = 10
# BATCH_SIZE = 8
# TRAIN_SIZE = 50

python train.py
```

### Option B: Full Training (2-4 hours on GPU)

```bash
# Use default config.py settings
python train.py
```

### Option C: Use Pretrained Model

```bash
# If you have a pretrained checkpoint:
python inference.py --checkpoint path/to/model.pth --create-sample
```

## Generate Your First Painting (30 seconds)

```bash
# Option 1: Create sample sketch and convert
python inference.py --create-sample --checkpoint checkpoints/best_model.pth

# Option 2: Convert your own sketch
python inference.py --input my_sketch.jpg --output my_painting.png
```

## What You'll See

### During Training:
```
Epoch 5/10: 100%|████████| 10/10 [00:15<00:00]
  Gen Loss: 25.4231
  Disc Loss: 0.4521
  Val L1 Loss: 0.1234
✓ New best model saved!
```

### After Inference:
- `output_painting.png` - Your generated Monet-style painting
- `output_painting_comparison.png` - Side-by-side comparison

## Troubleshooting

### "CUDA out of memory"
```python
# Reduce batch size in config.py
Config.BATCH_SIZE = 4
```

### "No dataset found"
```bash
# Re-run dataset preparation
python download_dataset.py
```

### "Checkpoint not found"
```bash
# Check available checkpoints
ls checkpoints/*/
```

## Next Steps

1. **Improve Quality**: Train for more epochs (100-200)
2. **Try Different Losses**: Edit `Config.LOSS_TYPE` in config.py
3. **Use Real Data**: Download Monet dataset with download_dataset.py
4. **Run Experiments**: `python experiments.py`

## File Overview

- `train.py` - Train the model
- `inference.py` - Generate paintings
- `config.py` - All settings
- `download_dataset.py` - Get training data

## Quick Commands Reference

```bash
# Train
python train.py

# Generate from sketch
python inference.py --input sketch.jpg --output painting.png

# Create sample sketch
python inference.py --create-sample

# Batch process
python inference.py --input sketches/ --output paintings/ --batch

# Run experiments
python experiments.py
```

## Expected Results

### After 10 epochs (Quick Test):
- Basic color and structure
- Some artifacts
- Good for testing pipeline

### After 100 epochs (Full Training):
- Good artistic quality
- Monet-style brushstrokes
- Proper color palette

### After 200 epochs (Production):
- Excellent quality
- Minimal artifacts
- Strong artistic style

## Tips for Best Results

1. **Use GPU**: 100x faster than CPU
2. **More Data**: 500+ images recommended
3. **Combined Loss**: Best artistic quality
4. **Monitor Progress**: Check samples/ folder during training

---

**Ready to paint? Start with `python download_dataset.py`** 🎨
