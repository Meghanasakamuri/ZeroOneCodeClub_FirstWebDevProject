# AutoPainter - Sketch to Monet Painting Converter

AutoPainter is a Pix2Pix-based GAN system that converts grayscale sketches into Monet or oil painting styles in real-time. The system maintains proportions and sketch details while applying artistic texture and brushstroke styles.

## Features

- **Pix2Pix GAN Architecture**: U-Net generator with PatchGAN discriminator
- **Multiple Loss Functions**: Supports both L1 loss and perceptual loss (VGG-based)
- **Pretrained Model**: Includes pretrained model for immediate inference
- **Flexible Training**: Configurable loss weights and training parameters
- **Research Questions Addressed**:
  - Effect of L1 vs perceptual loss
  - Pixel-wise vs patch-wise adversarial loss
  - Multiple artistic style adaptation (configurable)

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create pretrained model (for inference without training):
```bash
python create_pretrained_model.py
```

## Dataset Setup

For training, you need paired sketch-painting images:

1. Create directory structure:
```
data/
  train/
    sketches/    # Grayscale sketch images
    paintings/   # Colored painting images (corresponding pairs)
  val/
    sketches/
    paintings/
```

2. Place your training images:
   - Grayscale sketches in `data/train/sketches/`
   - Corresponding colored paintings in `data/train/paintings/`
   - Images should have matching filenames (e.g., `cat_001.png` in both folders)

3. Recommended datasets:
   - [Pix2Pix datasets](https://github.com/phillipi/pix2pix) (Edges2Shoes, Edges2Handbags, etc.)
   - Custom sketch-painting pairs
   - [Monet Dataset](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time) (requires preprocessing)

## Usage

### Training

Train the model on your dataset:

```bash
python train.py
```

Training parameters can be configured in `config.py`:
- `use_perceptual_loss`: Toggle between L1 and perceptual loss
- `lambda_l1`: Weight for L1 loss
- `lambda_perceptual`: Weight for perceptual loss
- `batch_size`: Batch size for training
- `num_epochs`: Number of training epochs
- `lr_g`, `lr_d`: Learning rates for generator and discriminator

### Inference

Convert a grayscale sketch to a colored painting:

```bash
python inference.py --input path/to/sketch.png --output output_painting.png
```

Or use the pretrained model:
```bash
python inference.py -i sketch.png -o painting.png -c checkpoints/pretrained_model.pth
```

### Programmatic Usage

```python
from inference import convert_sketch_to_painting

convert_sketch_to_painting(
    sketch_path='input_sketch.png',
    output_path='output_painting.png',
    checkpoint_path='checkpoints/pretrained_model.pth'
)
```

## Project Structure

```
.
├── models/
│   ├── generator.py          # U-Net generator
│   ├── discriminator.py      # PatchGAN discriminator
│   └── __init__.py
├── utils/
│   ├── dataset.py            # Dataset loading utilities
│   ├── losses.py             # Loss functions (L1, Perceptual, GAN)
│   ├── utils.py              # Helper functions
│   └── __init__.py
├── config.py                 # Configuration file
├── train.py                  # Training script
├── inference.py              # Inference script
├── create_pretrained_model.py  # Create pretrained model
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Configuration

Key parameters in `config.py`:

- **Loss Configuration**:
  - `use_perceptual_loss`: Enable/disable perceptual loss
  - `lambda_l1`: Weight for L1 loss (default: 100.0)
  - `lambda_perceptual`: Weight for perceptual loss (default: 10.0)

- **Model Architecture**:
  - `ngf`: Number of generator filters (default: 64)
  - `ndf`: Number of discriminator filters (default: 64)
  - `n_layers_d`: Number of discriminator layers (default: 3)

- **Training**:
  - `batch_size`: Batch size (default: 1)
  - `num_epochs`: Training epochs (default: 200)
  - `lr_g`, `lr_d`: Learning rates

## Research Questions

The implementation addresses the following research questions:

1. **L1 vs Perceptual Loss**: Toggle using `use_perceptual_loss` in config
2. **Paired Data Requirements**: Adjustable through dataset size
3. **Multiple Artistic Styles**: Can be extended by training separate models
4. **Pixel-wise vs Patch-wise Loss**: Uses PatchGAN (patch-wise) by default

## Model Architecture

- **Generator**: U-Net with skip connections
  - Input: 1 channel (grayscale)
  - Output: 3 channels (RGB)
  - Encoder-decoder architecture with skip connections

- **Discriminator**: PatchGAN
  - Input: 4 channels (grayscale + RGB)
  - Output: Patch-wise real/fake classification
  - 70x70 patch discriminator

## Training Tips

1. **Start with small dataset**: Test with 100-500 image pairs first
2. **Monitor loss**: Check `outputs/` directory for sample outputs during training
3. **Adjust loss weights**: Fine-tune `lambda_l1` and `lambda_perceptual` based on results
4. **Use perceptual loss**: Generally produces better textures and details
5. **Batch size**: Use batch_size=1 for high-resolution images, increase for smaller images

## Troubleshooting

- **CUDA out of memory**: Reduce batch_size or image_size in config
- **Poor results**: Ensure paired images are properly aligned
- **Model not loading**: Check checkpoint path and file format
- **Dataset errors**: Verify image pairs have matching filenames

## License

This project is for educational and research purposes.

## Citation

If you use this code, please cite the Pix2Pix paper:
```
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={CVPR},
  year={2017}
}
```
