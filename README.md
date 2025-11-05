# AutoPainter: Pix2Pix Sketch-to-Painting System

This repository contains **AutoPainter**, a Pix2Pix-based GAN pipeline that transforms grayscale sketches into Monet-style or oil-like colour paintings in real time. It includes:

- Modular PyTorch training pipeline with L1 and perceptual losses
- Dataset loader that auto-downloads paired painting datasets (e.g. `monet2photo`, `vangogh2photo`)
- Support for comparing pixel-wise vs. patch-wise adversarial objectives
- Scripts for full training and inference directly runnable from PyCharm or the terminal
- Utilities to download and adapt publicly available pretrained Pix2Pix models

## Quick Start

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Download a pretrained generator** (e.g. edges→handbags) for immediate testing:

   ```bash
   python autopainter/scripts/fetch_pretrained.py --model edges2handbags --output checkpoints/autopainter_pretrained_edges2handbags.pt
   ```

3. **Train on Monet-style textures** (auto-downloads datasets into `autopainter/data/`):

   ```bash
   python train.py --datasets monet2photo,vangogh2photo --epochs 200 --batch-size 4 --lambda-perceptual 10.0 --gan-mode lsgan
   ```

4. **Run inference on a grayscale sketch**:

   ```bash
   python infer.py --input path/to/cat_sketch.png --output outputs/cat_monet.png --checkpoint checkpoints/autopainter_epoch_200.pt
   ```

## Research Questions Addressed

- **Effect of L1 vs. perceptual loss:** toggle `--lambda-perceptual` to compare PSNR/SSIM during validation.
- **Data requirements:** adjust `--datasets` and `--epochs` to measure convergence vs. dataset size; the pipeline logs sample quality and L1 metrics.
- **Multi-style adaptation:** extend `dataset.style_mapping` in `config.py` to introduce additional styles on-the-fly (e.g. Monet, Van Gogh, Cezanne).
- **Pixel-wise vs. patch-wise GAN:** set `--discriminator pixel` to contrast against the default 70×70 PatchGAN discriminator.

Additional details and advanced configuration live in the in-code docstrings and scripts under `autopainter/`.