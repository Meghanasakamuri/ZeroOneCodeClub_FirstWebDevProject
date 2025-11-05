## AutoPainter Research Guide

This document outlines how to investigate the key research questions raised for AutoPainter.

### 1. Effect of L1 vs Perceptual Loss
- Toggle `loss.lambda_perceptual` between `0` (pure L1) and `10` (default hybrid) in `configs/monet.yaml`.
- Train for at least 50 epochs each; log TensorBoard scalars `train/g_recon` and `train/g_perc`.
- Capture validation samples (`runs/.../samples`) and compute LPIPS or SSIM for quantitative comparison (external script suggested).

### 2. Paired Data Requirements
- Use `download_assets.py --val-split` plus manual sub-sampling to create 5%, 25%, 50%, and 100% training subsets.
- Keep hyperparameters fixed; track FID/LPIPS (external) and convergence speed.
- Analyse overfitting risk via validation recon loss curves.

### 3. Dynamic Multi-Style Adaptation
- Download multiple styles: `python download_assets.py --styles monet cezanne vangogh`.
- Update `configs/monet.yaml` -> `data.style_labels` to include the new styles.
- During training, the generator receives a style embedding; evaluate style-specific outputs by sweeping `infer.py --style <index>`.
- Measure style confusion by computing Gram matrix distances or CLIP similarity between generated images and style exemplars.

### 4. Pixel vs Patch Adversarial Loss
- Switch `loss.adversarial_mode` between `patch` and `pixel`.
- Observe discriminator stability (loss oscillations) and output sharpness.
- Record validation metrics and visual sharpness; patch mode typically yields sharper textures, pixel mode stabilises smaller datasets.

### Suggested Evaluation Metrics
- **LPIPS** for perceptual consistency
- **SSIM / PSNR** for structure retention
- **FID** against ground-truth paintings for realism

Use the stored checkpoints and TensorBoard logs to assemble comparative tables and qualitative grids.
