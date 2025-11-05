# AutoPainter – Pix2Pix Sketch → Monet Painter

AutoPainter is a complete Pix2Pix-based GAN pipeline that converts grayscale sketches into richly coloured Monet-style (or other oil-painting) renderings in real time. The project includes:

- Data tooling to synthesise paired sketch/painting samples from Monet, Cézanne, and Van Gogh collections
- A configurable UNet generator + Patch / Pixel GAN discriminator with multi-style conditioning
- Support for L1 vs perceptual loss, pixel-wise vs patch-wise adversaries, and feature-matching
- Training, evaluation, and TensorBoard logging utilities
- Ready-to-use pretrained Monet weights and inference script for instant sketch colourisation


## 1. Environment Setup

- Python 3.10+
- GPU with CUDA (recommended for full training speed)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## 2. Assets: Datasets & Pretrained Weights

Run the helper script to download paired painting datasets, create processed splits, and fetch the pretrained Monet generator weights.

```bash
python download_assets.py --styles monet cezanne vangogh
```

Outputs:

- `data/raw/<style>2photo/` – original CycleGAN datasets
- `data/processed/monet_sketches/<style>/{train,val,test}` – curated RGB paintings (sketches generated on-the-fly)
- `checkpoints/autopainter_<style>/autopainter_<style>_generator.pt` – pretrained generator weights (Monet weights hosted on Hugging Face)

> **Note:** If the pretrained download fails (e.g., air-gapped environment), you can still train from scratch with `train.py`. After convergence, point `infer.py` at your latest checkpoint.


## 3. Training

The default config (`configs/monet.yaml`) enables multi-style conditioning, PatchGAN, perceptual loss, and feature matching.

```bash
python train.py --config configs/monet.yaml
```

Artifacts:

- Checkpoints: `checkpoints/autopainter_monet/epoch_XXXX.pt`
- TensorBoard logs & sample grids: `runs/autopainter/autopainter_monet/`

Resume training with `python train.py --config configs/monet.yaml --resume checkpoints/autopainter_monet/epoch_00XX.pt`.


### Research Toggles

- **L1 vs Perceptual Loss:** set `loss.lambda_perceptual` to `0` (pure L1) or keep default (hybrid). Compare validation L1 in TensorBoard.
- **Data Efficiency:** limit training images by temporarily slicing `train_dir` or adjusting `download_assets.py --val-split` to simulate low-data scenarios; monitor FID/LPIPS (can be computed with external tooling) or qualitative samples.
- **Dynamic Multi-Style:** include extra styles via `download_assets.py --styles monet cezanne` and list them in `configs/monet.yaml -> data.style_labels`. Choose style at inference time with `infer.py --style <index>`.
- **Patch vs Pixel GAN:** switch `loss.adversarial_mode` between `patch` and `pixel` to measure impact on sharpness vs stability.


## 4. Inference

Colourise any grayscale sketch with the pretrained Monet generator (or your own checkpoint).

```bash
python infer.py \
  path/to/sketch.png \
  checkpoints/autopainter_monet/autopainter_monet_generator.pt \
  outputs/monet_cat.png \
  --style 0 \
  --image-size 256
```

Key options:

- `--config`: override config if you trained a custom style layout
- `--style`: integer index into `data.style_labels` for multi-style models
- `--device`: `cuda`, `cpu`, or specific GPU id


## 5. Project Structure

```
autopainter/
  ├── config.py              # dataclass configs + YAML loader
  ├── data/                  # dataset + dataloader wrappers
  ├── models/                # UNet generator, discriminator, losses
  └── trainer.py             # AMP training loop, logging, checkpoints
configs/
  └── monet.yaml             # reference experiment config
download_assets.py           # dataset + pretrained bootstrapper
train.py                     # training entrypoint
infer.py                     # inference utility
requirements.txt             # dependencies
```


## 6. Extending the System

- **Higher Resolution:** increase `data.image_size` and adjust `generator_channels`/`discriminator_channels` (consider 2× gradient accumulation).
- **Style Interpolation:** expose generator style embeddings (latent vectors) for blending multiple artistic modes.
- **Quantitative Evaluation:** integrate FID/LPIPS scripts for rigorous comparisons when toggling loss configurations.
- **Real-time Deployment:** trace `infer.py`’s generator with TorchScript or export to ONNX, then serve via Triton or TensorRT.


## 7. Troubleshooting

- *CUDA OOM:* reduce `data.batch_size` or disable perceptual loss.
- *Dataset missing:* rerun `download_assets.py`; ensure directories exist under `data/raw/` and `data/processed/`.
- *Weights download blocked:* use `--skip-pretrained` and train to generate your own checkpoints; update `infer.py` call accordingly.


Happy painting! 🎨