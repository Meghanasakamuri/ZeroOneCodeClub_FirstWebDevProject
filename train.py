"""Train the AutoPainter Pix2Pix model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

from autopainter.config import AutoPainterConfig, DEFAULT_CONFIG, DatasetConfig, LossConfig, ModelConfig, OptimizerConfig, TrainingConfig, InferenceConfig
from autopainter.pipeline import AutoPainterSystem, build_dataloaders
from autopainter.utils import configure_logging


def load_config(config_path: Path) -> Dict[str, Any]:
    if config_path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("pyyaml is required to load YAML configurations")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_dataclass(dc, values: Dict[str, Any]):
    for key, value in values.items():
        if hasattr(dc, key):
            setattr(dc, key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AutoPainter Pix2Pix model")
    parser.add_argument("--config", type=str, help="Path to JSON/YAML config", default=None)
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated pix2pix dataset names (e.g. monet2photo,vangogh2photo)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lambda-perceptual", type=float, default=None, help="Weight for perceptual loss")
    parser.add_argument("--gan-mode", type=str, default=None, choices=["lsgan", "vanilla", "hinge"], help="GAN loss mode")
    parser.add_argument("--discriminator", type=str, default=None, choices=["patch", "pixel"], help="Discriminator type")
    parser.add_argument("--pretrained-generator", type=str, default=None, help="Path to pretrained generator weights")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu or cuda)")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    config = AutoPainterConfig(
        dataset=DatasetConfig(**vars(DEFAULT_CONFIG.dataset)),
        model=ModelConfig(**vars(DEFAULT_CONFIG.model)),
        losses=LossConfig(**vars(DEFAULT_CONFIG.losses)),
        optimizer=OptimizerConfig(**vars(DEFAULT_CONFIG.optimizer)),
        training=TrainingConfig(**vars(DEFAULT_CONFIG.training)),
        inference=InferenceConfig(**vars(DEFAULT_CONFIG.inference)),
    )

    if args.config:
        cfg_dict = load_config(Path(args.config))
        for section, values in cfg_dict.items():
            if hasattr(config, section):
                update_dataclass(getattr(config, section), values)

    if args.datasets:
        config.dataset.source = args.datasets
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lambda_perceptual is not None:
        config.losses.lambda_perceptual = args.lambda_perceptual
    if args.gan_mode:
        config.losses.gan_mode = args.gan_mode
    if args.discriminator:
        config.model.discriminator_type = args.discriminator
    if args.resume:
        config.training.resume_from = args.resume
    if args.device:
        config.inference.device = args.device

    dataloaders = build_dataloaders(config)
    autopainter = AutoPainterSystem(config, device=args.device)

    if args.pretrained_generator:
        autopainter.load_pretrained_generator(Path(args.pretrained_generator))

    autopainter.train(dataloaders["train"], dataloaders.get("val"))


if __name__ == "__main__":
    main()

