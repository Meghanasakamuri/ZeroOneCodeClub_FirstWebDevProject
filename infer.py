"""Run inference with a trained AutoPainter model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

from PIL import Image
import torch
from torchvision import transforms

from autopainter.config import AutoPainterConfig, DEFAULT_CONFIG, DatasetConfig, LossConfig, ModelConfig, OptimizerConfig, TrainingConfig, InferenceConfig
from autopainter.pipeline import AutoPainterSystem
from autopainter.utils import configure_logging


def load_config(config_path: Path) -> Dict[str, Any]:
    if config_path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("pyyaml is required to load YAML configurations")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoPainter inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input grayscale sketch")
    parser.add_argument("--output", type=str, required=True, help="Output path for the colored painting")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint (.pt)")
    parser.add_argument("--config", type=str, default=None, help="Optional config override file")
    parser.add_argument("--style", type=str, default="monet", help="Style name to render")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu or cuda)")
    return parser.parse_args()


def update_dataclass(dc, values: Dict[str, Any]) -> None:
    for key, value in values.items():
        if hasattr(dc, key):
            setattr(dc, key, value)


def load_image(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("L")
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return transform(image)


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

    if args.device:
        config.inference.device = args.device

    checkpoint = Path(args.checkpoint)
    state_dict = torch.load(checkpoint, map_location="cpu")
    if "config" in state_dict:
        for section, values in state_dict["config"].items():
            if hasattr(config, section):
                update_dataclass(getattr(config, section), values)

    autopainter = AutoPainterSystem(config, device=args.device)
    if "netG" in state_dict:
        autopainter.netG.load_state_dict(state_dict["netG"], strict=False)
    else:
        autopainter.netG.load_state_dict(state_dict, strict=False)

    style_mapping = config.dataset.style_mapping
    if args.style not in style_mapping:
        available = ", ".join(style_mapping.keys()) or "(none)"
        raise ValueError(f"Style '{args.style}' not available in checkpoint. Available styles: {available}")
    style_index = style_mapping[args.style]

    sketch_tensor = load_image(Path(args.input), config.dataset.image_size)
    output = autopainter.inference(sketch_tensor, style_index=style_index)
    out_image = (output * 0.5 + 0.5).clamp(0, 1)
    save_path = Path(args.output)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    transforms.ToPILImage()(out_image).save(save_path)
    print(f"Saved output to {save_path}")


if __name__ == "__main__":
    main()

