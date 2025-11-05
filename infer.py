from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage

from autopainter.config import ExperimentConfig, load_config
from autopainter.models.generator import UNetGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoPainter inference script")
    parser.add_argument("input", type=Path, help="Path to grayscale sketch input image")
    parser.add_argument("checkpoint", type=Path, help="Path to generator checkpoint (.pt)")
    parser.add_argument("output", type=Path, help="Path to save the generated painting")
    parser.add_argument("--config", type=Path, default=None, help="Optional config YAML to override checkpoint config")
    parser.add_argument("--style", type=int, default=0, help="Style index to use (if multi-style)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=256, help="Resize input before inference")
    return parser.parse_args()


def load_image(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("L")
        img = img.resize((size, size), Image.BICUBIC)
        tensor = transforms.ToTensor()(img)  # 1 x H x W in [0,1]
        tensor = tensor * 2.0 - 1.0  # [-1,1]
        return tensor.unsqueeze(0)


def resolve_config(args: argparse.Namespace, checkpoint: dict) -> ExperimentConfig:
    if args.config:
        return load_config(args.config)

    cfg = checkpoint.get("config")
    if isinstance(cfg, ExperimentConfig):
        return cfg

    return load_config()


def build_generator(cfg: ExperimentConfig, state_dict: dict, device: torch.device) -> UNetGenerator:
    num_styles = len(cfg.data.style_labels) if cfg.multi_style else None
    generator = UNetGenerator(
        in_channels=1,
        out_channels=3,
        base_channels=cfg.generator_channels,
        num_styles=num_styles,
    ).to(device)
    generator.load_state_dict(state_dict, strict=True)
    generator.eval()
    return generator


def run_inference(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "generator" in checkpoint:
        state_dict = checkpoint["generator"]
    else:
        state_dict = checkpoint

    cfg = resolve_config(args, checkpoint)
    generator = build_generator(cfg, state_dict, device)

    sketch = load_image(args.input, args.image_size).to(device)
    with torch.no_grad():
        if cfg.multi_style:
            style = torch.tensor([args.style], device=device)
            output = generator(sketch, style)
        else:
            output = generator(sketch)

    output = ((output.clamp(-1, 1) + 1.0) / 2.0).cpu()
    to_pil = ToPILImage()
    image = to_pil(output.squeeze(0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Saved generated painting to {args.output}")


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()