"""Download and adapt pretrained generators from Torch Hub."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch


TORCH_HUB_REPO = "junyanz/pytorch-CycleGAN-and-pix2pix"

MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    "edges2handbags": ("pix2pix", "edges2handbags"),
    "edges2shoes": ("pix2pix", "edges2shoes"),
    "facades": ("pix2pix", "facades_label2photo"),
    "maps": ("pix2pix", "maps_A2B"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch a pretrained generator from Torch Hub")
    parser.add_argument("--model", type=str, default="edges2handbags", choices=MODEL_REGISTRY.keys())
    parser.add_argument("--output", type=str, default="checkpoints/autopainter_pretrained.pt", help="Path to save the state_dict")
    parser.add_argument("--force", action="store_true", help="Overwrite existing file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        print(f"{output_path} already exists. Use --force to overwrite.")
        return

    api, model_name = MODEL_REGISTRY[args.model]
    print(f"Loading {api} model '{model_name}' from {TORCH_HUB_REPO}...")
    net = torch.hub.load(TORCH_HUB_REPO, api, model_name=model_name, pretrained=True)

    if hasattr(net, "eval"):
        net.eval()

    state_dict = net.state_dict()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_path)
    print(f"Saved generator weights to {output_path}")


if __name__ == "__main__":
    main()

