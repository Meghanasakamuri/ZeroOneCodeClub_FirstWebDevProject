"""Utility helpers for AutoPainter."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import requests
import torch


logger = logging.getLogger("autopainter")


def configure_logging(level: int = logging.INFO) -> None:
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)


def download_file(url: str, target_path: Path, chunk_size: int = 1 << 20) -> None:
    configure_logging()
    logger.info("Downloading %s", url)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)

    logger.info("Saved to %s", target_path)


def log_tensor_stats(name: str, tensor: torch.Tensor) -> None:
    configure_logging()
    tensor = tensor.detach()
    logger.debug(
        "%s stats -> shape=%s mean=%.4f std=%.4f min=%.4f max=%.4f",
        name,
        tuple(tensor.shape),
        tensor.mean().item(),
        tensor.std().item(),
        tensor.min().item(),
        tensor.max().item(),
    )


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def style_one_hot(style_indices: torch.Tensor, num_styles: int) -> torch.Tensor:
    one_hot = torch.zeros(style_indices.size(0), num_styles, device=style_indices.device)
    one_hot.scatter_(1, style_indices.view(-1, 1), 1.0)
    return one_hot


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()

