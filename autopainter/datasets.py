"""Dataset utilities for AutoPainter."""

from __future__ import annotations

import tarfile
from dataclasses import replace
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageFilter, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .config import DatasetConfig
from .utils import download_file, log_tensor_stats


PIX2PIX_DATASETS_BASE = (
    "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets"
)


def _ensure_dataset_available(name: str, target_root: Path) -> Path:
    """Download and extract a Pix2Pix dataset if missing."""

    dataset_dir = target_root / name
    if dataset_dir.exists():
        return dataset_dir

    target_root.mkdir(parents=True, exist_ok=True)

    archive_path = target_root / f"{name}.tar.gz"
    if not archive_path.exists():
        url = f"{PIX2PIX_DATASETS_BASE}/{name}.tar.gz"
        download_file(url, archive_path)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=target_root)

    return dataset_dir


class MonetSketchDataset(Dataset):
    """Create sketch-painting pairs from pix2pix-style datasets."""

    def __init__(
        self,
        cfg: DatasetConfig,
        split: str = "train",
        style_override: Optional[str] = None,
    ) -> None:
        self.cfg = replace(cfg, split=split)
        if not split.startswith("train"):
            self.cfg = replace(self.cfg, random_horizontal_flip=False, random_vertical_flip=False)
        self.split = split
        self.style_override = style_override

        root = Path("autopainter/data")
        sources = [s.strip() for s in cfg.source.split(",") if s.strip()]
        if not sources:
            raise ValueError("DatasetConfig.source must contain at least one dataset name")

        self.samples: List[Tuple[Path, int]] = []
        self.style_to_idx: Dict[str, int] = dict(cfg.style_mapping)

        for name in sources:
            ds_dir = _ensure_dataset_available(name, root)
            style_name = style_override or name.split("2")[0]
            style_idx = self.style_to_idx.setdefault(style_name, len(self.style_to_idx))

            domain = "trainA" if split.startswith("train") else "testA"
            if split.startswith("val"):
                domain = "testA"

            image_dir = ds_dir / domain
            if not image_dir.exists():
                raise FileNotFoundError(f"Expected {image_dir} for dataset {name}")

            paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
            if cfg.max_samples:
                paths = paths[: cfg.max_samples]
            for path in paths:
                self.samples.append((path, style_idx))

        if not self.samples:
            raise RuntimeError("No samples found for the configured datasets")

        resize_size = cfg.image_size if isinstance(cfg.image_size, int) else 256
        self.target_transform = transforms.Compose(
            [
                transforms.Resize((resize_size, resize_size), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.grayscale_transform = transforms.Compose(
            [
                transforms.Resize((resize_size, resize_size), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def _make_sketch(self, image: Image.Image) -> Image.Image:
        gray = ImageOps.grayscale(image)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = ImageOps.autocontrast(ImageOps.invert(edges))
        return edges

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path, style_idx = self.samples[idx]
        image = Image.open(path).convert("RGB")

        painting_image = image
        sketch_image = self._make_sketch(image)

        if self.cfg.random_horizontal_flip and random.random() < 0.5:
            painting_image = ImageOps.mirror(painting_image)
            sketch_image = ImageOps.mirror(sketch_image)
        if self.cfg.random_vertical_flip and random.random() < 0.5:
            painting_image = ImageOps.flip(painting_image)
            sketch_image = ImageOps.flip(sketch_image)

        painting_tensor = self.target_transform(painting_image)
        sketch_tensor = self.grayscale_transform(sketch_image)

        sample = {
            "sketch": sketch_tensor,
            "painting": painting_tensor,
            "style": torch.tensor(style_idx, dtype=torch.long),
            "path": str(path),
        }

        if idx == 0:
            log_tensor_stats("sketch", sketch_tensor)
            log_tensor_stats("painting", painting_tensor)

        return sample


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    sketches = torch.stack([x["sketch"] for x in batch])
    paintings = torch.stack([x["painting"] for x in batch])
    styles = torch.stack([x["style"] for x in batch])
    paths = [x["path"] for x in batch]
    return {
        "sketch": sketches,
        "painting": paintings,
        "style": styles,
        "path": paths,
    }


def build_dataset(cfg: DatasetConfig, split: str) -> MonetSketchDataset:
    return MonetSketchDataset(cfg, split=split)


def create_dataloader(
    cfg: DatasetConfig,
    split: str,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = build_dataset(cfg, split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

