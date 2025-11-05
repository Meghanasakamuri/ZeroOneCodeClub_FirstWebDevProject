from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class DatasetItem:
    image_path: Path
    style_index: int


def _to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
    tensor = transforms.ToTensor()(image)
    if normalize:
        tensor = tensor * 2.0 - 1.0
    return tensor


class AutoPainterDataset(Dataset):
    def __init__(
        self,
        root: Path,
        image_size: int = 256,
        split: str = "train",
        style_labels: Optional[list[str]] = None,
        augment: bool = False,
        grayscale_mode: str = "canny",
    ) -> None:
        super().__init__()
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root {self.root} does not exist")

        self.split = split
        self.image_size = image_size
        self.grayscale_mode = grayscale_mode
        self.augment = augment
        self.items: list[DatasetItem] = []

        if style_labels is None:
            style_labels = [d.name for d in self.root.iterdir() if d.is_dir()]

        for style_idx, style_name in enumerate(sorted(style_labels)):
            style_dir = self.root / style_name / split
            if not style_dir.exists():
                continue
            for image_path in style_dir.glob("*.jpg"):
                self.items.append(DatasetItem(image_path=image_path, style_index=style_idx))
            for image_path in style_dir.glob("*.png"):
                self.items.append(DatasetItem(image_path=image_path, style_index=style_idx))

        if not self.items:
            raise RuntimeError(f"No images found in {self.root} for split {split}")

        self.resize = transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC)
        self.flip = transforms.RandomHorizontalFlip()
        self.color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, path: Path) -> Image.Image:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self.resize(img)
            return img

    def _build_sketch(self, image: Image.Image) -> Image.Image:
        np_img = np.array(image)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        if self.grayscale_mode == "canny":
            edges = cv2.Canny(gray, 80, 180)
            sketch = 255 - edges
        elif self.grayscale_mode == "laplacian":
            lap = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
            sketch = 255 - lap
        elif self.grayscale_mode == "pencil":
            inv = 255 - gray
            blur = cv2.GaussianBlur(inv, (21, 21), sigmaX=0, sigmaY=0)
            sketch = cv2.divide(gray, 255 - blur, scale=256)
        else:
            raise ValueError(f"Unknown grayscale mode {self.grayscale_mode}")

        sketch = Image.fromarray(sketch).convert("L")
        sketch = self.resize(sketch)
        return sketch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.items[idx]
        image = self._load_image(item.image_path)

        if self.augment and self.split == "train":
            image = self.flip(image)
            image = self.color_jitter(image)

        sketch = self._build_sketch(image)

        gray_tensor = _to_tensor(sketch)
        color_tensor = _to_tensor(image)
        return {
            "sketch": gray_tensor,
            "painting": color_tensor,
            "style_index": torch.tensor(item.style_index, dtype=torch.long),
        }
