from __future__ import annotations

from pathlib import Path
from typing import Dict

from torch.utils.data import DataLoader

from autopainter.config import ExperimentConfig

from .dataset import AutoPainterDataset


def create_dataloaders(cfg: ExperimentConfig) -> Dict[str, DataLoader]:
    data_root = Path(cfg.data.dataset_root)
    styles = cfg.data.style_labels

    train_dataset = AutoPainterDataset(
        root=data_root,
        image_size=cfg.data.image_size,
        split="train",
        style_labels=styles,
        augment=cfg.trainer.random_flip,
        grayscale_mode=cfg.data.grayscale_mode,
    )

    val_dataset = AutoPainterDataset(
        root=data_root,
        image_size=cfg.data.image_size,
        split="val",
        style_labels=styles,
        augment=False,
        grayscale_mode=cfg.data.grayscale_mode,
    )

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=max(1, cfg.data.batch_size // 2),
            shuffle=False,
            num_workers=max(1, cfg.data.num_workers // 2),
            pin_memory=cfg.data.pin_memory,
        ),
    }

    test_dir = data_root / "test"
    if test_dir.exists():
        test_dataset = AutoPainterDataset(
            root=data_root,
            image_size=cfg.data.image_size,
            split="test",
            style_labels=styles,
            augment=False,
            grayscale_mode=cfg.data.grayscale_mode,
        )
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=max(1, cfg.data.batch_size // 2),
            shuffle=False,
            num_workers=max(1, cfg.data.num_workers // 2),
            pin_memory=cfg.data.pin_memory,
        )

    return loaders
