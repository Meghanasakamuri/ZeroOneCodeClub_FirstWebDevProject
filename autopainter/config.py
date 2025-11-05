from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class DataConfig:
    dataset_root: Path = Path("data/processed/monet_sketches")
    raw_root: Path = Path("data/raw")
    image_size: int = 256
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    use_huggingface: bool = False
    hf_dataset_name: str = "huggan/monet-train"
    hf_split: str = "train"
    grayscale_mode: Literal["canny", "laplacian", "pencil"] = "canny"
    style_labels: list[str] = field(default_factory=lambda: ["monet"])
    val_split: float = 0.05


@dataclass
class LossConfig:
    lambda_l1: float = 100.0
    lambda_perceptual: float = 10.0
    lambda_gan: float = 1.0
    perceptual_layers: tuple[str, ...] = ("relu_1", "relu_2", "relu_3", "relu_4")
    adversarial_mode: Literal["patch", "pixel"] = "patch"
    feature_matching: bool = True


@dataclass
class OptimConfig:
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 0.0


@dataclass
class TrainerConfig:
    epochs: int = 200
    accumulate_steps: int = 1
    mixed_precision: bool = True
    gradient_clip_norm: Optional[float] = 1.0
    random_flip: bool = True
    random_jitter: bool = True
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("runs/autopainter")
    save_every: int = 5
    evaluate_every: int = 1
    num_val_samples: int = 8
    seed: int = 42


@dataclass
class ExperimentConfig:
    name: str = "autopainter_monet"
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    generator_channels: int = 64
    discriminator_channels: int = 64
    multi_style: bool = True
    perceptual_backbone: Literal["vgg19", "vgg16"] = "vgg19"


def load_config(path: Optional[str | Path] = None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()

    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    def _merge(dataclass_cls, values):
        base = dataclass_cls()
        for key, value in (values or {}).items():
            if hasattr(base, key):
                current = getattr(base, key)
                if dataclasses.is_dataclass(current):
                    setattr(base, key, _merge(type(current), value))
                else:
                    setattr(base, key, value)
        return base

    cfg = ExperimentConfig()
    for k, v in (data or {}).items():
        if hasattr(cfg, k):
            current = getattr(cfg, k)
            if dataclasses.is_dataclass(current):
                setattr(cfg, k, _merge(type(current), v))
            else:
                setattr(cfg, k, v)
    return cfg
