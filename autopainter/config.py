"""Configuration dataclasses for the AutoPainter project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class DatasetConfig:
    """Configuration for dataset creation and preprocessing."""

    source: str = "monet2photo"
    split: str = "train"
    image_size: int = 256
    max_samples: Optional[int] = None
    random_horizontal_flip: bool = True
    random_vertical_flip: bool = False
    grayscale_input: bool = True
    cache_dir: Optional[str] = None
    num_workers: int = 4
    use_webdataset_style: bool = False
    paired_folder_root: Optional[str] = None
    paired_folder_structure: str = "sketches->paintings"
    style_mapping: Dict[str, int] = field(default_factory=lambda: {"monet": 0})
    infer_style: Optional[str] = "monet"


@dataclass
class ModelConfig:
    """Configuration for model architectures."""

    input_channels: int = 1
    output_channels: int = 3
    style_condition_dim: int = 2
    generator_filters: int = 64
    discriminator_filters: int = 64
    use_dropout: bool = True
    norm_layer: str = "instance"  # or "batch"
    discriminator_type: str = "patch"  # or "pixel"
    init_type: str = "normal"
    init_gain: float = 0.02


@dataclass
class LossConfig:
    """Configuration for training losses."""

    gan_mode: str = "lsgan"  # or "vanilla", "hinge"
    lambda_l1: float = 100.0
    lambda_perceptual: float = 10.0
    perceptual_layer: str = "relu3_3"
    pixel_loss: bool = False


@dataclass
class OptimizerConfig:
    """Configuration for optimizers."""

    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 0.0


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    epochs: int = 200
    batch_size: int = 4
    val_interval: int = 1
    checkpoint_interval: int = 5
    resume_from: Optional[str] = None
    mixed_precision: bool = True
    accumulate_grad_batches: int = 1
    gradient_clip_norm: Optional[float] = None
    log_interval: int = 50
    sample_interval: int = 500
    output_dir: str = "checkpoints"


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""

    checkpoint_path: str = "checkpoints/autopainter_monet.pt"
    device: str = "cuda"
    style: str = "monet"
    fp16: bool = True


@dataclass
class AutoPainterConfig:
    """Top-level configuration object for AutoPainter."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def as_dict(self) -> Dict[str, Dict[str, object]]:
        return {
            "dataset": vars(self.dataset),
            "model": vars(self.model),
            "losses": vars(self.losses),
            "optimizer": vars(self.optimizer),
            "training": vars(self.training),
            "inference": vars(self.inference),
        }


DEFAULT_CONFIG = AutoPainterConfig()

