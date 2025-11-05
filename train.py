from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from autopainter.config import ExperimentConfig, load_config
from autopainter.trainer import AutoPainterTrainer, load_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AutoPainter Pix2Pix model")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    cfg: ExperimentConfig = load_config(args.config)
    set_seed(cfg.trainer.seed)

    trainer = load_trainer(cfg, args.resume)
    trainer.fit()


if __name__ == "__main__":
    main()
