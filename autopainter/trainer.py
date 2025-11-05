from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from autopainter.config import ExperimentConfig
from autopainter.data import create_dataloaders
from autopainter.models import (
    AdversarialLoss,
    PatchDiscriminator,
    PerceptualLoss,
    ReconstructionLoss,
    UNetGenerator,
)
from autopainter.models.losses import feature_matching_loss


class AutoPainterTrainer:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loaders = create_dataloaders(cfg)

        num_styles = len(cfg.data.style_labels) if cfg.multi_style else None
        self.generator = UNetGenerator(
            in_channels=1,
            out_channels=3,
            base_channels=cfg.generator_channels,
            num_styles=num_styles,
        ).to(self.device)
        self.discriminator = PatchDiscriminator(
            in_channels=1 + 3,
            base_channels=cfg.discriminator_channels,
            mode=cfg.loss.adversarial_mode,
        ).to(self.device)

        self.recon_loss = ReconstructionLoss("l1")
        self.adv_loss = AdversarialLoss(mode="bce")
        self.perc_loss: Optional[PerceptualLoss] = None
        if cfg.loss.lambda_perceptual > 0:
            self.perc_loss = PerceptualLoss(cfg.loss.perceptual_layers, cfg.perceptual_backbone).to(self.device)

        self.opt_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=cfg.optim.lr,
            betas=(cfg.optim.beta1, cfg.optim.beta2),
            weight_decay=cfg.optim.weight_decay,
        )
        self.opt_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=cfg.optim.lr,
            betas=(cfg.optim.beta1, cfg.optim.beta2),
            weight_decay=cfg.optim.weight_decay,
        )

        self.scaler_g = amp.GradScaler(enabled=cfg.trainer.mixed_precision)
        self.scaler_d = amp.GradScaler(enabled=cfg.trainer.mixed_precision)

        self.global_step = 0
        self.start_epoch = 0

        self.writer = SummaryWriter(log_dir=str(cfg.trainer.log_dir / cfg.name))
        self.ckpt_dir = cfg.trainer.checkpoint_dir / cfg.name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def fit(self) -> None:
        for epoch in range(self.start_epoch, self.cfg.trainer.epochs):
            start = time.time()
            train_metrics = self._train_one_epoch(epoch)
            duration = time.time() - start

            if epoch % self.cfg.trainer.evaluate_every == 0:
                val_metrics = self._evaluate(split="val", epoch=epoch)
            else:
                val_metrics = {}

            if epoch % self.cfg.trainer.save_every == 0:
                self._save_checkpoint(epoch)

            summary = {
                "epoch": epoch,
                "duration_sec": duration,
                "train": train_metrics,
                "val": val_metrics,
            }
            print(json.dumps(summary, indent=2))

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.generator.train()
        self.discriminator.train()

        metrics = {
            "g_adv": 0.0,
            "g_recon": 0.0,
            "g_perc": 0.0,
            "g_fm": 0.0,
            "d_real": 0.0,
            "d_fake": 0.0,
        }
        num_batches = 0

        for batch_idx, batch in enumerate(self.loaders["train"]):
            num_batches += 1
            sketch = batch["sketch"].to(self.device)
            target = batch["painting"].to(self.device)
            styles = batch["style_index"].to(self.device)

            # Train discriminator
            self._toggle_grad(self.discriminator, True)
            with amp.autocast(enabled=self.cfg.trainer.mixed_precision):
                fake = self.generator(sketch, styles if self.cfg.multi_style else None).detach()
                logits_real, real_features = self.discriminator(sketch, target, return_features=True)
                logits_fake, fake_features = self.discriminator(sketch, fake, return_features=True)

                loss_d_real = self.adv_loss(logits_real, True)
                loss_d_fake = self.adv_loss(logits_fake, False)
                loss_d = (loss_d_real + loss_d_fake) * 0.5

            self.opt_d.zero_grad(set_to_none=True)
            self.scaler_d.scale(loss_d).backward()
            if self.cfg.trainer.gradient_clip_norm:
                self.scaler_d.unscale_(self.opt_d)
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.cfg.trainer.gradient_clip_norm)
            self.scaler_d.step(self.opt_d)
            self.scaler_d.update()

            # Train generator
            self._toggle_grad(self.discriminator, False)
            with amp.autocast(enabled=self.cfg.trainer.mixed_precision):
                fake = self.generator(sketch, styles if self.cfg.multi_style else None)
                logits_fake_g = self.discriminator(sketch, fake)
                loss_g_adv = self.adv_loss(logits_fake_g, True) * self.cfg.loss.lambda_gan
                loss_recon = self.recon_loss(fake, target) * self.cfg.loss.lambda_l1

                loss_perc = torch.tensor(0.0, device=self.device)
                if self.perc_loss and self.cfg.loss.lambda_perceptual > 0:
                    loss_perc = self.perc_loss(fake, target) * self.cfg.loss.lambda_perceptual

                loss_fm = torch.tensor(0.0, device=self.device)
                if self.cfg.loss.feature_matching:
                    _, real_features_g = self.discriminator(sketch, target, return_features=True)
                    _, fake_features_g = self.discriminator(sketch, fake, return_features=True)
                    loss_fm = feature_matching_loss(real_features_g, fake_features_g)

                loss_g = loss_g_adv + loss_recon + loss_perc + loss_fm

            self.opt_g.zero_grad(set_to_none=True)
            self.scaler_g.scale(loss_g).backward()
            if self.cfg.trainer.gradient_clip_norm:
                self.scaler_g.unscale_(self.opt_g)
                nn.utils.clip_grad_norm_(self.generator.parameters(), self.cfg.trainer.gradient_clip_norm)
            self.scaler_g.step(self.opt_g)
            self.scaler_g.update()
            self._toggle_grad(self.discriminator, True)

            metrics["g_adv"] += loss_g_adv.item()
            metrics["g_recon"] += loss_recon.item()
            metrics["g_perc"] += loss_perc.item()
            metrics["g_fm"] += loss_fm.item()
            metrics["d_real"] += loss_d_real.item()
            metrics["d_fake"] += loss_d_fake.item()

            if batch_idx % 50 == 0:
                self._log_images(epoch, sketch, target, fake)

            self._write_scalars({
                "train/g_adv": loss_g_adv.item(),
                "train/g_recon": loss_recon.item(),
                "train/g_perc": loss_perc.item(),
                "train/g_fm": loss_fm.item(),
                "train/d_real": loss_d_real.item(),
                "train/d_fake": loss_d_fake.item(),
            })

            self.global_step += 1

        for key in metrics:
            metrics[key] /= max(1, num_batches)
        return metrics

    @torch.no_grad()
    def _evaluate(self, split: str, epoch: int) -> Dict[str, float]:
        self.generator.eval()
        self.discriminator.eval()

        loader = self.loaders[split]
        metrics = {"recon": 0.0}
        num_batches = 0

        for batch_idx, batch in enumerate(loader):
            sketch = batch["sketch"].to(self.device)
            target = batch["painting"].to(self.device)
            styles = batch["style_index"].to(self.device)

            fake = self.generator(sketch, styles if self.cfg.multi_style else None)
            recon_loss = self.recon_loss(fake, target)
            metrics["recon"] += recon_loss.item()

            if batch_idx < self.cfg.trainer.num_val_samples:
                self._log_images(epoch, sketch, target, fake, tag=f"{split}/samples_{batch_idx}")

            num_batches += 1

        metrics = {k: v / max(1, num_batches) for k, v in metrics.items()}
        self._write_scalars({f"{split}/{k}": v for k, v in metrics.items()}, step=epoch)
        return metrics

    def _log_images(
        self,
        epoch: int,
        sketch: torch.Tensor,
        target: torch.Tensor,
        fake: torch.Tensor,
        tag: str = "train/samples",
    ) -> None:
        def denorm(x: torch.Tensor) -> torch.Tensor:
            return (x + 1.0) / 2.0

        sketch_rgb = denorm(sketch.repeat(1, 3, 1, 1))
        grid = torch.cat([sketch_rgb, denorm(fake), denorm(target)], dim=0)
        grid = make_grid(grid, nrow=sketch.size(0))
        self.writer.add_image(tag, grid, global_step=self.global_step)

        out_dir = self.cfg.trainer.log_dir / self.cfg.name / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_image(grid, out_dir / f"epoch_{epoch:04d}_step_{self.global_step:08d}.png")

    def _write_scalars(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        step = self.global_step if step is None else step
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, global_step=step)

    def _save_checkpoint(self, epoch: int) -> None:
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "scaler_g": self.scaler_g.state_dict(),
            "scaler_d": self.scaler_d.state_dict(),
            "config": self.cfg,
        }
        path = self.ckpt_dir / f"epoch_{epoch:04d}.pt"
        torch.save(ckpt, path)

    def load_checkpoint(self, checkpoint_path: Path, strict: bool = True) -> None:
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(ckpt["generator"], strict=strict)
        self.discriminator.load_state_dict(ckpt["discriminator"], strict=strict)
        self.opt_g.load_state_dict(ckpt["opt_g"])
        self.opt_d.load_state_dict(ckpt["opt_d"])
        self.scaler_g.load_state_dict(ckpt["scaler_g"])
        self.scaler_d.load_state_dict(ckpt["scaler_d"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.global_step = ckpt.get("global_step", 0)

    @staticmethod
    def _toggle_grad(model: nn.Module, requires_grad: bool) -> None:
        for param in model.parameters():
            param.requires_grad = requires_grad


def load_trainer(cfg: ExperimentConfig, checkpoint: Optional[Path] = None) -> AutoPainterTrainer:
    trainer = AutoPainterTrainer(cfg)
    if checkpoint:
        trainer.load_checkpoint(checkpoint)
    return trainer
