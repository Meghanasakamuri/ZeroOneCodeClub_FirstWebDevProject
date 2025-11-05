"""High-level training and inference pipeline for AutoPainter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from .config import AutoPainterConfig
from .datasets import build_dataset, collate_batch
from .losses import GANLoss, PerceptualLoss
from .models import build_discriminator, build_generator, expand_style_channel
from .utils import configure_logging, logger, set_seed, style_one_hot


class AutoPainterSystem:
    def __init__(self, config: AutoPainterConfig, device: Optional[str] = None) -> None:
        configure_logging()
        self.cfg = config
        set_seed()
        self.device = torch.device(device or (config.inference.device if torch.cuda.is_available() else "cpu"))

        self.num_styles = max(config.dataset.style_mapping.values()) + 1
        self.style_dim = config.model.style_condition_dim if self.num_styles > 1 else 0

        self.netG = build_generator(config.model, self.num_styles).to(self.device)
        self.netD = build_discriminator(config.model, self.num_styles).to(self.device)

        self.gan_loss = GANLoss(config.losses.gan_mode).to(self.device)
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = (
            PerceptualLoss(config.losses.perceptual_layer).to(self.device)
            if config.losses.lambda_perceptual > 0
            else None
        )

        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(),
            lr=config.optimizer.lr,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            weight_decay=config.optimizer.weight_decay,
        )
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(),
            lr=config.optimizer.lr,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            weight_decay=config.optimizer.weight_decay,
        )

        self.scaler = GradScaler(enabled=config.training.mixed_precision and self.device.type == "cuda")

        if hasattr(self.netG, "model") and isinstance(self.netG.model, nn.Module) and hasattr(self.netG.model, "model"):
            self.generator_in_channels = self.netG.model.model[0].weight.shape[1]
        else:
            self.generator_in_channels = config.model.input_channels + self.style_dim

        if hasattr(self.netD, "model") and isinstance(self.netD.model, nn.Sequential):
            self.discriminator_in_channels = self.netD.model[0].weight.shape[1]
        else:
            self.discriminator_in_channels = config.model.input_channels + config.model.output_channels + self.style_dim

        logger.info("Generator in channels: %s", self.generator_in_channels)
        logger.info("Discriminator in channels: %s", self.discriminator_in_channels)

    def _style_vector(self, styles: torch.Tensor) -> Optional[torch.Tensor]:
        if self.style_dim == 0:
            return None
        one_hot = style_one_hot(styles, self.num_styles)
        if one_hot.size(1) > self.style_dim:
            one_hot = one_hot[:, : self.style_dim]
        return one_hot

    def _prepare_generator_input(self, sketch: torch.Tensor, style_vec: Optional[torch.Tensor]) -> torch.Tensor:
        if sketch.size(1) == self.generator_in_channels:
            return sketch
        if style_vec is not None:
            return expand_style_channel(sketch, style_vec)
        if sketch.size(1) == 1 and self.generator_in_channels == 3:
            return sketch.repeat(1, 3, 1, 1)
        raise RuntimeError(
            f"Unable to assemble generator input: sketch_channels={sketch.size(1)}, required={self.generator_in_channels}"
        )

    def _prepare_discriminator_input(
        self, sketch_input: torch.Tensor, painting: torch.Tensor
    ) -> torch.Tensor:
        if sketch_input.size(1) + painting.size(1) == self.discriminator_in_channels:
            return torch.cat([sketch_input, painting], dim=1)
        if sketch_input.size(1) + painting.size(1) < self.discriminator_in_channels:
            pad = self.discriminator_in_channels - (sketch_input.size(1) + painting.size(1))
            padding = torch.zeros((sketch_input.size(0), pad, sketch_input.size(2), sketch_input.size(3)), device=sketch_input.device)
            return torch.cat([sketch_input, painting, padding], dim=1)
        raise RuntimeError("Discriminator input channels mismatch")

    def save_checkpoint(self, path: Path, epoch: int, step: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.cfg.as_dict(),
                "netG": self.netG.state_dict(),
                "netD": self.netD.state_dict(),
                "optimizer_G": self.optimizer_G.state_dict(),
                "optimizer_D": self.optimizer_D.state_dict(),
                "epoch": epoch,
                "step": step,
            },
            path,
        )
        logger.info("Saved checkpoint to %s", path)

    def load_checkpoint(self, path: Path, strict: bool = True) -> Dict[str, int]:
        checkpoint = torch.load(path, map_location=self.device)
        self.netG.load_state_dict(checkpoint["netG"], strict=strict)
        self.netD.load_state_dict(checkpoint["netD"], strict=strict)
        self.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        self.optimizer_D.load_state_dict(checkpoint["optimizer_D"])
        logger.info("Loaded checkpoint from %s", path)
        return {"epoch": checkpoint.get("epoch", 0), "step": checkpoint.get("step", 0)}

    def load_pretrained_generator(self, path: Path, adapt_channels: bool = True) -> None:
        state_dict = torch.load(path, map_location="cpu")
        if adapt_channels:
            first_key = next(k for k in state_dict if k.endswith("model.0.weight"))
            weight = state_dict[first_key]
            if weight.size(1) == 3 and self.generator_in_channels == 1:
                state_dict[first_key] = weight.mean(dim=1, keepdim=True)
        missing, unexpected = self.netG.load_state_dict(state_dict, strict=False)
        logger.info("Loaded pretrained generator %s | missing=%s unexpected=%s", path, missing, unexpected)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        cfg = self.cfg
        global_step = 0
        start_epoch = 0
        if cfg.training.resume_from:
            resume_info = self.load_checkpoint(Path(cfg.training.resume_from))
            start_epoch = resume_info.get("epoch", 0)
            global_step = resume_info.get("step", 0)

        for epoch in range(start_epoch, cfg.training.epochs):
            self.netG.train()
            self.netD.train()
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}", ncols=100)
            for batch in progress:
                global_step += 1
                loss_dict = self.training_step(batch)
                if global_step % cfg.training.log_interval == 0:
                    progress.set_postfix({k: f"{v:.3f}" for k, v in loss_dict.items()})

                if cfg.training.sample_interval and global_step % cfg.training.sample_interval == 0:
                    self.sample_and_save(batch, global_step)

            if cfg.training.checkpoint_interval and (epoch + 1) % cfg.training.checkpoint_interval == 0:
                ckpt_path = Path(cfg.training.output_dir) / f"autopainter_epoch_{epoch+1}.pt"
                self.save_checkpoint(ckpt_path, epoch + 1, global_step)

            if val_loader and cfg.training.val_interval and (epoch + 1) % cfg.training.val_interval == 0:
                self.validate(val_loader)

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        sketch = batch["sketch"].to(self.device)
        painting = batch["painting"].to(self.device)
        styles = batch["style"].to(self.device)

        style_vec = self._style_vector(styles)
        gen_input = self._prepare_generator_input(sketch, style_vec)

        self.optimizer_D.zero_grad(set_to_none=True)
        with autocast(enabled=self.scaler.is_enabled()):
            fake_painting = self.netG(gen_input.detach())
            fake_pair = self._prepare_discriminator_input(gen_input.detach(), fake_painting.detach())
            pred_fake = self.netD(fake_pair)
            loss_D_fake = self.gan_loss(pred_fake, False)

            real_pair = self._prepare_discriminator_input(gen_input.detach(), painting)
            pred_real = self.netD(real_pair)
            loss_D_real = self.gan_loss(pred_real, True)
            loss_D = (loss_D_fake + loss_D_real) * 0.5

        self.scaler.scale(loss_D).backward()
        self.scaler.step(self.optimizer_D)

        self.optimizer_G.zero_grad(set_to_none=True)
        with autocast(enabled=self.scaler.is_enabled()):
            fake_painting = self.netG(gen_input)
            pair_for_disc = self._prepare_discriminator_input(gen_input, fake_painting)
            pred_fake = self.netD(pair_for_disc)
            loss_G_GAN = self.gan_loss(pred_fake, True)
            loss_G_L1 = self.l1_loss(fake_painting, painting) * self.cfg.losses.lambda_l1
            loss_G_perc = (
                self.perceptual_loss(fake_painting, painting) * self.cfg.losses.lambda_perceptual
                if self.perceptual_loss is not None
                else torch.tensor(0.0, device=self.device)
            )
            loss_G = loss_G_GAN + loss_G_L1 + loss_G_perc

        self.scaler.scale(loss_G).backward()
        self.scaler.step(self.optimizer_G)
        self.scaler.update()

        return {
            "loss_D": loss_D.item(),
            "loss_G": loss_G.item(),
            "loss_G_GAN": loss_G_GAN.item(),
            "loss_G_L1": loss_G_L1.item(),
            "loss_G_perc": loss_G_perc.item() if isinstance(loss_G_perc, torch.Tensor) else loss_G_perc,
        }

    def validate(self, dataloader: DataLoader) -> None:
        self.netG.eval()
        total_l1 = 0.0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                sketch = batch["sketch"].to(self.device)
                painting = batch["painting"].to(self.device)
                styles = batch["style"].to(self.device)
                style_vec = self._style_vector(styles)
                gen_input = self._prepare_generator_input(sketch, style_vec)
                fake = self.netG(gen_input)
                total_l1 += self.l1_loss(fake, painting).item()
                total += 1
        logger.info("Validation L1: %.4f", total_l1 / max(total, 1))

    def sample_and_save(self, batch: Dict[str, torch.Tensor], step: int) -> None:
        self.netG.eval()
        with torch.no_grad():
            sketch = batch["sketch"].to(self.device)
            styles = batch["style"].to(self.device)
            style_vec = self._style_vector(styles)
            gen_input = self._prepare_generator_input(sketch, style_vec)
            fake = self.netG(gen_input)
            sketch_vis = sketch if sketch.size(1) == 3 else sketch.repeat(1, 3, 1, 1)
            grid = torch.cat([sketch_vis, fake], dim=0)
            out_path = Path(self.cfg.training.output_dir) / f"samples_step_{step}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(grid * 0.5 + 0.5, out_path, nrow=sketch.size(0))
            logger.info("Saved samples to %s", out_path)
        self.netG.train()

    def inference(self, image: torch.Tensor, style_index: int = 0) -> torch.Tensor:
        self.netG.eval()
        with torch.no_grad():
            image = image.to(self.device)
            style_tensor = torch.tensor([style_index], device=self.device, dtype=torch.long)
            style_vec = self._style_vector(style_tensor)
            gen_input = self._prepare_generator_input(image.unsqueeze(0), style_vec)
            output = self.netG(gen_input)[0]
        return output.cpu()

    def export_config(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.cfg.as_dict(), f, indent=2)
        logger.info("Saved config to %s", path)


def build_dataloaders(config: AutoPainterConfig) -> Dict[str, DataLoader]:
    train_dataset = build_dataset(config.dataset, split="train")
    config.dataset.style_mapping = train_dataset.style_to_idx
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    val_dataset = build_dataset(config.dataset, split="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    return {"train": train_loader, "val": val_loader}

