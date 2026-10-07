"""Shared Trainer for model training and evaluation.

Provides a small, well-documented Trainer and TrainerConfig used by app/* scripts.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from models.checkpoint import load_checkpoint
from models.logging_utils import setup_logging


@dataclasses.dataclass
class TrainerConfig:
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    device: str | None = None
    save_every: int = 1
    ckpt_path: str | None = None
    resume_from: str | None = None
    log_file: str = "training.log"


class Trainer:
    """Simple Trainer that encapsulates training/validation loops, logging and checkpointing.

    Usage: create Trainer(model, optimizer, loss_fn, train_loader, val_loader, config)
    and call trainer.fit(). Apps should keep CLI parsing; Trainer keeps training logic shared.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: Any,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: TrainerConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger or setup_logging(config.log_file)

        # device selection
        if config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        self.model.to(self.device)

        # history
        self.train_history: list[float] = []
        self.val_history: list[float] = []

    def _save_checkpoint(self, epoch: int) -> None:
        if not self.config.ckpt_path:
            return
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        latent_dims = getattr(self.model, "latent_dims", None)
        if latent_dims is not None:
            state["latent_dims"] = latent_dims
        path = f"{self.config.ckpt_path}_epoch{epoch:03d}.pth"
        torch.save(state, path)
        self.logger.info("Saved checkpoint: %s", path)

    def _maybe_load_checkpoint(self) -> int | None:
        if not self.config.resume_from:
            return None
        ckpt = load_checkpoint(self.config.resume_from, self.logger)
        if not ckpt:
            return None
        try:
            # Accept both current keys and the legacy "model_state"/"optimizer_state" keys
            model_state = ckpt.get("model_state_dict", ckpt.get("model_state"))
            optimizer_state = ckpt.get("optimizer_state_dict", ckpt.get("optimizer_state"))
            if model_state is not None:
                self.model.load_state_dict(model_state)
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)
            epoch = int(ckpt.get("epoch", 0))
            self.logger.info("Restored checkpoint epoch=%s", epoch)
            return epoch
        except Exception as e:
            self.logger.error("Failed to restore checkpoint: %s", e)
            return None

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        start = time.time()
        for batch_idx, (x, _) in enumerate(self.train_loader):
            x = x.to(self.device)
            self.optimizer.zero_grad()
            recon = self.model(x)
            loss = self.loss_fn(recon, x)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.detach().cpu())
            # if batch_idx % 100 == 0:
            #    self.logger.info(
            #        "Epoch %d Batch %d/%d loss=%.6f",
            #        epoch,
            #        batch_idx,
            #        len(self.train_loader),
            #        float(loss.detach().cpu()),
            #    )
        elapsed = time.time() - start
        avg_loss = total_loss / len(self.train_loader)
        # self.logger.info("Epoch %d training done. avg_loss=%.6f time=%.1fs", epoch, avg_loss, elapsed)
        return avg_loss

    def evaluate(self) -> float | None:
        if self.val_loader is None:
            return None
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, _ in self.val_loader:
                x = x.to(self.device)
                recon = self.model(x)
                loss = self.loss_fn(recon, x)
                total_loss += float(loss.detach().cpu())
        avg_loss = total_loss / len(self.val_loader)
        self.logger.info("Validation avg_loss=%.6f", avg_loss)
        return avg_loss

    def fit(self) -> tuple[list[float], list[float]]:
        start_epoch = 1
        restored = self._maybe_load_checkpoint()
        if restored is not None:
            start_epoch = restored + 1

        for epoch in range(start_epoch, self.config.epochs + 1):
            train_loss = self.train_epoch(epoch)
            # record
            self.train_history.append(train_loss)
            val_loss = self.evaluate()
            if val_loss is not None:
                self.val_history.append(val_loss)

            if epoch % self.config.save_every == 0:
                self._save_checkpoint(epoch)

        self.logger.info("Training finished")
        return self.train_history, self.val_history
