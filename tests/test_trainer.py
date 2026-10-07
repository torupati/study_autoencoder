import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.mnist.vae import VariationalAutoencoder

from app.common import Trainer, TrainerConfig


def _make_trainer(tmp_path: Path, **config_kwargs) -> Trainer:
    torch.manual_seed(0)
    x = torch.rand(8, 28 * 28)
    loader = DataLoader(TensorDataset(x, torch.zeros(8)), batch_size=4)
    model = VariationalAutoencoder(2, 28 * 28)
    opt = torch.optim.Adam(model.parameters())

    def loss_fn(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((recon - target) ** 2).sum() + model.encoder.kl

    config = TrainerConfig(
        ckpt_path=str(tmp_path / "ckpt"),
        log_file=str(tmp_path / "train.log"),
        **config_kwargs,
    )
    return Trainer(model, opt, loss_fn, loader, loader, config, logging.getLogger("test"))


def test_checkpoint_contains_latent_dims(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path, epochs=2, save_every=2)
    train_hist, val_hist = trainer.fit()

    assert len(train_hist) == 2
    assert len(val_hist) == 2
    ckpt = torch.load(tmp_path / "ckpt_epoch002.pth", weights_only=True)
    assert ckpt["epoch"] == 2
    assert ckpt["latent_dims"] == 2
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt


def test_resume_continues_from_saved_epoch(tmp_path: Path) -> None:
    _make_trainer(tmp_path, epochs=2, save_every=2).fit()

    resumed = _make_trainer(
        tmp_path, epochs=3, save_every=1, resume_from=str(tmp_path / "ckpt_epoch002.pth")
    )
    train_hist, _ = resumed.fit()

    assert len(train_hist) == 1
    assert (tmp_path / "ckpt_epoch003.pth").is_file()
