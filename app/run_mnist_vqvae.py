"""
Vector Quantized Variational Autoencoder (VQ-VAE) for MNIST dataset.

This script provides training and evaluation of VQ-VAE on MNIST dataset
with support for:
- CLI arguments for configuration
- Checkpoint loading and resume training
- Dual logging (console and file)
"""

import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.checkpoint import load_checkpoint
from models.logging_utils import setup_logging
from models.mnist.dataset_mnist import get_mnist_dataset
from models.mnist.mnist_utils import (
    plot_latent,
    plot_latent_each_digit,
    plot_loss,
    plot_reconstructed,
)
from models.mnist.vqvae import VQVAE


def train_vqvae(
    model: VQVAE,
    optimizer: optim.Optimizer,
    trainloader: DataLoader,
    testloader: DataLoader,
    start_epoch: int,
    num_epochs: int,
    output_dir: str,
    device: str,
    logger: logging.Logger,
    no_progress: bool = False,
) -> tuple[list[float], list[float]]:
    """Train VQ-VAE model for multiple epochs.

    Args:
        model: VQVAE model instance
        optimizer: Optimizer for training
        trainloader: DataLoader for training data
        testloader: DataLoader for test data
        start_epoch: Starting epoch number
        num_epochs: Total number of epochs to train
        output_dir: Directory to save checkpoints
        device: Device to run training on ('cuda' or 'cpu')
        logger: Logger instance

    Returns:
        Tuple of (train_loss_log, test_loss_log) containing loss values for each epoch
    """
    train_loss_log: list[float] = []
    test_loss_log: list[float] = []

    for i in range(start_epoch, num_epochs):
        logger.info(f"Epoch {i + 1}/{num_epochs}")
        train_loss: float = 0.0
        test_loss: float = 0.0
        model = model.to(device)

        # Training phase
        model.train()
        for img, _ in tqdm(trainloader, desc="Training", disable=no_progress):
            img = img.to(device, dtype=torch.float)
            optimizer.zero_grad()
            embedding_loss, x_hat = model(img)
            recon_loss = nn.MSELoss()(x_hat, img)
            loss = recon_loss + embedding_loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            for img_t, _ in tqdm(testloader, desc="Testing", disable=no_progress):
                img = img_t.to(device, dtype=torch.float)
                embedding_loss, x_hat = model(img)
                recon_loss = nn.MSELoss()(x_hat, img)
                loss = recon_loss + embedding_loss
                test_loss += loss.item()

        # Calculate average losses
        dataset_size_train: int = (
            len(trainloader.dataset) if hasattr(trainloader.dataset, "__len__") else 1
        )
        dataset_size_test: int = (
            len(testloader.dataset) if hasattr(testloader.dataset, "__len__") else 1
        )
        train_loss /= float(dataset_size_train)
        test_loss /= float(dataset_size_test)

        # Log and record losses
        log_msg = f"Epoch {i} - train_loss: {train_loss:.5f}, test_loss: {test_loss:.5f}"
        logger.info(log_msg)
        train_loss_log.append(train_loss)
        test_loss_log.append(test_loss)

        # Save checkpoint every 5 epochs
        if (i + 1) % 5 == 0:
            ckpt_path = os.path.join(output_dir, f"VQVAE_{i:03d}.pth")
            torch.save(
                {
                    "param": model.to("cpu").state_dict(),
                    "opt": optimizer.state_dict(),
                    "epoch": i + 1,
                },
                ckpt_path,
            )
            logger.info(f"Saved checkpoint to {ckpt_path}")

    return train_loss_log, test_loss_log


def main(args: argparse.Namespace) -> None:
    """Main training and evaluation function.

    Args:
        args: Command line arguments
    """
    torch.manual_seed(0)
    logger = setup_logging("vqvae_mnist.log")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # Prepare dataset (MNIST) and dataloader
    dataset = get_mnist_dataset(args.dataset_dir, train=True)
    obs_dim = 28 * 28
    logger.debug("Prepare MNIST dataset from custom implementation")
    logger.debug("%s", dataset)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare data
    logger.info(f"Loading MNIST dataset from {args.dataset_dir}")
    transform = transforms.Compose(
        [
            transforms.Normalize((0.5,), (0.5,))
            # Normalize to [-1, 1] range
        ]
    )

    try:
        trainset = get_mnist_dataset(root=args.dataset_dir, train=True, transform=transform)
        testset = get_mnist_dataset(root=args.dataset_dir, train=False, transform=transform)
    except FileNotFoundError as e:
        logger.error(f"Dataset files not found: {e}")
        return

    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=0
    )
    testloader = DataLoader(
        testset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=0
    )
    logger.info(f"Train samples: {len(trainset)}, Test samples: {len(testset)}")

    # Model initialization or loading from checkpoint
    ckpt_path = Path(args.ckpt)
    if ckpt_path.is_file():
        checkpoint = load_checkpoint(args.ckpt, logger)
        if checkpoint is None:
            logger.error("Failed to load checkpoint, exiting")
            return
        e_dim = checkpoint.get("e_dim", args.e_dim)
        num_e = checkpoint.get("num_e", args.num_e)
        cur_epoch = checkpoint.get("epoch", 0)

        logger.info("Loaded checkpoint: epoch=%d, e_dim=%d, num_e=%d", cur_epoch, e_dim, num_e)

        # TODO: Initialize VQ-VAE model and load state dict
        # vqvae = VQVAE(e_dim=e_dim, num_e=num_e, obs_dim=obs_dim)
        # vqvae.load_state_dict(checkpoint["model_state_dict"])

        if args.resume:
            logger.info("Resuming training from epoch %d", cur_epoch)
            # TODO: Resume training
            # vqvae = train_vqvae(vqvae, data, args.num_epoch, cur_epoch)
    else:
        logger.info(
            "Initializing VQ-VAE: e_dim=%d, num_e=%d, obs_dim=%d", args.e_dim, args.num_e, obs_dim
        )
        # Initialize model
        logger.info(f"Initializing VQVAE model (e_dim={args.e_dim}, num_e={args.num_e})")
        model = VQVAE(128, args.e_dim, 2, args.num_e, 64, 0.25).to(device)
        # TODO: Initialize VQ-VAE model
        # vqvae = VQVAE(e_dim=args.e_dim, num_e=args.num_e, obs_dim=obs_dim).to(device)
        # vqvae = train_vqvae(vqvae, data, args.num_epoch, 0)

    # Initialize optimizer
    opt = optim.Adam(model.parameters(), lr=3e-4, betas=(0.5, 0.9))

    # Training logs
    train_loss_log: list[float] = []
    test_loss_log: list[float] = []
    start_epoch = cur_epoch if ckpt_path.is_file() else 0

    logger.info("Starting training loop")
    train_loss_log, test_loss_log = train_vqvae(
        model=model,
        optimizer=opt,
        trainloader=trainloader,
        testloader=testloader,
        start_epoch=start_epoch,
        num_epochs=args.num_epoch,
        output_dir=args.output_dir,
        device=device,
        logger=logger,
        no_progress=args.no_progress,
    )

    # Plot and save loss curves
    plot_loss(
        train_loss_log,
        test_loss_log,
        args.output_dir,
        logger,
        title="VQ-VAE Loss",
        filename="VQVAE_loss.png",
    )

    logger.info("VQ-VAE training script completed")


def prepare_argparse() -> argparse.ArgumentParser:
    """Create argument parser for VQ-VAE training.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="PyTorch Vector Quantized Variational Autoencoder for MNIST dataset"
    )

    # Training conditions
    parser.add_argument(
        "--dataset_dir",
        default="./data/MNIST/",
        type=str,
        help="MNIST dataset directory",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Batch size for data loader",
    )
    parser.add_argument(
        "--num_epoch",
        "-n",
        default=40,
        type=int,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints"
    )
    # VQ-VAE specific parameters
    parser.add_argument(
        "--e_dim",
        "-e",
        default=64,
        type=int,
        help="Embedding dimension for vector quantization",
    )
    parser.add_argument(
        "--num_e",
        "-k",
        default=512,
        type=int,
        help="Number of embeddings in the codebook",
    )

    # Checkpoint options
    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--ckpt",
        "-c",
        default="",
        type=str,
        help="Checkpoint file path",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bars",
    )
    return parser


if __name__ == "__main__":
    parser = prepare_argparse()
    args = parser.parse_args()
    main(args)
