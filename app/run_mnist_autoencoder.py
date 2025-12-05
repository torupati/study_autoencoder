"""
Autoencoder for MNIST dataset.

This script provides training and evaluation of Autoencoder on MNIST dataset
with support for:
- CLI arguments for configuration
- Checkpoint loading and resume training
- Dual logging (console and file)
- Training with both train and test datasets
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
from models.mnist.base_ae import Autoencoder
from models.mnist.dataset_mnist import get_mnist_dataset
from models.mnist.logging_utils import setup_logging
from models.mnist.mnist_utils import (
    plot_latent,
    plot_latent_each_digit,
    plot_reconstructed,
)


def train_autoenc(
    model: Autoencoder,
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
    """Train Autoencoder model for multiple epochs.

    Args:
        model: Autoencoder model instance
        optimizer: Optimizer for training
        trainloader: DataLoader for training data
        testloader: DataLoader for test data
        start_epoch: Starting epoch number
        num_epochs: Total number of epochs to train
        output_dir: Directory to save checkpoints
        device: Device to run training on ('cuda' or 'cpu')
        logger: Logger instance
        no_progress: Whether to disable progress bars (default: False)

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
            img_flat = torch.flatten(img, start_dim=1)
            img_hat = model(img_flat)
            loss = ((img_flat - img_hat) ** 2).sum()
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            for img_t, _ in tqdm(testloader, desc="Testing", disable=no_progress):
                img = img_t.to(device, dtype=torch.float)
                img_flat = torch.flatten(img, start_dim=1)
                img_hat = model(img_flat)
                loss = ((img_flat - img_hat) ** 2).sum()
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

        # Save checkpoint every 10 epochs
        if (i + 1) % 10 == 0:
            ckpt_path = os.path.join(output_dir, f"ckpt_mnist_autoencoder_train_epoch{i:03d}.pth")
            torch.save(
                {
                    "epoch": i + 1,
                    "model_state_dict": model.to("cpu").state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "latent_dim": model.latent_dim,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                },
                ckpt_path,
            )
            logger.info(f"Saved checkpoint to {ckpt_path}")

    return train_loss_log, test_loss_log


def plot_loss(
    train_loss_log: list[float],
    test_loss_log: list[float],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Plot and save training/test loss curves.

    Args:
        train_loss_log: List of training losses
        test_loss_log: List of test losses
        output_dir: Directory to save the plot
        logger: Logger instance
    """
    plt.figure(figsize=(10, 6))
    plt.suptitle("Autoencoder Loss")
    plt.plot(train_loss_log, label="train_loss", marker="o")
    plt.plot(test_loss_log, label="test_loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    loss_plot_path = os.path.join(output_dir, "ae_loss.png")
    plt.savefig(loss_plot_path, dpi=100, bbox_inches="tight")
    logger.info(f"Saved loss plot to {loss_plot_path}")
    plt.show()


def main(args: argparse.Namespace) -> None:
    """Main training and evaluation function.

    Args:
        args: Command line arguments
    """
    torch.manual_seed(0)
    logger = setup_logging("autoencoder_mnist.log")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

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
        logger.error(f"Error loading MNIST dataset: {e}")
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
    start_epoch = 0
    cur_latent_dim = args.latent_dim

    if ckpt_path.is_file():
        checkpoint = load_checkpoint(args.ckpt, logger)
        if checkpoint is None:
            logger.error("Failed to load checkpoint, exiting")
            return
        cur_latent_dim = checkpoint.get("latent_dim", args.latent_dim)
        start_epoch = checkpoint.get("epoch", 0)

        logger.info("Loaded checkpoint: epoch=%d, latent_dim=%d", start_epoch, cur_latent_dim)

        # Initialize model and load state
        logger.info(f"Initializing Autoencoder (latent_dim={cur_latent_dim})")
        ae = Autoencoder(cur_latent_dim, 28 * 28).to(device)
        ae.load_state_dict(checkpoint["model_state_dict"])

        if args.resume:
            logger.info("Resuming training from epoch %d", start_epoch)
    else:
        logger.info("Initializing Autoencoder: latent_dim=%d", args.latent_dim)
        ae = Autoencoder(args.latent_dim, 28 * 28).to(device)

    # Initialize optimizer
    opt = optim.Adam(ae.parameters())

    # Training
    train_loss_log: list[float] = []
    test_loss_log: list[float] = []

    logger.info("Starting training loop")
    train_loss_log, test_loss_log = train_autoenc(
        model=ae,
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
    plot_loss(train_loss_log, test_loss_log, args.output_dir, logger)

    # Visualization
    logger.info("Generating visualizations")
    fig, ax = plt.subplots(1, 1)
    ax = plot_latent_each_digit(ax, ae, trainset)
    pngfile = os.path.join(args.output_dir, "ae_latent_each_digit.png")
    fig.savefig(pngfile)
    logger.info(f"Saved {pngfile}")

    fig, ax = plt.subplots(1, 1)
    ax = plot_latent(ax, ae, trainset, 100)
    pngfile = os.path.join(args.output_dir, "ae_latent_space_projection.png")
    fig.savefig(pngfile)
    logger.info(f"Saved {pngfile}")

    fig, ax = plt.subplots(1, 1)
    ax = plot_reconstructed(ax, ae, r0=(-3, 3), r1=(-3, 3))
    pngfile = os.path.join(args.output_dir, "ae_reconstructed.png")
    fig.savefig(pngfile)
    logger.info(f"Saved {pngfile}")

    logger.info("Autoencoder training script completed")


def prepare_argparse() -> argparse.ArgumentParser:
    """Create argument parser for Autoencoder training.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description="PyTorch Autoencoder for MNIST dataset")

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
    # Model parameters
    parser.add_argument(
        "--latent_dim",
        "-d",
        default=2,
        type=int,
        help="Latent variable dimension",
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
