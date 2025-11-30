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
from os import path
from pathlib import Path

import torch

torch.manual_seed(0)
import matplotlib.pyplot as plt
import torch.utils

from models.mnist.dataset_mnist import get_mnist_dataset


def setup_logging(log_file: str = "vqvae_mnist.log") -> logging.Logger:
    """Setup logging to both console and file.

    Args:
        log_file: Path to log file

    Returns:
        Logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicates
    logger.handlers = []

    # Console handler (INFO level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s"
    )
    stream_handler.setFormatter(stream_formatter)

    # File handler (DEBUG level)
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def load_checkpoint(ckpt_path: str, logger: logging.Logger) -> dict:
    """Load model checkpoint from file.

    Args:
        ckpt_path: Path to checkpoint file
        logger: Logger instance

    Returns:
        Dictionary containing checkpoint data with keys:
        - model_state_dict: Model weights
        - optimizer_state_dict: Optimizer state
        - epoch: Training epoch
        - e_dim: Embedding dimension
        - num_e: Number of embeddings
        - loss: Training loss
    """
    logger.info("Loading checkpoint from: %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, weights_only=True)
    logger.info("Checkpoint keys: %s", list(checkpoint.keys()))
    return checkpoint


def main(args: argparse.Namespace) -> None:
    """Main training and evaluation function.

    Args:
        args: Command line arguments
    """
    logger = setup_logging()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # Prepare dataset (MNIST) and dataloader
    dataset = get_mnist_dataset(args.dataset_dir, train=True)
    obs_dim = 28 * 28
    logger.debug("Prepare MNIST dataset from custom implementation")
    logger.debug("%s", dataset)

    data = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    logger.info("Data loader created with batch size=%d", args.batch_size)

    # Model initialization or loading from checkpoint
    if path.isfile(args.ckpt):
        checkpoint = load_checkpoint(args.ckpt, logger)
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

        # TODO: Initialize VQ-VAE model
        # vqvae = VQVAE(e_dim=args.e_dim, num_e=args.num_e, obs_dim=obs_dim).to(device)
        # vqvae = train_vqvae(vqvae, data, args.num_epoch, 0)

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

    return parser


if __name__ == "__main__":
    parser = prepare_argparse()
    args = parser.parse_args()
    main(args)
