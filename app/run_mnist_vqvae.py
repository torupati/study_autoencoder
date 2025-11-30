import argparse
import logging
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import models.mnist.vqvae as vqvae_module
from models.mnist.dataset_mnist import get_mnist_dataset
from models.mnist.vqvae import VQVAE

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Set device for vqvae module
vqvae_module.device = device


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging to both console and file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"vqvae_train_{timestamp}.log")

    # Create logger
    logger = logging.getLogger("vqvae_train")
    logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def load_checkpoint(ckpt_path: str, model: VQVAE, opt: optim.Optimizer) -> int:
    """Load model and optimizer from checkpoint, return starting epoch."""
    if not os.path.exists(ckpt_path):
        return 0

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["param"])
    opt.load_state_dict(checkpoint["opt"])
    epoch = checkpoint.get("epoch", 0)
    return epoch


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train VQ-VAE on MNIST dataset")
    parser.add_argument(
        "--dataset_dir", type=str, default="./data/MNIST", help="Path to MNIST dataset directory"
    )
    parser.add_argument("--e_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument(
        "--num_e", type=int, default=512, help="Number of embeddings (codebook size)"
    )
    parser.add_argument("--num_epoch", type=int, default=5, help="Number of training epochs")
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Path to checkpoint file to resume training"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints"
    )
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for log files")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info("Starting VQ-VAE training")
    logger.info(f"Device: {device}")
    logger.info(f"Arguments: {args}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare data
    logger.info(f"Loading MNIST dataset from {args.dataset_dir}")
    transform = transforms.Compose(
        [
            transforms.Normalize((0.5,), (0.5,))
            # Normalize to [-1, 1] range
        ]
    )

    trainset = get_mnist_dataset(root=args.dataset_dir, train=True, transform=transform)
    testset = get_mnist_dataset(root=args.dataset_dir, train=False, transform=transform)

    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=0
    )
    testloader = DataLoader(
        testset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=0
    )
    logger.info(f"Train samples: {len(trainset)}, Test samples: {len(testset)}")

    # Initialize model
    logger.info(f"Initializing VQVAE model (e_dim={args.e_dim}, num_e={args.num_e})")
    model = VQVAE(128, args.e_dim, 2, args.num_e, 64, 0.25).to(device)

    # Initialize optimizer
    opt = optim.Adam(model.parameters(), lr=3e-4, betas=(0.5, 0.9))

    # Load checkpoint if provided
    start_epoch = 0
    if args.ckpt:
        start_epoch = load_checkpoint(args.ckpt, model, opt)
        logger.info(f"Loaded checkpoint from {args.ckpt}, resuming from epoch {start_epoch}")

    # Training logs
    train_loss_log = []
    test_loss_log = []

    logger.info("Starting training loop")
    for i in range(start_epoch, args.num_epoch):
        logger.info(f"Epoch {i + 1}/{args.num_epoch}")
        train_loss: float = 0.0
        test_loss: float = 0.0
        model = model.to(device)

        model.train()  # training mode
        for img, _ in tqdm(trainloader, desc="Training"):
            img = img.to(device, dtype=torch.float)
            opt.zero_grad()
            embedding_loss, x_hat = model(img)
            recon_loss = nn.MSELoss()(x_hat, img)
            loss = recon_loss + embedding_loss
            train_loss += loss.item()
            loss.backward()
            opt.step()

        model.eval()  # evaluation mode
        with torch.no_grad():
            for img_t, _ in tqdm(testloader, desc="Testing"):
                img = img_t.to(device, dtype=torch.float)
                embedding_loss, x_hat = model(img)
                recon_loss = nn.MSELoss()(x_hat, img)
                loss = recon_loss + embedding_loss
                test_loss += loss.item()

        # Calculate and log losses for each epoch
        dataset_size_train: int = (
            len(trainloader.dataset) if hasattr(trainloader.dataset, "__len__") else 1
        )
        dataset_size_test: int = (
            len(testloader.dataset) if hasattr(testloader.dataset, "__len__") else 1
        )
        train_loss /= float(dataset_size_train)
        test_loss /= float(dataset_size_test)

        log_msg = f"Epoch {i} - train_loss: {train_loss:.5f}, test_loss: {test_loss:.5f}"
        logger.info(log_msg)
        train_loss_log.append(train_loss)
        test_loss_log.append(test_loss)

        # Save checkpoint every 5 epochs
        if (i + 1) % 5 == 0:
            ckpt_path = os.path.join(args.output_dir, f"VQVAE_{i:03d}.pth")
            torch.save(
                {"param": model.to("cpu").state_dict(), "opt": opt.state_dict(), "epoch": i + 1},
                ckpt_path,
            )
            logger.info(f"Saved checkpoint to {ckpt_path}")

    logger.info("Training completed")

    # Plot and save loss graph
    plt.suptitle("VQ-VAE Loss")
    plt.plot(train_loss_log, label="train_loss")
    plt.plot(test_loss_log, label="test_loss")
    plt.grid(axis="y")
    plt.legend()
    loss_plot_path = os.path.join(args.output_dir, "VQVAE_loss.png")
    plt.savefig(loss_plot_path)
    logger.info(f"Saved loss plot to {loss_plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
