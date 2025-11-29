import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from models.image128.dataset_image128 import GrayscaleImage128Dataset
from models.image128.v_autoencoder_128 import VariationalAutoencoder128, train_vae_128

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load dataset
    dataset = GrayscaleImage128Dataset(args.image_dir)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    logger.info(f"Loaded {len(dataset)} images from {args.image_dir}")

    # Initialize or load model
    if Path(args.ckpt).is_file():
        logger.info(f"Loading checkpoint: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device)
        latent_dims = checkpoint["latent_dims"]
        vae = VariationalAutoencoder128(latent_dims).to(device)
        vae.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
    else:
        logger.info(f"Initializing new model with latent_dims={args.latent_dims}")
        vae = VariationalAutoencoder128(args.latent_dims).to(device)
        start_epoch = 0

    # Train
    if args.train:
        vae = train_vae_128(vae, data_loader, args.epochs, start_epoch)

    # Generate samples
    vae.eval()
    with torch.no_grad():
        # Random sampling from latent space
        z = torch.randn(16, args.latent_dims).to(device)
        samples = vae.decode(z)

        # Plot
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i, 0].cpu(), cmap="gray")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig("vae128_samples.png")
        logger.info("Saved samples to vae128_samples.png")


def prepare_argparse():
    parser = argparse.ArgumentParser(description="VAE for 128x128 grayscale images")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--latent_dims", type=int, default=64, help="Latent dimensions")
    parser.add_argument("--ckpt", type=str, default="", help="Checkpoint path")
    parser.add_argument("--train", action="store_true", help="Train the model")
    return parser


if __name__ == "__main__":
    parser = prepare_argparse()
    args = parser.parse_args()
    main(args)
