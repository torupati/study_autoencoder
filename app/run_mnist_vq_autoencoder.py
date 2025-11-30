"""
Vector Quantized Variational Autoencoder (VQ-VAE) for MNIST dataset
"""

import argparse
import logging
from os import path
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.utils

torch.manual_seed(0)

from models.mnist.dataset_mnist import get_mnist_dataset
from models.mnist.mnist_utils import plot_latent, plot_latent_each_digit, plot_reconstructed
from models.mnist.vq_autoencoder import VQAutoencoder, train_vqvae

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s",
    handlers=[logging.FileHandler("vqvae_mnist.log", mode="w"), stream_handler],
)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device=%s is used for pytorch", device)

    # Prepare dataset (MNIST) and dataloader
    train_dataset = get_mnist_dataset(args.dataset_dir, train=True)
    test_dataset = get_mnist_dataset(args.dataset_dir, train=False)
    obs_dim = 28 * 28
    logger.debug("prepare MNIST dataset from custom implementation")
    logger.debug("%s", train_dataset)

    train_data = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_data = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    logger.info(f"data loader. batch size={args.batch_size}")

    if path.isfile(args.ckpt):
        ckpt_file = args.ckpt
        logger.info("load from checkpoint: %s", ckpt_file)
        ckpts = torch.load(ckpt_file, weights_only=True)
        print(ckpts.keys())
        _embedding_dim = ckpts["embedding_dim"]
        _num_embeddings = ckpts["num_embeddings"]
        vqvae = VQAutoencoder(_embedding_dim, obs_dim, _num_embeddings)
        vqvae.load_state_dict(ckpts["model_state_dict"])
        _cur_epoch = ckpts["epoch"]
        logger.info("resume: %s", args.resume)
        if args.resume:
            vqvae = train_vqvae(
                vqvae, train_data, args.num_epoch, _cur_epoch, True, test_data=test_data
            )
    else:
        embedding_dim = args.embedding_dim
        num_embeddings = args.num_embeddings
        logger.debug("embedding dimension=%d, num embeddings=%d", embedding_dim, num_embeddings)
        vqvae = VQAutoencoder(embedding_dim, obs_dim, num_embeddings).to(device)
        vqvae = train_vqvae(vqvae, train_data, args.num_epoch, 0, True, test_data=test_data)

    pngfile = "vqvae_latent_each_digit.png"
    if Path(pngfile).exists():
        logger.warning(f"{pngfile=} exists. Skip plot latent digit")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax = plot_latent_each_digit(ax, vqvae, train_dataset)
        fig.savefig(pngfile)
        logger.info(f"save {pngfile=}")

    pngfile = "vqvae_latent_space_projection.png"
    if Path(pngfile).exists():
        logger.warning(f"{pngfile=} exists. Training data projection onto latent space.")
    else:
        fig, ax = plt.subplots(1, 1)
        im = plot_latent(ax, vqvae, train_dataset, 1000)
        fig.colorbar(im, ax=ax, label="Digit")
        fig.savefig(pngfile)
        logger.info(f"save {pngfile=}")

    pngfile = "vqvae_reconstructed.png"
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = plot_reconstructed(ax, vqvae, r0=(-3, 3), r1=(-3, 3))
    fig.colorbar(im, ax=ax)
    fig.savefig(pngfile)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = plot_reconstructed(ax, vqvae, r0=(-1, 1), r1=(-1, 1))
    fig.colorbar(im, ax=ax)
    fig.savefig("vqvae_reconstructed2.png")


def prepare_argparse():
    parser = argparse.ArgumentParser(
        description="PyTorch Vector Quantized Variational Autoencoder (VQ-VAE) for MNIST dataset"
    )
    # training condition
    parser.add_argument(
        "--dataset_dir", default="./data/MNIST/", type=str, help="MNIST dataset directory"
    )
    parser.add_argument("--batch_size", default=128, type=int, help="batch size of data loader")
    parser.add_argument(
        "--num_epoch", "-n", default=40, type=int, help="number of epoch for training"
    )
    # model initialization/loading
    parser.add_argument(
        "--embedding_dim",
        "-d",
        default=64,
        type=int,
        help="Embedding dimension for latent space. Ignored when pretrained model is loaded",
    )
    parser.add_argument(
        "--num_embeddings",
        "-k",
        default=512,
        type=int,
        help="Number of embeddings in codebook. Ignored when pretrained model is loaded",
    )
    parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
    parser.add_argument("--ckpt", "-c", default="", type=str, help="checkpoint file path")
    return parser


if __name__ == "__main__":
    parser = prepare_argparse()
    args = parser.parse_args()
    main(args)
