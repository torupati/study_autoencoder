"""
Variational Autoencoder for MNIST dataset

This is based on (almost copied from) this awsome page.
https://avandekleut.github.io/vae/
"""
from pathlib import Path
from os import path

import numpy as np

import torch;
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision

import argparse

import matplotlib.pyplot as plt;

from models.v_autoencoder import VariationalAutoencoder, train_vae
from models.mnist_utils import plot_latent_each_digit, plot_latent, plot_reconstructed

import logging
logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
                    handlers=[logging.FileHandler("vae_mnist.log", mode='w'),
                              stream_handler])

def plot_latent_each_digit0(autoencoder, data):
    """
    Plot latent variable z in plane (1st and 2nd dim by default)
    """
    _device = next(autoencoder.parameters()).device.type
    fig, ax = plt.subplots(2, 5)
    fig.suptitle('Sample Projection on Latent Space')
    for y_digit in range(0, 10):
        ax_i, ax_j = y_digit//5, y_digit%5
        print(ax_i, ax_j)
        for i, (x, y) in enumerate(data):
            x = x[y==y_digit]
            y = y[y==y_digit]
            z = autoencoder.encoder(x.to(_device))
            z = z.to('cpu').detach().numpy()
            im = ax[ax_i, ax_j].scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        ax[ax_i, ax_j].grid(True)
        ax[ax_i, ax_j].set_xlim([-3, 3])
        ax[ax_i, ax_j].set_ylim([-3, 3])
        ax[ax_i, ax_j].set_aspect('equal')
        ax[ax_i, ax_j].set_xlabel(r'$z_1$')
        ax[ax_i, ax_j].set_ylabel(r'$z_2$')
        ax[ax_i, ax_j].set_title('{0}'.format(y_digit))
    return fig


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('device=%s is used for pytorch', device)

    # Prepare dataset (MNIST) and dataloader
    dataset = torchvision.datasets.MNIST(args.dataset_dir,
            transform=torchvision.transforms.ToTensor(),
            download=True)
    obs_dim = 28*28
    logger.debug('prepare MNIST dataset from torchvision')
    logger.debug('%s', dataset)

    data = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    logger.info('data logader. batch size={i}'.format(i=args.batch_size))

    if path.isfile(args.ckpt):
        ckpt_file = args.ckpt
        logger.info('load from checkpoint: %s', ckpt_file)
        ckpts = torch.load(ckpt_file, weights_only=True)
        print(ckpts.keys())
        _dim = ckpts['latent_dims']
        vae = VariationalAutoencoder(_dim, obs_dim)
        vae.load_state_dict(ckpts['model_state_dict'])
        _cur_epoch = ckpts['epoch']
        logger.info('resume: %s', args.resume)
        if args.resume:
            vae = train_vae(vae, data, args.num_epoch, _cur_epoch, True)
    else:
        latent_dims = args.latent_dims
        logger.debug('latent dimension=%d', latent_dims)
        vae = VariationalAutoencoder(latent_dims, obs_dim).to(device)
        vae = train_vae(vae, data, args.num_epoch, 0, True)

    pngfile = 'vae_latent_each_digit.png'
    if Path(pngfile).exists():
        logger.warning(f'{pngfile=} exists. Skip plot latent digit')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax = plot_latent_each_digit(ax, vae, dataset)
        fig.savefig(pngfile)
        logger.info(f'save {pngfile=}')

    pngfile = 'vae_latent_space_projection.png'
    if Path(pngfile).exists():
        logger.warning(f'{pngfile=} exists. Training data projection onto latent space.')
    else:  
        fig, ax = plt.subplots(1, 1)
        ax = plot_latent(ax, vae, data, 1000)
        fig.savefig()
        logger.info(f'save {pngfile=}')

    pngfile = 'vae_reconstructed.png'
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    ax = plot_reconstructed(ax, vae, r0=(-3, 3), r1=(-3, 3))
    fig.savefig(pngfile)

    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    ax = plot_reconstructed(ax, vae, r0=(-1, 1), r1=(-1, 1))
    fig.savefig('vae_reconstructed2.png')

def prepare_argparse():
    parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoder for MNIST dataset')
    # training condition
    parser.add_argument('--dataset_dir', default='./data/', type=str, help='MNIST dataset directory')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size of data loader')
    parser.add_argument('--num_epoch', '-n', default=40, type=int, help='number of epoch for training')
    # model initialization/loading
    parser.add_argument('--latent_dims', '-d', default=2, type=int, help='Latent variable dimension. Ignored when pretrained model is loaded')
    parser.add_argument('--resume',  '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--ckpt', '-c', default='', type=str, help='checkpoint file path')
    return parser

if __name__ == "__main__":
    parser = prepare_argparse()
    args = parser.parse_args()
    main(args)
