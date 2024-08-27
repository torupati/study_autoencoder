"""
Variational Autoencoder for MNIST dataset

This is based on (almost copied from) this awsome page.
https://avandekleut.github.io/vae/
"""
import numpy as np
from os import path

import torch;
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
from tqdm import tqdm

import argparse

import matplotlib.pyplot as plt;

from models.v_autoecoder import VariationalAutoencoder, train_vae

import logging
logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
                    handlers=[logging.FileHandler("vae_mnist.log", mode='w'),
                              stream_handler])

def plot_latent(autoencoder, data, num_lim=100):
    """
    Plot latent variable z in plane (1st and 2nd dim by default)
    """
    _device = next(autoencoder.parameters()).device.type
    fig, ax = plt.subplots(1, 1)
    ax.set_title('Sample Projection on Latent Space')
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(_device))
        z = z.to('cpu').detach().numpy()
        im = ax.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_lim:
            break
    ax.grid(True)
    ax.set_xlabel(r'$z_1$')
    ax.set_ylabel(r'$z_2$')
    #fig.colorbar(im, ax=ax)
    return fig

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



def plot_reconstructed(ax, autoencoder, r0=(-5, 10), r1=(-10, 5), num_img=12):
    """Plot reconstructed image x from z on grid

    Args:
        autoencoder (_type_): _description_
        r0 (tuple, optional): range of latent variable z1. Defaults to (-5, 10).
        r1 (tuple, optional): range of latent variable z2. Defaults to (-10, 5).
        num_img (int, optional): _description_. Defaults to 12.
    """
    w = 28
    img = np.zeros((num_img*w, num_img*w))
    _device = next(autoencoder.parameters()).device.type
    for i, y in enumerate(np.linspace(*r1, num_img)):
        for j, x in enumerate(np.linspace(*r0, num_img)):
            z = torch.Tensor([[x, y]]).to(_device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(w, w).to('cpu').detach().numpy()
            img[(num_img-1-i)*w:(num_img-1-i+1)*w, j*w:(j+1)*w] = x_hat
    im = ax.imshow(img, extent=[*r0, *r1])
    ax.set_xlabel(r'$z_1$')
    ax.set_ylabel(r'$z_2$')
    return ax

# multi-processing
# https://pytorch.org/docs/stable/notes/multiprocessing.html




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
        ckpts = torch.load(ckpt_file)
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
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax = plot_latent_each_digit(ax, vae, dataset)
    fig.savefig(pngfile)
    logger.info('save {0}'.format(pngfile))

    fig = plot_latent(vae, data, 1000)
    fig.savefig('vae_latent_space_projection.png')

    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    ax = plot_reconstructed(ax, vae, r0=(-3, 3), r1=(-3, 3))
    fig.savefig('vae_reconstructed.png')

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
