"""
Autoencoder for MNIST dataset
"""
from os import path
import argparse
import logging

import torch
torch.manual_seed(0)
import torch.utils
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.autoencoder import Autoencoder, train
from models.mnist_utils import plot_latent_each_digit, plot_latent, plot_reconstructed

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
                    handlers=[logging.FileHandler("vae_mnist.log", mode='a'),
                              stream_handler])

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
        ckpts = torch.load(ckpt_file, weights_only = True)
        print(ckpts.keys())
        _dim = ckpts['latent_dim']
        ae = Autoencoder(_dim, obs_dim)
        ae.load_state_dict(ckpts['model_state_dict'])
        _cur_epoch = ckpts['epoch']
        if args.resume:
            vae = train(vae, data, args.num_epoch, _cur_epoch)
    else:
        latent_dim = args.latent_dim
        logger.info('initialize autoencoder, latent dimension=%d', latent_dim)
        ae = Autoencoder(latent_dim, obs_dim).to(device)
        ae = train(ae, data, args.num_epoch, 0)

    fig, ax = plt.subplots(1, 1)
    ax = plot_latent_each_digit(ax, ae, dataset)
    pngfile = 'ae_latent_each_digit.png'
    fig.savefig(pngfile)
    logger.info('save {0}'.format(pngfile))

    fig, ax = plt.subplots(1, 1)
    ax = plot_latent(ax, ae, dataset, 100)
    fig.savefig('ae_latent_space_projection.png')

    fig, ax = plt.subplots(1, 1)
    ax = plot_reconstructed(ax, ae, r0=(-3, 3), r1=(-3, 3))
    fig.savefig('ae_reconstructed.png')


def prepare_argparse():
    parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoder for MNIST dataset')
    # training condition
    parser.add_argument('--dataset_dir', default='./data/', type=str, help='MNIST dataset directory')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size of data loader')
    parser.add_argument('--num_epoch', '-n', default=40, type=int, help='number of epoch for training')
    # model initialization/loading
    parser.add_argument('--latent_dim', '-d', default=2, type=int, help='Latent variable dimension. Ignored when pretrained model is loaded')
    parser.add_argument('--resume',  '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--ckpt', '-c', default='', type=str, help='checkpoint file path')
    return parser

if __name__ == "__main__":
    parser = prepare_argparse()
    args = parser.parse_args()
    main(args)
