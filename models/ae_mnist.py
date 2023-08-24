"""
Autoencoder for MNIST dataset
(C) torupati laboratory
"""

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import pickle
import matplotlib.pyplot as plt
from os import path
from tqdm import tqdm
import argparse
import logging
logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
                    handlers=[logging.FileHandler("vae_mnist.log", mode='w'),
                              stream_handler])

class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(784, 512, bias=True)
        self.linear2 = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))

class Autoencoder(nn.Module):
    def __init__(self, latent_dims:int):
        super(Autoencoder, self).__init__()
        print('init latent_dims=', latent_dims)
        self._latent_dims = latent_dims
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    @property
    def latent_dims(self) -> int:
        return self._latent_dims


def train(autoencoder, data, epochs=20, start_epoch:int = 0):
    _device = next(autoencoder.parameters()).device.type
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        train_loss, num_samples = 0.0, 0
        tbar = tqdm(data)
        for batch_idx, (x, y) in enumerate(tbar):
            x = x.to(_device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum()
            loss.backward()
            opt.step()

            train_loss += loss.item()
            num_samples += 1
            error_str = 'Epoch: %d, loss=%.4f' % (epoch, train_loss / (batch_idx + 1))
            tbar.set_description(error_str)
        if epoch % 10 == 0:
            _filename = f"ckpt_mnist_autoencoder_train_epoch{epoch:03d}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': autoencoder.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'latent_dims': autoencoder.latent_dims, 
                'loss': train_loss},
                _filename)
            logger.info('save log file %s', _filename)

    return autoencoder


def plot_latent(autoencoder, data, num_batches=100):
    """
    Plot latent variable z in plane (1st and 2nd dim by default)
    """
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('device=%s is used for pytorch', device)

    # Prepare dataset (MNIST) and dataloader
    dataset = torchvision.datasets.MNIST(args.dataset_dir,
            transform=torchvision.transforms.ToTensor(),
            download=True)
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
        ae = Autoencoder(_dim)
        ae.load_state_dict(ckpts['model_state_dict'])
        _cur_epoch = ckpts['epoch']
        if args.resume:
            vae = train(vae, data, args.num_epoch, _cur_epoch)
    else:
        latent_dims = args.latent_dims
        logger.info('initialize autoencoder, latent dimension=%d', latent_dims)
        ae = Autoencoder(latent_dims).to(device)
        ae = train(ae, data, args.num_epoch, 0)

    pngfile = 'ae_latent_each_digit.png'
    fig = plot_latent_each_digit(vae, data)
    fig.savefig(pngfile)
    logger.info('save {0}'.format(pngfile))

    fig = plot_latent(vae, data, 1000)
    fig.savefig('ae_latent_space_projection.png')

    fig = plot_reconstructed(vae, r0=(-3, 3), r1=(-3, 3))
    fig.savefig('ae_reconstructed.png')


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

