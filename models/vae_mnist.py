"""
This is based on (almost copied from) this awsome page.

https://avandekleut.github.io/vae/


todo:
https://pytorch.org/docs/stable/notes/multiprocessing.html
"""
import numpy as np

import torch;
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision

from ae_mnist import Decoder
from logging import getLogger, basicConfig, INFO

import matplotlib.pyplot as plt;


logger = getLogger(__name__)
import logging
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
                    handlers=[logging.FileHandler("my_log.log", mode='w'),
                              stream_handler])

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


def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), num_img=12):
    """Plot reconstructed image x from z on grid

    Args:
        autoencoder (_type_): _description_
        r0 (tuple, optional): range of latent variable z1. Defaults to (-5, 10).
        r1 (tuple, optional): range of latent variable z2. Defaults to (-10, 5).
        num_img (int, optional): _description_. Defaults to 12.
    """
    w = 28
    img = np.zeros((num_img*w, num_img*w))
    for i, y in enumerate(np.linspace(*r1, num_img)):
        for j, x in enumerate(np.linspace(*r0, num_img)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(num_img-1-i)*w:(num_img-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])

# multi-processing
# https://pytorch.org/docs/stable/notes/multiprocessing.html
#



class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        if torch.cuda.is_available():
            self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train_vae(autoencoder, data, epochs=20, save_model_train = False):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        logger.info('epoch={i}'.format(i=epoch))
        total_loss, num_samples = 0.0, 0
        for x, y in data:
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
            total_loss += loss.item()
            num_samples += 1
        logger.info('epoch {0} loss {1}'.format(epoch, total_loss/num_samples))
        if epoch % 10 == 0 and save_model_train:
            _filename = f"checkpoint_vae_train_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': autoencoder.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': total_loss},
                _filename)
            logger.info('save log file %s', _filename)
    return autoencoder


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('device=%s', device)

    do_train = True

    # Prepare dataset (MNIST) and dataloader
    dataset = torchvision.datasets.MNIST('./data',
            transform=torchvision.transforms.ToTensor(),
            download=True)
    logger.debug('prepare MNIST dataset from torchvision')
    logger.debug('%s', dataset)
    data = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    # Prepare Model.
    latent_dims = 2 # 2
    logger.debug('latent dimension=%d', latent_dims)

    vae = VariationalAutoencoder(latent_dims).to(device)

    if do_train:
        vae = train_vae(vae, data, 20, True)

    plot_latent(vae, data)
    plt.savefig('vae_2d.png')
    plot_reconstructed(vae, r0=(-3, 3), r1=(-3, 3))
    plt.savefig('vae2.png')

