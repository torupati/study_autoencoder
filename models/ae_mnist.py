"""
This is based on (almost copied from) this awsome page.

https://avandekleut.github.io/vae/


todo:
https://pytorch.org/docs/stable/notes/multiprocessing.html
"""

import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import pickle
import matplotlib.pyplot as plt;


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
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)
        self._latent_dims = latent_dims

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    @property
    def latent_dims(self) -> int:
        return self._latent_dims


def train(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        for x, y in data:
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum()
            loss.backward()
            opt.step()
        print(epoch, torch.norm(loss))
        if epoch % 10 == 0 and save_model_train:
            _filename = f"ckpt_mnist_autoencoder_train_epoch{epoch:03d}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': autoencoder.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'latent_dims': autoencoder.latent_dims, 
                'loss': total_loss},
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

def train_autoencoder(data, latent_dim:int):
    autoencoder = Autoencoder(latent_dims).to(device) # GPU
    autoencoder = train(autoencoder, data)

    plot_latent(autoencoder, data)
    plt.savefig('out.png')

    plot_reconstructed(autoencoder)
    plt.savefig('autoencoder_figures.png')


#with open("checkpoint_name.pkl", "wb") as cp_file:
#    cp = dict(autoencoder = autoencoder)
#    pickle.dump(cp, cp_file)

def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])
#
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

def train_vae(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        print('epoch={i}'.format(i=epoch))
        for x, y in data:
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
    return autoencoder


if __name__ == "__main__":
    #plt.rcParams['figure.dpi'] = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    do_train = True

    # Prepare dataset (MNIST) and dataloader
    dataset = torchvision.datasets.MNIST('./data',
            transform=torchvision.transforms.ToTensor(),
            download=True)
    print(dataset)
    data = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    # Prepare Model.
    latent_dims = 2 # 2
    #train_autoencoder(data, latent_dim)

    vae = VariationalAutoencoder(latent_dims).to(device)

    if do_train:
        vae = train_vae(vae, data)

    plot_latent(vae, data)
    plt.savefig('vae_2d.png')
    plt.close()
    plot_reconstructed(vae, r0=(-3, 3), r1=(-3, 3))
    plt.savefig('vae2.png')

