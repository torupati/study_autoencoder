import logging

import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VariationalAutoencoder(nn.Module):
    class Encoder(nn.Module):
        def __init__(self, latent_dims: int, obs_dim: int):
            super(VariationalAutoencoder.Encoder, self).__init__()
            self.linear1 = nn.Linear(obs_dim, 512)
            self.linear2 = nn.Linear(512, latent_dims)
            self.linear3 = nn.Linear(512, latent_dims)

            self.N = torch.distributions.Normal(0, 1)  # Gaussian Distribution for additive noise
            if torch.cuda.is_available():
                self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
                self.N.scale = self.N.scale.cuda()
            self.kl = 0

        def forward(self, x):
            """
            Args
            ----
            x(torch.Tensor): input data

            Returns
            -------
            d(dict): latent variable z and model parameter of p(z|x)
            'z', 'mu', and 'sigma' is included. These are torch.Tensor instances.

            Note
            ----
            This module returns dictionary instead of torch.Tensor.
            """
            x = torch.flatten(x, start_dim=1)
            x = F.relu(self.linear1(x))
            mu = self.linear2(x)
            sigma = torch.exp(self.linear3(x))
            z = mu + sigma * self.N.sample(mu.shape)
            self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1 / 2).sum()
            return {"z": z, "mu": mu, "sigma": sigma}

    @staticmethod
    class Decoder(nn.Module):
        def __init__(self, latent_dims: int, obs_dim: int):
            super(VariationalAutoencoder.Decoder, self).__init__()
            self.linear1 = nn.Linear(latent_dims, 512)
            self.linear2 = nn.Linear(512, obs_dim)

        def forward(self, z):
            z = F.relu(self.linear1(z))
            z = torch.sigmoid(self.linear2(z))
            return z

        # return z.reshape((-1, 1, 28, 28))

    def __init__(self, latent_dims: int, obs_dim: int):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalAutoencoder.Encoder(latent_dims, obs_dim)
        self.decoder = VariationalAutoencoder.Decoder(latent_dims, obs_dim)
        self._latent_dims = latent_dims

    def forward(self, x):
        _ret = self.encoder(x)
        z = _ret["z"]
        x_hat = self.decoder(z)
        return x_hat

    @property
    def latent_dims(self) -> int:
        """Get latent dimension. Note this dimension cannot be changed after the instanciation.

        Returns:
            int: dimension of latent space
        """
        return self._latent_dims


def evaluate_vae(autoencoder, data):
    """Evaluate VAE on given data and return average loss.

    Args:
        autoencoder: VariationalAutoencoder model
        data: DataLoader for evaluation

    Returns:
        float: Average loss over the dataset
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autoencoder.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for x, _ in data:
            x = x.to(device)
            x = torch.flatten(x, start_dim=1)
            x_hat = autoencoder(x)
            loss = ((x - x_hat) ** 2).sum() + autoencoder.encoder.kl
            total_loss += loss.item()
            num_batches += 1
    autoencoder.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def train_vae(
    autoencoder, data, epochs=20, start_epoch: int = 0, save_model_train: bool = False, test_data=None
):
    """Training VAE from given data

    Args:
        autoencoder (VariationalAutoencoder): instance of VAE
        data (torch.DataLoader): pytorch data loader
        epochs (int, optional): Number of epochs to train. Defaults to 20.
        start_epoch (int, optional): Starting epoch number. Defaults to 0.
        save_model_train (bool, optional): Save model checkpoints during training. Defaults to False.
        test_data (torch.DataLoader, optional): DataLoader for test set evaluation. Defaults to None.

    Returns:
        VariationalAutoencoder: Trained VAE model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(start_epoch, epochs):
        autoencoder.train()
        total_loss, num_samples = 0.0, 0
        tbar = tqdm(data)
        for batch_idx, (x, y) in enumerate(tbar):
            x = x.to(device)  # GPU
            opt.zero_grad()
            x = torch.flatten(x, start_dim=1)
            x_hat = autoencoder(x)
            loss = ((x - x_hat) ** 2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
            total_loss += loss.item()
            num_samples += 1

            error_str = "Epoch: %d, loss=%.4f" % (epoch, total_loss / (batch_idx + 1))
            tbar.set_description(error_str)

        # Calculate test loss if test data is provided
        if test_data is not None:
            test_loss = evaluate_vae(autoencoder, test_data)
            logger.info(
                f"Epoch {epoch}: train_loss={total_loss / num_samples:.4f}, "
                f"test_loss={test_loss:.4f}"
            )
        else:
            logger.debug(f"epoch {epoch} loss {total_loss / num_samples:.4f}")

        if epoch % 10 == 0 and save_model_train:
            _filename = f"checkpoint_vae_train_epoch{epoch:03d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": autoencoder.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "latent_dims": autoencoder.latent_dims,
                    "loss": total_loss,
                },
                _filename,
            )
            logger.info("save log file %s", _filename)
    return autoencoder
