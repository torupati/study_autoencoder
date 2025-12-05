import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    """
    Simple implementation of auto-encoder.

    Note
    ----
    This implementation assumes input features are normalized to the range [-1, 1].
    The decoder uses tanh activation to constrain outputs to [-1, 1], matching the input distribution.
    """

    @staticmethod
    class Encoder(nn.Module):
        """
        Encoder which takes observation as input and output latent variable.
        """

        def __init__(self, latent_dims: int, obs_dim: int):
            super(Autoencoder.Encoder, self).__init__()
            self.linear1 = nn.Linear(obs_dim, 512, bias=True)
            self.linear2 = nn.Linear(512, latent_dims)

        def forward(self, x):
            x = F.relu(self.linear1(x))
            x = self.linear2(x)
            return x

        @property
        def in_features(self) -> int:
            return int(self.linear1.in_features)

    @staticmethod
    class Decoder(nn.Module):
        """
        Decoder which takes latent variable and reconstructs the input.
        Uses tanh activation to output values in [-1, 1] range.
        """

        def __init__(self, latent_dims: int, obs_dim: int):
            super(Autoencoder.Decoder, self).__init__()
            self.linear1 = nn.Linear(latent_dims, 512)
            self.linear2 = nn.Linear(512, obs_dim)

        def forward(self, z):
            z = F.relu(self.linear1(z))
            z = torch.tanh(self.linear2(z))  # Output in [-1, 1]
            return z

    def __init__(self, latent_dims: int, obs_dim: int):
        super(Autoencoder, self).__init__()
        print(f"init {latent_dims=} {obs_dim=}")
        self._latent_dims: int = latent_dims
        self.encoder = Autoencoder.Encoder(latent_dims, obs_dim)
        self.decoder = Autoencoder.Decoder(latent_dims, obs_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        z = self.encoder(x)
        return self.decoder(z)

    @property
    def latent_dim(self) -> int:
        return self._latent_dims


def evaluate(autoencoder, data):
    """Evaluate autoencoder on given data and return average loss.

    Args:
        autoencoder: Autoencoder model
        data: DataLoader for evaluation

    Returns:
        float: Average loss over the dataset
    """
    _device = next(autoencoder.parameters()).device.type
    autoencoder.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for x, _ in data:
            x = x.to(_device)
            x = torch.flatten(x, start_dim=1)
            x_hat = autoencoder(x)
            loss = ((x - x_hat) ** 2).sum()
            total_loss += loss.item()
            num_batches += 1
    autoencoder.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def train(autoencoder, data, epochs=20, start_epoch: int = 0, test_data=None):
    """Train autoencoder on given data.

    Args:
        autoencoder: Autoencoder model
        data: DataLoader for training
        epochs: Number of epochs to train
        start_epoch: Starting epoch number
        test_data: Optional DataLoader for test set evaluation

    Returns:
        Trained autoencoder model
    """
    _device = next(autoencoder.parameters()).device.type
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        autoencoder.train()
        train_loss, num_samples = 0.0, 0
        tbar = tqdm(data)
        for batch_idx, (x, _) in enumerate(tbar):
            x = x.to(_device)  # GPU
            opt.zero_grad()
            x = torch.flatten(x, start_dim=1)
            x_hat = autoencoder(x)
            loss = ((x - x_hat) ** 2).sum()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            num_samples += 1
            error_str = "Epoch: %d, loss=%.4f" % (epoch, train_loss / (batch_idx + 1))
            tbar.set_description(error_str)

        # Calculate test loss if test data is provided
        if test_data is not None:
            test_loss = evaluate(autoencoder, test_data)
            logger.info(
                f"Epoch {epoch}: train_loss={train_loss / num_samples:.4f}, "
                f"test_loss={test_loss:.4f}"
            )

        if epoch % 10 == 0:
            _filename = f"ckpt_mnist_autoencoder_train_epoch{epoch:03d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": autoencoder.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "latent_dim": autoencoder.latent_dim,
                    "loss": train_loss,
                },
                _filename,
            )
            logger.info("save log file %s", _filename)
    return autoencoder
