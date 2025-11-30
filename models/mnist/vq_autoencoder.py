"""
Vector Quantized Variational Autoencoder (VQ-VAE) for MNIST dataset.

This implementation is based on the VQ-VAE paper:
"Neural Discrete Representation Learning" (van den Oord et al., 2017)
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer that maps continuous latent vectors to discrete codebook entries.

    Args:
        num_embeddings: Size of the codebook (number of discrete codes)
        embedding_dim: Dimension of each code vector
        commitment_cost: Weight for the commitment loss term
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        """
        Args:
            z: Continuous latent tensor of shape (batch_size, embedding_dim)

        Returns:
            dict containing:
                - 'z_q': Quantized latent vectors
                - 'loss': VQ loss (codebook + commitment)
                - 'encoding_indices': Indices of selected codebook entries
        """
        # Calculate distances between z and codebook entries
        # z: (batch_size, embedding_dim)
        # embedding.weight: (num_embeddings, embedding_dim)
        distances = (
            torch.sum(z**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z, self.embedding.weight.t())
        )

        # Get nearest codebook entry indices
        encoding_indices = torch.argmin(distances, dim=1)

        # Quantize: look up the codebook
        z_q = self.embedding(encoding_indices)

        # Calculate VQ loss
        # Codebook loss: move codebook entries towards encoder outputs
        codebook_loss = F.mse_loss(z_q.detach(), z)
        # Commitment loss: encourage encoder outputs to stay close to codebook
        commitment_loss = F.mse_loss(z_q, z.detach())

        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: copy gradients from decoder input to encoder output
        z_q = z + (z_q - z).detach()

        return {"z_q": z_q, "loss": loss, "encoding_indices": encoding_indices}


class VQAutoencoder(nn.Module):
    """
    Vector Quantized Autoencoder for MNIST.

    Architecture:
        Encoder: obs_dim -> 512 -> embedding_dim
        VQ: embedding_dim -> embedding_dim (quantized)
        Decoder: embedding_dim -> 512 -> obs_dim
    """

    class Encoder(nn.Module):
        def __init__(self, embedding_dim: int, obs_dim: int):
            super(VQAutoencoder.Encoder, self).__init__()
            self.linear1 = nn.Linear(obs_dim, 512)
            self.linear2 = nn.Linear(512, embedding_dim)

        def forward(self, x):
            x = F.relu(self.linear1(x))
            x = self.linear2(x)
            return x

    class Decoder(nn.Module):
        def __init__(self, embedding_dim: int, obs_dim: int):
            super(VQAutoencoder.Decoder, self).__init__()
            self.linear1 = nn.Linear(embedding_dim, 512)
            self.linear2 = nn.Linear(512, obs_dim)

        def forward(self, z):
            z = F.relu(self.linear1(z))
            z = torch.sigmoid(self.linear2(z))
            return z

    def __init__(
        self,
        embedding_dim: int,
        obs_dim: int,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
    ):
        """
        Args:
            embedding_dim: Dimension of the latent space
            obs_dim: Dimension of the observation (784 for MNIST)
            num_embeddings: Size of the codebook
            commitment_cost: Weight for the commitment loss
        """
        super(VQAutoencoder, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.encoder = VQAutoencoder.Encoder(embedding_dim, obs_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = VQAutoencoder.Decoder(embedding_dim, obs_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        z_e = self.encoder(x)
        vq_output = self.vq(z_e)
        z_q = vq_output["z_q"]
        x_hat = self.decoder(z_q)
        return x_hat, vq_output

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def num_embeddings(self) -> int:
        return self._num_embeddings


def evaluate_vqvae(autoencoder, data):
    """Evaluate VQ-VAE on given data and return average loss.

    Args:
        autoencoder: VQAutoencoder model
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
            x_hat, vq_output = autoencoder(x)
            recon_loss = ((x - x_hat) ** 2).sum()
            vq_loss = vq_output["loss"]
            loss = recon_loss + vq_loss
            total_loss += loss.item()
            num_batches += 1
    autoencoder.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def train_vqvae(
    autoencoder,
    data,
    epochs: int = 20,
    start_epoch: int = 0,
    save_model_train: bool = False,
    test_data=None,
):
    """Train VQ-VAE on given data.

    Args:
        autoencoder: VQAutoencoder model
        data: DataLoader for training
        epochs: Number of epochs to train
        start_epoch: Starting epoch number
        save_model_train: Whether to save checkpoints during training
        test_data: Optional DataLoader for test set evaluation

    Returns:
        Trained VQAutoencoder model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    opt = torch.optim.Adam(autoencoder.parameters())

    for epoch in range(start_epoch, epochs):
        autoencoder.train()
        total_loss, num_samples = 0.0, 0
        tbar = tqdm(data)

        for batch_idx, (x, _) in enumerate(tbar):
            x = x.to(device)
            opt.zero_grad()

            x = torch.flatten(x, start_dim=1)
            x_hat, vq_output = autoencoder(x)

            # Reconstruction loss + VQ loss
            recon_loss = ((x - x_hat) ** 2).sum()
            vq_loss = vq_output["loss"]
            loss = recon_loss + vq_loss

            loss.backward()
            opt.step()

            total_loss += loss.item()
            num_samples += 1

            error_str = "Epoch: %d, loss=%.4f" % (epoch, total_loss / (batch_idx + 1))
            tbar.set_description(error_str)

        # Calculate test loss if test data is provided
        if test_data is not None:
            test_loss = evaluate_vqvae(autoencoder, test_data)
            logger.info(
                f"Epoch {epoch}: train_loss={total_loss / num_samples:.4f}, "
                f"test_loss={test_loss:.4f}"
            )
        else:
            logger.debug(f"epoch {epoch} loss {total_loss / num_samples:.4f}")

        if epoch % 10 == 0 and save_model_train:
            _filename = f"checkpoint_vqvae_train_epoch{epoch:03d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": autoencoder.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "embedding_dim": autoencoder.embedding_dim,
                    "num_embeddings": autoencoder.num_embeddings,
                    "loss": total_loss,
                },
                _filename,
            )
            logger.info("save log file %s", _filename)

    return autoencoder
