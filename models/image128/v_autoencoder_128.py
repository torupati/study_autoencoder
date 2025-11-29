import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VariationalAutoencoder128(nn.Module):
    """
    Variational Autoencoder for 128x128 grayscale images using CNN.

    Architecture:
        Encoder: 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8 -> latent
        Decoder: latent -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
    """

    def __init__(self, latent_dims: int = 64):
        super(VariationalAutoencoder128, self).__init__()
        self.latent_dims = latent_dims

        # Encoder
        self.encoder = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        # Latent space
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dims)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dims)

        # Decoder
        self.fc_decode = nn.Linear(latent_dims, 256 * 8 * 8)

        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # 64x64 -> 128x128
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def encode(self, x):
        """Encode input to latent parameters (mu, logvar)"""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        """Decode latent variable to image"""
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    @property
    def name(self):
        return f"VAE128_latent{self.latent_dims}"


def vae_loss(x_recon, x, mu, logvar):
    """
    VAE loss = Reconstruction loss + KL divergence

    Args:
        x_recon: Reconstructed images
        x: Original images
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
    """
    # Reconstruction loss (Binary Cross Entropy)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    batch_size = x.size(0)
    recon_loss /= batch_size
    kl_div /= batch_size
    return recon_loss + kl_div


def train_vae_128(
    vae: VariationalAutoencoder128,
    data_loader,
    epochs: int = 20,
    start_epoch: int = 0,
    save_checkpoint: bool = True,
):
    """
    Train VAE for 128x128 grayscale images

    Args:
        vae: VariationalAutoencoder128 model
        data_loader: DataLoader for training data
        epochs: Number of training epochs
        start_epoch: Starting epoch (for resuming training)
        save_checkpoint: Whether to save checkpoints
    """
    device = next(vae.parameters()).device
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)  # Lower learning rate

    vae.train()

    for epoch in range(start_epoch, start_epoch + epochs):
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{start_epoch + epochs}")

        for batch_idx, (x, _) in enumerate(progress_bar):
            x = x.to(device)

            # Forward pass
            x_recon, mu, logvar = vae(x)

            # Compute loss
            loss = vae_loss(x_recon, x, mu, logvar)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)

            optimizer.step()

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Loss became NaN/Inf at epoch {epoch + 1}, batch {batch_idx}")
                raise ValueError("Training diverged: loss is NaN or Inf")

            total_loss += loss.item()

            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        avg_epoch_loss = total_loss / len(data_loader.dataset)
        logger.info(f"Epoch {epoch + 1}: Average loss = {avg_epoch_loss:.4f}")

        # Save checkpoint
        if save_checkpoint and (epoch + 1) % 10 == 0:
            checkpoint_path = f"vae128_epoch{epoch + 1}.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": vae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_epoch_loss,
                    "latent_dims": vae.latent_dims,
                },
                checkpoint_path,
            )
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    return vae
