"""
Vector Quantized Variational Autoencoder (VQ-VAE) implementation for MNIST.

References:
- Neural Discrete Representation Learning (van den Oord et al., 2017)
- https://arxiv.org/abs/1711.00937
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Get device (will be set by the caller)
device = "cpu"


class ResidualLayer(nn.Module):
    """Residual layer for use in encoder/decoder stacks."""

    def __init__(self, in_dim, h_dim, res_h_dim):
        """
        Args:
            in_dim: Input dimension
            h_dim: Hidden dimension
            res_h_dim: Residual hidden dimension
        """
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        """Apply residual connection."""
        return x + self.res_block(x)


class ResidualStack(nn.Module):
    """Stack of residual layers."""

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        """
        Args:
            in_dim: Input dimension
            h_dim: Hidden dimension
            res_h_dim: Residual hidden dimension
            n_res_layers: Number of residual layers to stack
        """
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList([ResidualLayer(in_dim, h_dim, res_h_dim)] * n_res_layers)

    def forward(self, x):
        """Pass input through all residual layers."""
        for layer in self.stack:
            x = layer(x)
        return F.relu(x)


class Encoder(nn.Module):
    """Convolutional encoder for VQ-VAE."""

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        """
        Args:
            in_dim: Number of input channels (1 for grayscale)
            h_dim: Hidden dimension
            n_res_layers: Number of residual layers
            res_h_dim: Residual hidden dimension
        """
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        )

    def forward(self, x):
        """Encode input to feature space."""
        return self.conv_stack(x)


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer module for VQ-VAE.

    This module implements the core quantization mechanism that maps continuous
    encoder outputs to a discrete latent space by finding the nearest embedding vector.

    The embedding vectors form a learnable codebook (dictionary) that represents
    the discrete latent space. Each encoding from the encoder is quantized to the
    nearest embedding vector in this codebook.

    Attributes:
        beta (float): Weight for the commitment loss
        embedding (nn.Embedding): Codebook containing all embedding vectors
                                  Shape: [n_e, e_dim]
                                  This is the set of vectors in the latent space.
    """

    def __init__(self, n_e, e_dim, beta):
        """
        Initialize the Vector Quantizer.

        Args:
            n_e (int): Number of embedding vectors in the codebook (e.g., 512)
            e_dim (int): Dimension of each embedding vector (e.g., 64)
            beta (float): Weight for commitment loss (typically 0.25)
        """
        super(VectorQuantizer, self).__init__()
        self.beta = beta

        # Create the codebook: a learnable embedding matrix
        # Shape: [n_e, e_dim] - This is the set of discrete vectors in the latent space
        self.embedding = nn.Embedding(n_e, e_dim)
        # Initialize embedding vectors uniformly
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    @property
    def n_e(self) -> int:
        """Read-only property: Number of embedding vectors (codebook size)."""
        return int(self.embedding.weight.shape[0])

    @property
    def e_dim(self) -> int:
        """Read-only property: Dimension of each embedding vector."""
        return int(self.embedding.weight.shape[1])

    def forward(self, z):
        """
        Quantize continuous latent codes to the nearest embedding vectors.

        Args:
            z (Tensor): Continuous latent codes from encoder
                       Shape: [batch_size, e_dim, height, width]

        Returns:
            _loss (Tensor): VQ loss (commitment loss + embedding loss)
            z_q (Tensor): Quantized latent codes
                         Shape: [batch_size, e_dim, height, width]
            min_encodings (Tensor): One-hot encoded indices of nearest embeddings
            min_encoding_indices (Tensor): Indices of nearest embedding vectors

        Process:
            1. Reshape z for distance calculation
            2. Compute Euclidean distance from each z to all embedding vectors
            3. Find nearest embedding for each z element
            4. Replace z with quantized z_q (nearest embedding)
            5. Compute VQ losses to optimize both encoder and codebook
        """
        # Permute z from [B, C, H, W] to [B, H, W, C] for processing
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # Calculate Euclidean distance between z_flattened and all embedding vectors
        # d[i,j] = ||z_i - e_j||^2
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # Find the index of the nearest embedding vector for each z
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # Get the quantized values (nearest embedding vectors)
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # Compute VQ loss: commitment loss + embedding loss
        # Commitment loss: encourages encoder output to stay close to chosen embedding
        # Embedding loss: encourages embedding vectors to move towards encoder outputs
        _loss = torch.mean((z.detach() - z_q) ** 2) + self.beta * torch.mean(
            (z - z_q.detach()) ** 2
        )

        # Straight-through estimator: copy gradients from z_q to z for backprop
        z_q = z + (z_q - z).detach()

        # Permute back from [B, H, W, C] to [B, C, H, W]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return _loss, z_q, min_encodings, min_encoding_indices


class Decoder(nn.Module):
    """Convolutional decoder for VQ-VAE."""

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        """
        Args:
            in_dim: Number of input channels (embedding dimension)
            h_dim: Hidden dimension
            n_res_layers: Number of residual layers
            res_h_dim: Residual hidden dimension
        """
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2
        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim // 2, 1, kernel_size=kernel, stride=stride, padding=1),
        )

    def forward(self, x):
        """Decode latent codes to image space."""
        return self.inverse_conv_stack(x)


class VQVAE(nn.Module):
    """
    Vector Quantized Variational Autoencoder for MNIST.

    Architecture:
        Input → Encoder → Pre-quantization Conv → VectorQuantizer → Decoder → Output
    """

    def __init__(self, h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta):
        """
        Initialize VQ-VAE model.

        Args:
            h_dim (int): Hidden dimension for encoder/decoder (e.g., 128)
            res_h_dim (int): Residual hidden dimension (e.g., 32)
            n_res_layers (int): Number of residual layers (e.g., 2)
            n_embeddings (int): Number of embedding vectors (codebook size, e.g., 512)
            embedding_dim (int): Dimension of each embedding vector (e.g., 64)
            beta (float): Weight for commitment loss (e.g., 0.25)
        """
        super(VQVAE, self).__init__()
        self.encoder = Encoder(1, h_dim, n_res_layers, res_h_dim)
        # Pre-quantization conv: projects encoder output from h_dim to embedding_dim
        # Shape: [batch_size, h_dim, H, W] → [batch_size, embedding_dim, H, W]
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

    def forward(self, x):
        """
        Forward pass through VQ-VAE.

        Args:
            x (Tensor): Input images, shape [batch_size, 1, 28, 28]

        Returns:
            embedding_loss (Tensor): VQ loss for optimization
            x_hat (Tensor): Reconstructed images, shape [batch_size, 1, 28, 28]
        """
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)
        return embedding_loss, x_hat
