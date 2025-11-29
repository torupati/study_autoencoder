"""
Unit tests for vqvae.py (Vector Quantized Variational Autoencoder)
"""

import pytest
import torch
import torch.nn as nn
from models.mnist.vqvae import (
    ResidualLayer,
    ResidualStack,
    Encoder,
    VectorQuantizer,
    Decoder,
    VQVAE,
)
import models.mnist.vqvae as vqvae_module


class TestResidualLayer:
    """Test cases for the ResidualLayer."""

    @pytest.fixture
    def layer(self):
        """Create a residual layer for testing."""
        return ResidualLayer(in_dim=64, h_dim=64, res_h_dim=32)

    def test_output_shape(self, layer):
        """Test that residual layer preserves input shape."""
        x = torch.randn(4, 64, 28, 28)
        y = layer(x)
        assert y.shape == x.shape

    def test_residual_connection(self, layer):
        """Test that residual connection is applied."""
        x = torch.randn(4, 64, 28, 28)
        y = layer(x)
        # Output should be different from input due to skip connection
        assert not torch.allclose(y, x, atol=1e-5)

    def test_gradients_flow(self, layer):
        """Test that gradients are computed through residual layer."""
        # Use non-leaf input to avoid in-place operation issues
        x_orig = torch.randn(4, 64, 28, 28, requires_grad=True)
        x = x_orig * 1.0  # Create a new node in the computation graph
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert x_orig.grad is not None


class TestResidualStack:
    """Test cases for the ResidualStack."""

    @pytest.fixture
    def stack(self):
        """Create a residual stack for testing."""
        return ResidualStack(in_dim=64, h_dim=64, res_h_dim=32, n_res_layers=2)

    def test_output_shape(self, stack):
        """Test that residual stack preserves input shape."""
        x = torch.randn(4, 64, 28, 28)
        y = stack(x)
        assert y.shape == x.shape

    def test_multiple_layers(self):
        """Test stack with different number of layers."""
        for n_layers in [1, 2, 4]:
            stack = ResidualStack(64, 64, 32, n_layers)
            x = torch.randn(4, 64, 28, 28)
            y = stack(x)
            assert y.shape == x.shape


class TestEncoder:
    """Test cases for the Encoder."""

    @pytest.fixture
    def encoder(self):
        """Create an encoder for testing."""
        return Encoder(in_dim=1, h_dim=128, n_res_layers=2, res_h_dim=32)

    def test_output_shape(self, encoder):
        """Test encoder output shape."""
        x = torch.randn(4, 1, 28, 28)  # Batch of 4, 1 channel, 28x28
        z = encoder(x)
        # Encoder downsamples: 28 -> 14 -> 7 -> 7
        assert z.shape == (4, 128, 7, 7)

    def test_output_channels(self, encoder):
        """Test that encoder outputs correct number of channels."""
        x = torch.randn(4, 1, 28, 28)
        z = encoder(x)
        assert z.shape[1] == 128  # h_dim


class TestVectorQuantizer:
    """Test cases for the VectorQuantizer."""

    @pytest.fixture
    def quantizer(self):
        """Create a quantizer for testing."""
        # Set device for vqvae module
        vqvae_module.device = "cpu"
        return VectorQuantizer(n_e=512, e_dim=64, beta=0.25)

    def test_embedding_shape(self, quantizer):
        """Test embedding codebook shape."""
        assert quantizer.embedding.weight.shape == (512, 64)

    def test_n_e_property(self, quantizer):
        """Test n_e read-only property."""
        assert quantizer.n_e == 512

    def test_e_dim_property(self, quantizer):
        """Test e_dim read-only property."""
        assert quantizer.e_dim == 64

    def test_forward_output_shape(self, quantizer):
        """Test quantizer output shapes."""
        z = torch.randn(4, 64, 7, 7)
        loss, z_q, min_encodings, min_encoding_indices = quantizer(z)
        
        assert z_q.shape == z.shape, "Quantized output should match input shape"
        assert min_encodings.shape[0] == 4 * 7 * 7  # Batch * spatial dims
        assert min_encoding_indices.shape[0] == 4 * 7 * 7

    def test_quantization_loss(self, quantizer):
        """Test that quantization loss is computed."""
        z = torch.randn(4, 64, 7, 7)
        loss, _, _, _ = quantizer(z)
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_embedding_update(self, quantizer):
        """Test that embeddings are used for quantization."""
        z = torch.randn(1, 64, 1, 1)
        _, z_q, _, _ = quantizer(z)
        
        # z_q should be close to one of the embeddings
        # (or a linear combination due to straight-through estimator)
        assert z_q.shape == z.shape


class TestDecoder:
    """Test cases for the Decoder."""

    @pytest.fixture
    def decoder(self):
        """Create a decoder for testing."""
        return Decoder(in_dim=64, h_dim=128, n_res_layers=2, res_h_dim=32)

    def test_output_shape(self, decoder):
        """Test decoder output shape."""
        z = torch.randn(4, 64, 7, 7)  # Quantized latent from encoder
        x_hat = decoder(z)
        # Decoder upsamples: 7 -> 7 -> 14 -> 28
        assert x_hat.shape == (4, 1, 28, 28)

    def test_output_channels(self, decoder):
        """Test that decoder outputs 1 channel (grayscale)."""
        z = torch.randn(4, 64, 7, 7)
        x_hat = decoder(z)
        assert x_hat.shape[1] == 1  # 1 channel


class TestVQVAE:
    """Test cases for the complete VQVAE."""

    @pytest.fixture
    def model(self):
        """Create a VQVAE for testing."""
        vqvae_module.device = "cpu"
        return VQVAE(
            h_dim=128,
            res_h_dim=32,
            n_res_layers=2,
            n_embeddings=512,
            embedding_dim=64,
            beta=0.25,
        )

    def test_forward_pass(self, model):
        """Test complete forward pass."""
        x = torch.randn(4, 1, 28, 28)
        embedding_loss, x_hat = model(x)
        
        assert x_hat.shape == x.shape
        assert embedding_loss.item() >= 0
        assert not torch.isnan(embedding_loss)

    def test_different_batch_sizes(self, model):
        """Test model with different batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 1, 28, 28)
            embedding_loss, x_hat = model(x)
            assert x_hat.shape == (batch_size, 1, 28, 28)

    def test_gradients_flow(self, model):
        """Test that gradients flow through the complete model."""
        x = torch.randn(4, 1, 28, 28, requires_grad=True)
        embedding_loss, x_hat = model(x)
        
        loss = embedding_loss + ((x - x_hat) ** 2).mean()
        loss.backward()
        
        # Check that gradients are computed in different parts
        assert model.encoder.conv_stack[0].weight.grad is not None

    def test_model_on_device(self):
        """Test model can be moved to device."""
        vqvae_module.device = "cpu"
        model = VQVAE(128, 32, 2, 512, 64, 0.25)
        model_cpu = model.to("cpu")
        assert next(model_cpu.parameters()).device.type == "cpu"

    def test_deterministic_forward_reconstruction(self, model):
        """Test reconstruction is consistent with same latent codes."""
        z = torch.randn(4, 64, 7, 7)
        
        with torch.no_grad():
            x_hat1 = model.decoder(z)
            x_hat2 = model.decoder(z)
        
        assert torch.allclose(x_hat1, x_hat2)

    def test_encoder_output_dimension_match(self, model):
        """Test that encoder output matches VectorQuantizer input dimension."""
        x = torch.randn(4, 1, 28, 28)
        z_e = model.encoder(x)
        z_e_proj = model.pre_quantization_conv(z_e)
        
        # Should match embedding_dim
        assert z_e_proj.shape[1] == 64

    def test_output_range(self, model):
        """Test output pixel values are in valid range."""
        x = torch.randn(4, 1, 28, 28)
        _, x_hat = model(x)
        
        # Values should be reasonable (between extreme values)
        assert x_hat.min() >= -2.0 and x_hat.max() <= 2.0


class TestVQVAEIntegration:
    """Integration tests for VQ-VAE training."""

    @pytest.fixture
    def simple_dataloader(self):
        """Create a simple dataloader for testing."""
        x = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        dataset = torch.utils.data.TensorDataset(x, y)
        return torch.utils.data.DataLoader(dataset, batch_size=4)

    def test_training_loop_execution(self, simple_dataloader):
        """Test that a training loop can execute without errors."""
        vqvae_module.device = "cpu"
        model = VQVAE(128, 32, 2, 512, 64, 0.25).to("cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        model.train()
        for batch_idx, (x, _) in enumerate(simple_dataloader):
            embedding_loss, x_hat = model(x)
            recon_loss = torch.nn.functional.mse_loss(x_hat, x)
            loss = recon_loss + embedding_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            assert not torch.isnan(loss), "Loss became NaN"
            
            if batch_idx == 1:  # Just run 2 batches for quick test
                break
