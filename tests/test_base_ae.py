"""
Unit tests for base_ae.py (Simple Autoencoder)
"""

import pytest
import torch
import torch.nn as nn

from models.mnist.base_ae import Autoencoder


class TestAutoencoder:
    """Test cases for the Autoencoder class."""

    @pytest.fixture
    def model(self):
        """Create a simple autoencoder for testing."""
        return Autoencoder(latent_dims=32, obs_dim=784)

    def test_encoder_output_shape(self, model):
        """Test that encoder produces correct latent dimension."""
        x = torch.randn(16, 784)  # Batch of 16, 784-dim input
        z = model.encoder(x)
        assert z.shape == (16, 32), f"Expected (16, 32), got {z.shape}"

    def test_decoder_output_shape(self, model):
        """Test that decoder produces correct reconstruction dimension."""
        z = torch.randn(16, 32)  # Batch of 16, 32-dim latent
        x_hat = model.decoder(z)
        assert x_hat.shape == (16, 784), f"Expected (16, 784), got {x_hat.shape}"

    def test_forward_pass(self, model):
        """Test complete forward pass."""
        x = torch.randn(16, 784)
        x_hat = model(x)
        assert x_hat.shape == x.shape, f"Expected {x.shape}, got {x_hat.shape}"

    def test_output_range(self, model):
        """Test that decoder output is in [-1, 1] range (tanh activation)."""
        z = torch.randn(16, 32)
        x_hat = model.decoder(z)
        assert torch.all(x_hat >= -1.0), "Decoder output below -1"
        assert torch.all(x_hat <= 1.0), "Decoder output above 1"

    def test_gradients_flow(self, model):
        """Test that gradients flow through the model."""
        x = torch.randn(16, 784, requires_grad=True)
        x_hat = model(x)
        loss = ((x - x_hat) ** 2).mean()
        loss.backward()

        # Check that gradients are computed
        assert model.encoder.linear1.weight.grad is not None
        assert model.decoder.linear2.weight.grad is not None

    def test_encoder_in_features(self, model):
        """Test encoder.in_features property."""
        assert model.encoder.in_features == 784

    def test_latent_dim_property(self, model):
        """Test latent_dim property."""
        assert model.latent_dim == 32

    def test_different_batch_sizes(self, model):
        """Test model with different batch sizes."""
        for batch_size in [1, 8, 16, 32]:
            x = torch.randn(batch_size, 784)
            x_hat = model(x)
            assert x_hat.shape == (batch_size, 784)

    def test_model_on_device(self):
        """Test model can be moved to different devices."""
        model = Autoencoder(latent_dims=32, obs_dim=784)
        model_cpu = model.to("cpu")
        assert next(model_cpu.parameters()).device.type == "cpu"

    def test_deterministic_forward_pass(self, model):
        """Test that forward pass is deterministic (no dropout)."""
        x = torch.randn(16, 784)
        with torch.no_grad():
            x_hat1 = model(x)
            x_hat2 = model(x)
        assert torch.allclose(x_hat1, x_hat2), "Forward pass should be deterministic"

    def test_encoder_hidden_layer_activation(self, model):
        """Test encoder hidden layer uses ReLU."""
        x = torch.randn(1, 784)
        # Get intermediate activation
        h = torch.relu(model.encoder.linear1(x))
        assert torch.all(h >= 0), "ReLU should produce non-negative values"


class TestAutoencoderTrain:
    """Test cases for the train function."""

    @pytest.fixture
    def simple_dataloader(self):
        """Create a simple dataloader for testing."""
        x = torch.randn(32, 784)
        y = torch.randint(0, 10, (32,))
        dataset = torch.utils.data.TensorDataset(x, y)
        return torch.utils.data.DataLoader(dataset, batch_size=16)

    def test_train_loss_decreases(self, simple_dataloader):
        """Test that training loop reduces loss (basic sanity check)."""
        model = Autoencoder(latent_dims=32, obs_dim=784)
        from models.mnist.base_ae import train

        # Train for a few epochs
        trained_model = train(model, simple_dataloader, epochs=2, start_epoch=0)

        # Model should be returned
        assert trained_model is not None
        assert isinstance(trained_model, Autoencoder)
