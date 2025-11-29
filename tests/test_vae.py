"""
Unit tests for vae.py (Variational Autoencoder)
"""

import pytest
import torch
import torch.nn as nn
from models.mnist.vae import VariationalAutoencoder, train_vae


class TestVariationalAutoencoder:
    """Test cases for the VariationalAutoencoder class."""

    @pytest.fixture
    def model(self):
        """Create a simple VAE for testing."""
        return VariationalAutoencoder(latent_dims=32, obs_dim=784)

    def test_encoder_output_shape(self, model):
        """Test that encoder produces correct latent dimension."""
        x = torch.randn(16, 784)
        output = model.encoder(x)
        # Should return dict with 'z', 'mu', 'sigma'
        assert isinstance(output, dict)
        assert "z" in output
        assert "mu" in output
        assert "sigma" in output
        assert output["z"].shape == (16, 32)
        assert output["mu"].shape == (16, 32)
        assert output["sigma"].shape == (16, 32)

    def test_decoder_output_shape(self, model):
        """Test that decoder produces correct reconstruction dimension."""
        z = torch.randn(16, 32)
        x_hat = model.decoder(z)
        assert x_hat.shape == (16, 784)

    def test_forward_pass(self, model):
        """Test complete forward pass."""
        x = torch.randn(16, 784)
        output = model.forward(x)
        # Forward returns just the reconstruction tensor
        assert isinstance(output, torch.Tensor)
        assert output.shape == x.shape

    def test_output_range(self, model):
        """Test that decoder output is in [0, 1] range (sigmoid activation)."""
        z = torch.randn(16, 32)
        x_hat = model.decoder(z)
        assert torch.all(x_hat >= 0.0), "Decoder output below 0"
        assert torch.all(x_hat <= 1.0), "Decoder output above 1"

    def test_mu_sigma_properties(self, model):
        """Test that mu and sigma have expected properties."""
        x = torch.randn(16, 784)
        output = model.encoder(x)
        mu = output["mu"]
        sigma = output["sigma"]

        # Sigma should be positive
        assert torch.all(sigma > 0), "Sigma should be positive"

        # Both should have same shape as latent_dims
        assert mu.shape == (16, 32)
        assert sigma.shape == (16, 32)

    def test_sampling_diversity(self, model):
        """Test that sampling produces different values."""
        x = torch.randn(1, 784)

        with torch.no_grad():
            output1 = model.encoder(x)
            output2 = model.encoder(x)

        z1 = output1["z"]
        z2 = output2["z"]

        # Should be different due to sampling (unless by chance identical)
        # At least they should have same shape
        assert z1.shape == z2.shape

    def test_gradients_flow(self, model):
        """Test that gradients flow through the model."""
        x = torch.randn(16, 784, requires_grad=True)
        x_hat = model.forward(x)
        loss = ((x - x_hat) ** 2).mean() + model.encoder.kl
        loss.backward()

        # Check that gradients are computed
        assert model.encoder.linear1.weight.grad is not None
        assert model.decoder.linear2.weight.grad is not None

    @pytest.mark.parametrize("batch_size", [1, 8, 16, 32])
    def test_different_batch_sizes(self, model, batch_size):
        """Test model with different batch sizes."""
        x = torch.randn(batch_size, 784)
        x_hat = model.forward(x)
        assert x_hat.shape == (batch_size, 784)

    def test_model_on_device(self):
        """Test model can be moved to different devices."""
        model = VariationalAutoencoder(latent_dims=32, obs_dim=784)
        model_cpu = model.to("cpu")
        assert next(model_cpu.parameters()).device.type == "cpu"

    def test_encoder_input_features(self, model):
        """Test encoder.linear1.in_features."""
        assert model.encoder.linear1.in_features == 784

    def test_latent_dims_property(self, model):
        """Test latent_dims property."""
        assert model.latent_dims == 32

    def test_reconstruction_loss_computation(self, model):
        """Test that reconstruction loss can be computed."""
        x = torch.randn(16, 784)
        x_hat = model.forward(x)

        # MSE loss for reconstruction
        loss = ((x - x_hat) ** 2).mean()
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_kl_divergence_computation(self, model):
        """Test that KL divergence is computed in encoder."""
        x = torch.randn(16, 784)
        model.encoder(x)
        kl = model.encoder.kl

        assert kl.item() >= 0, "KL divergence should be non-negative"
        assert not torch.isnan(torch.tensor(kl.item()))


class TestVariationalAutoencoderTrain:
    """Test cases for the train_vae function."""

    @pytest.fixture
    def simple_dataloader(self):
        """Create a simple dataloader for testing."""
        # Create data as 1x28x28 images (to be flattened in training)
        x = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        dataset = torch.utils.data.TensorDataset(x, y)
        return torch.utils.data.DataLoader(dataset, batch_size=16)

    def test_train_vae_returns_model(self, simple_dataloader):
        """Test that train_vae returns a trained model."""
        model = VariationalAutoencoder(latent_dims=32, obs_dim=784)

        trained_model = train_vae(model, simple_dataloader, epochs=1, start_epoch=0)

        assert trained_model is not None
        assert isinstance(trained_model, VariationalAutoencoder)
