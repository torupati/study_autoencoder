from pathlib import Path

import numpy as np
import pytest
import torch

from models.mnist import inference
from models.mnist.vae import VariationalAutoencoder


@pytest.fixture
def model() -> VariationalAutoencoder:
    torch.manual_seed(0)
    model = VariationalAutoencoder(2, 28 * 28)
    model.eval()
    return model


@pytest.mark.parametrize(
    "state_key, with_latent_dims",
    [("model_state_dict", True), ("model_state", False)],
)
def test_load_vae_checkpoint_formats(
    tmp_path: Path, model: VariationalAutoencoder, state_key: str, with_latent_dims: bool
) -> None:
    ckpt = {state_key: model.state_dict()}
    if with_latent_dims:
        ckpt["latent_dims"] = 2
    path = tmp_path / "ckpt.pth"
    torch.save(ckpt, path)

    loaded = inference.load_vae(str(path))

    assert loaded.latent_dims == 2
    assert not loaded.training


def test_encode_is_deterministic(model: VariationalAutoencoder) -> None:
    img = np.random.default_rng(0).random((28, 28)).astype(np.float32)
    mu1, sigma1 = inference.encode(model, img)
    mu2, _ = inference.encode(model, img)

    assert mu1.shape == (1, 2)
    assert sigma1.shape == (1, 2)
    assert (sigma1 > 0).all()
    np.testing.assert_array_equal(mu1, mu2)


def test_decode_shape_and_range(model: VariationalAutoencoder) -> None:
    out = inference.decode(model, np.zeros((5, 2)))

    assert out.shape == (5, 28, 28)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_interpolate_endpoints() -> None:
    z = inference.interpolate(np.array([0.0, 0.0]), np.array([1.0, -2.0]), 5)

    assert z.shape == (5, 2)
    np.testing.assert_allclose(z[0], [0.0, 0.0])
    np.testing.assert_allclose(z[-1], [1.0, -2.0])


def test_sample_and_grid(model: VariationalAutoencoder) -> None:
    assert inference.sample(model, 6, seed=0).shape == (6, 28, 28)
    assert inference.latent_grid(model, n=4).shape == (4 * 28, 4 * 28)
    assert inference.tile(np.ones((5, 28, 28)), 3).shape == (2 * 28, 3 * 28)


def test_latent_grid_requires_2d() -> None:
    with pytest.raises(ValueError):
        inference.latent_grid(VariationalAutoencoder(3, 28 * 28))


def test_preprocess_drawing_centers_digit() -> None:
    canvas = np.zeros((280, 280, 4), dtype=np.uint8)  # transparent background
    canvas[20:120, 30:60] = [0, 0, 0, 255]  # black vertical stroke in the top-left

    out = inference.preprocess_drawing(canvas)

    assert out is not None
    assert out.shape == (28, 28)
    assert out.max() <= 1.0
    assert out.max() > 0.5
    yy, xx = np.mgrid[:28, :28]
    cy, cx = (yy * out).sum() / out.sum(), (xx * out).sum() / out.sum()
    assert abs(cy - 13.5) <= 1.0
    assert abs(cx - 13.5) <= 1.0


def test_preprocess_drawing_empty() -> None:
    assert inference.preprocess_drawing(np.zeros((280, 280, 4), dtype=np.uint8)) is None
