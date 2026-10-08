"""Inference helpers for a trained MNIST VAE.

UI-independent functions used by the web demo (app/web_vae_demo.py): load a checkpoint,
encode images to the latent mean, decode latent vectors, interpolate and sample.
Images are 28x28 float arrays with pixel values in [0, 1], as used in training.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from models.mnist.vae import VariationalAutoencoder

IMG_SIZE = 28


def load_vae(ckpt_path: str, device: str = "cpu") -> VariationalAutoencoder:
    """Load a VAE from a checkpoint saved by Trainer or the legacy training loop.

    latent_dims is read from the checkpoint, or inferred from the weights when absent.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt.get("model_state"))
    if state is None:
        raise KeyError(f"No model state found in checkpoint: {ckpt_path}")
    latent_dims = ckpt.get("latent_dims") or state["encoder.linear2.weight"].shape[0]
    model = VariationalAutoencoder(int(latent_dims), IMG_SIZE * IMG_SIZE)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def encode(model: VariationalAutoencoder, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode images of shape (N, 28, 28) or (28, 28) to latent (mu, sigma), each (N, D)."""
    x = torch.as_tensor(images, dtype=torch.float32).reshape(-1, IMG_SIZE * IMG_SIZE)
    out = model.encoder(x)
    return out["mu"].numpy(), out["sigma"].numpy()


@torch.no_grad()
def decode(model: VariationalAutoencoder, z: np.ndarray) -> np.ndarray:
    """Decode latent vectors of shape (N, D) or (D,) to images of shape (N, 28, 28)."""
    z_t = torch.as_tensor(z, dtype=torch.float32).reshape(-1, model.latent_dims)
    return model.decoder(z_t).reshape(-1, IMG_SIZE, IMG_SIZE).numpy()


def reconstruct(model: VariationalAutoencoder, image: np.ndarray) -> np.ndarray:
    """Reconstruct a single 28x28 image from its latent mean (deterministic)."""
    mu, _ = encode(model, image)
    return decode(model, mu)[0]


def interpolate(z0: np.ndarray, z1: np.ndarray, steps: int) -> np.ndarray:
    """Linearly interpolate between two latent vectors, endpoints included: (steps, D)."""
    t = np.linspace(0.0, 1.0, steps)[:, None]
    return (1.0 - t) * np.asarray(z0)[None, :] + t * np.asarray(z1)[None, :]


def sample(model: VariationalAutoencoder, n: int, seed: int | None = None) -> np.ndarray:
    """Decode n samples drawn from the prior N(0, I)."""
    rng = np.random.default_rng(seed)
    return decode(model, rng.standard_normal((n, model.latent_dims)))


def latent_grid(
    model: VariationalAutoencoder,
    r0: tuple[float, float] = (-3.0, 3.0),
    r1: tuple[float, float] = (-3.0, 3.0),
    n: int = 15,
) -> np.ndarray:
    """Decode an n x n grid over a 2-D latent space into one (n*28, n*28) image.

    z1 increases upward so the image matches a scatter plot of the latent space.
    """
    if model.latent_dims != 2:
        raise ValueError("latent_grid requires a model with latent_dims == 2")
    xs = np.linspace(*r0, n)
    ys = np.linspace(*r1, n)[::-1]
    z = np.array([[x, y] for y in ys for x in xs])
    return tile(decode(model, z), n)


def tile(images: np.ndarray, ncols: int) -> np.ndarray:
    """Arrange (N, H, W) images into a grid with ncols columns (blank-padded)."""
    n, h, w = images.shape
    nrows = -(-n // ncols)
    canvas = np.zeros((nrows * h, ncols * w), dtype=images.dtype)
    for i, img in enumerate(images):
        r, c = divmod(i, ncols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    return canvas


def preprocess_drawing(rgba: np.ndarray) -> np.ndarray | None:
    """Convert a dark-on-light (or on transparent) drawing to an MNIST-style 28x28 image.

    Mimics MNIST preprocessing: the digit is scaled to fit a 20x20 box, then placed
    in a 28x28 frame so that its center of mass is at the center.
    Returns None for an empty drawing.
    """
    arr = np.asarray(rgba, dtype=np.float32)
    if arr.ndim == 2:
        ink = 1.0 - arr / 255.0
    else:
        rgb = arr[..., :3].mean(axis=-1) / 255.0
        alpha = arr[..., 3] / 255.0 if arr.shape[-1] == 4 else np.ones(rgb.shape, np.float32)
        # composite onto white, then invert so strokes become bright on black
        ink = 1.0 - (alpha * rgb + (1.0 - alpha))
    ink = np.clip(ink, 0.0, 1.0)

    rows = np.flatnonzero(ink.max(axis=1) > 0.1)
    cols = np.flatnonzero(ink.max(axis=0) > 0.1)
    if rows.size == 0:
        return None
    crop = ink[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    pil = Image.fromarray((crop * 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)
    digit = np.asarray(pil, dtype=np.float32) / 255.0

    out = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    top, left = (IMG_SIZE - new_h) // 2, (IMG_SIZE - new_w) // 2
    out[top : top + new_h, left : left + new_w] = digit

    # shift so the center of mass lands at the image center
    total = out.sum()
    if total > 0:
        yy, xx = np.mgrid[:IMG_SIZE, :IMG_SIZE]
        dy = round(IMG_SIZE / 2 - (yy * out).sum() / total - 0.5)
        dx = round(IMG_SIZE / 2 - (xx * out).sum() / total - 0.5)
        out = np.roll(out, (dy, dx), axis=(0, 1))
    return out
