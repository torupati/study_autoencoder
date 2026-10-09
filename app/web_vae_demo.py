"""Web UI demo for the MNIST Variational Autoencoder (Gradio).

Tabs:
- Reconstruct: draw a digit (or pick a test image) and see its reconstruction and latent code
- Latent explorer: move sliders (or click the 2-D latent grid) to decode any latent vector
- Interpolation: decode points on the line between two test images in latent space
- Sampling: decode random draws from the prior N(0, I)

Usage:
    uv sync --extra web
    uv run python app/web_vae_demo.py --ckpt checkpoints/vae_z2/checkpoint_vae_epoch040.pth
"""

from __future__ import annotations

import argparse
import glob
import logging
import os

import matplotlib

# Local demo: no usage analytics sent to Gradio/Hugging Face unless explicitly enabled
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
matplotlib.use("Agg")
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from models.datasets import get_mnist_dataset
from models.mnist import inference
from models.mnist.vae import VariationalAutoencoder

logger = logging.getLogger("web_vae_demo")

MAX_SLIDERS = 10
SLIDER_RANGE = 4.0
GRID_RANGE = (-3.0, 3.0)
GRID_N = 15
SCATTER_POINTS = 3000


def to_display(img: np.ndarray, scale: int = 8) -> np.ndarray:
    """Upscale a [0, 1] grayscale image with nearest neighbour for crisp display."""
    up = np.kron(np.clip(img, 0.0, 1.0), np.ones((scale, scale)))
    return (up * 255).astype(np.uint8)


def find_checkpoints(patterns: list[str]) -> list[str]:
    """List VAE checkpoint files matching the glob patterns, newest name last."""
    found: set[str] = set()
    for pattern in patterns:
        found.update(glob.glob(pattern, recursive=True))
    return sorted(found)


class DemoState:
    """Holds the loaded model and test-set embeddings (single-user local demo)."""

    def __init__(self, dataset_dir: str) -> None:
        testset = get_mnist_dataset(root=dataset_dir, train=False)
        self.images = np.stack([testset[i][0].numpy()[0] for i in range(len(testset))])
        self.labels = np.array([testset[i][1] for i in range(len(testset))])
        self.model: VariationalAutoencoder | None = None
        self.mu = np.zeros((0, 2))
        self.ckpt = ""

    def load(self, ckpt_path: str) -> str:
        self.model = inference.load_vae(ckpt_path)
        self.mu, _ = inference.encode(self.model, self.images[:SCATTER_POINTS])
        self.ckpt = ckpt_path
        msg = f"Loaded `{ckpt_path}` (latent_dims={self.model.latent_dims})"
        logger.info(msg)
        return msg

    @property
    def latent_dims(self) -> int:
        assert self.model is not None
        return self.model.latent_dims


def latent_scatter(state: DemoState, marks: np.ndarray | None = None):
    """Scatter test-set latent means (first two dims), coloured by digit label."""
    fig, ax = plt.subplots(figsize=(5, 5))
    n = len(state.mu)
    sc = ax.scatter(
        state.mu[:, 0], state.mu[:, 1], c=state.labels[:n], cmap="tab10", s=4, alpha=0.6
    )
    if marks is not None:
        ax.plot(marks[:, 0], marks[:, 1], "k-", lw=1, alpha=0.7)
        ax.scatter(marks[:, 0], marks[:, 1], c="black", marker="x", s=60, zorder=3)
    fig.colorbar(sc, ax=ax, ticks=range(10), label="digit")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$" if state.latent_dims > 1 else "")
    ax.set_xlim(-SLIDER_RANGE, SLIDER_RANGE)
    ax.set_ylim(-SLIDER_RANGE, SLIDER_RANGE)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def build_ui(state: DemoState, checkpoints: list[str]) -> gr.Blocks:
    def slider_updates(values: np.ndarray | None = None) -> list:
        d = state.latent_dims
        return [
            gr.update(
                visible=i < d,
                value=float(values[i]) if values is not None and i < d else 0.0,
            )
            for i in range(MAX_SLIDERS)
        ]

    # --- Reconstruct -------------------------------------------------------
    def reconstruct_image(img28: np.ndarray, title: str):
        assert state.model is not None
        mu, sigma = inference.encode(state.model, img28)
        recon = inference.decode(state.model, mu)[0]
        mse = float(((recon - img28) ** 2).sum())
        info = (
            f"{title}\n\nmu = {np.array2string(mu[0], precision=3)}\n\n"
            f"sigma = {np.array2string(sigma[0], precision=3)}\n\n"
            f"squared error = {mse:.2f}"
        )
        return to_display(img28), to_display(recon), info, latent_scatter(state, mu[:, :2])

    def on_draw(sketch):
        if sketch is None or sketch.get("composite") is None:
            return None, None, "Draw a digit on the canvas.", None
        img28 = inference.preprocess_drawing(sketch["composite"])
        if img28 is None:
            return None, None, "Draw a digit on the canvas.", None
        return reconstruct_image(img28, "**Your drawing**")

    def on_random_test():
        idx = int(np.random.randint(len(state.images)))
        return reconstruct_image(
            state.images[idx], f"**Test image #{idx}** (label {state.labels[idx]})"
        )

    # --- Latent explorer ---------------------------------------------------
    def on_sliders(*values):
        assert state.model is not None
        z = np.array(values[: state.latent_dims])
        return to_display(inference.decode(state.model, z)[0], 10)

    def explorer_grid():
        if state.latent_dims != 2:
            return None
        grid = inference.latent_grid(state.model, GRID_RANGE, GRID_RANGE, GRID_N)
        return to_display(grid, 2)

    def on_grid_click(evt: gr.SelectData):
        # evt.index is (x, y) in displayed pixels; map back to latent coordinates
        size = GRID_N * inference.IMG_SIZE * 2
        x, y = evt.index
        lo, hi = GRID_RANGE
        z1 = lo + (hi - lo) * x / size
        z2 = hi - (hi - lo) * y / size
        return slider_updates(np.array([z1, z2]))

    # --- Interpolation -----------------------------------------------------
    def on_interpolate(idx_a: float, idx_b: float, steps: float):
        assert state.model is not None
        a, b = int(idx_a), int(idx_b)
        mu, _ = inference.encode(state.model, state.images[[a, b]])
        z = inference.interpolate(mu[0], mu[1], int(steps))
        strip = inference.tile(inference.decode(state.model, z), int(steps))
        caption = f"#{a} (label {state.labels[a]}) → #{b} (label {state.labels[b]})"
        return to_display(strip, 4), caption, latent_scatter(state, z[:, :2])

    def on_random_pair(steps: float):
        a, b = np.random.choice(len(state.images), 2, replace=False)
        return (int(a), int(b), *on_interpolate(a, b, steps))

    # --- Sampling ----------------------------------------------------------
    def on_sample(n: float, seed: float):
        assert state.model is not None
        n = int(n)
        imgs = inference.sample(state.model, n, seed=int(seed))
        return to_display(inference.tile(imgs, min(n, 8)), 4)

    # --- Checkpoint switching ----------------------------------------------
    def on_checkpoint(path: str):
        msg = state.load(path)
        return (msg, explorer_grid(), latent_scatter(state), *slider_updates())

    with gr.Blocks(title="MNIST VAE Demo") as demo:
        gr.Markdown("# MNIST Variational Autoencoder demo")
        with gr.Row():
            ckpt_dd = gr.Dropdown(
                choices=checkpoints, value=state.ckpt, label="Checkpoint", scale=3
            )
            status = gr.Markdown(f"Loaded `{state.ckpt}` (latent_dims={state.latent_dims})")

        with gr.Tab("Reconstruct"):
            with gr.Row():
                with gr.Column():
                    sketch = gr.Sketchpad(
                        label="Draw a digit (0-9)",
                        canvas_size=(280, 280),
                        brush=gr.Brush(default_size=18, colors=["#000000"], color_mode="fixed"),
                        type="numpy",
                        image_mode="RGBA",
                        layers=False,
                    )
                    rand_btn = gr.Button("Random test image")
                with gr.Column():
                    with gr.Row():
                        inp_img = gr.Image(label="Model input (28×28)", height=224)
                        rec_img = gr.Image(label="Reconstruction", height=224)
                    rec_info = gr.Markdown("Draw a digit or pick a random test image.")
                rec_plot = gr.Plot(label="Latent mean (×) on test-set latent space")
            rec_outputs = [inp_img, rec_img, rec_info, rec_plot]
            sketch.change(on_draw, sketch, rec_outputs)
            rand_btn.click(on_random_test, None, rec_outputs)

        with gr.Tab("Latent explorer"):
            with gr.Row():
                with gr.Column():
                    sliders = [
                        gr.Slider(
                            -SLIDER_RANGE,
                            SLIDER_RANGE,
                            value=0.0,
                            step=0.05,
                            label=f"z{i + 1}",
                            visible=i < state.latent_dims,
                        )
                        for i in range(MAX_SLIDERS)
                    ]
                    decoded = gr.Image(label="Decoded image", height=280)
                with gr.Column():
                    grid_img = gr.Image(
                        value=explorer_grid(),
                        label=f"Decoded grid over z ∈ {GRID_RANGE}² — click to select",
                        interactive=False,
                    )
                    explorer_plot = gr.Plot(
                        value=latent_scatter(state), label="Test-set latent means"
                    )
            for s in sliders:
                s.change(on_sliders, sliders, decoded)
            grid_img.select(on_grid_click, None, sliders)
            demo.load(on_sliders, sliders, decoded)

        with gr.Tab("Interpolation"):
            with gr.Row():
                idx_a = gr.Number(value=0, precision=0, label="Test image A", minimum=0)
                idx_b = gr.Number(value=1, precision=0, label="Test image B", minimum=0)
                steps = gr.Slider(3, 16, value=10, step=1, label="Steps")
            with gr.Row():
                interp_btn = gr.Button("Interpolate", variant="primary")
                pair_btn = gr.Button("Random pair")
            interp_caption = gr.Markdown()
            interp_img = gr.Image(label="Decoded path from A to B")
            interp_plot = gr.Plot(label="Path in latent space")
            interp_outputs = [interp_img, interp_caption, interp_plot]
            interp_btn.click(on_interpolate, [idx_a, idx_b, steps], interp_outputs)
            pair_btn.click(on_random_pair, steps, [idx_a, idx_b, *interp_outputs])

        with gr.Tab("Sampling"):
            with gr.Row():
                n_samples = gr.Slider(1, 64, value=32, step=1, label="Number of samples")
                seed = gr.Number(value=0, precision=0, label="Seed")
                sample_btn = gr.Button("Sample z ~ N(0, I)", variant="primary")
            sample_img = gr.Image(label="Generated digits")
            sample_btn.click(on_sample, [n_samples, seed], sample_img)

        ckpt_dd.change(on_checkpoint, ckpt_dd, [status, grid_img, explorer_plot, *sliders])
    return demo


def prepare_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gradio web demo for the MNIST VAE")
    parser.add_argument("--ckpt", "-c", default="", help="VAE checkpoint (default: newest found)")
    parser.add_argument("--dataset_dir", default="./data/MNIST/", help="MNIST dataset directory")
    # HOST/PORT env vars let containers (e.g. Cloud Run, which sets PORT) override defaults
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"), help="Server host")
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", "7860")), help="Server port"
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    return parser


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    checkpoints = find_checkpoints(
        ["checkpoints/**/checkpoint_vae*.pth", "checkpoint_vae*.pth", args.ckpt or "-"]
    )
    ckpt = args.ckpt or (checkpoints[-1] if checkpoints else "")
    if not ckpt or not os.path.isfile(ckpt):
        raise SystemExit("No VAE checkpoint found; train one with app/run_mnist_vae.py")

    state = DemoState(args.dataset_dir)
    state.load(ckpt)
    demo = build_ui(state, checkpoints)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main(prepare_argparse().parse_args())
