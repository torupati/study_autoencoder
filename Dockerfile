# syntax=docker/dockerfile:1
# Container image for the MNIST VAE web demo (app/web_vae_demo.py), e.g. for Cloud Run.
#
# The trained checkpoint is not in git: place it at model/vae.pth before building
# (cloudbuild.yaml copies it from Cloud Storage).
#
#   docker build -t vae-demo .
#   docker build --build-arg COMPILE_BYTECODE=0 -t vae-demo .   # faster local test build
#   docker run --rm -p 8080:8080 vae-demo   # then open http://localhost:8080

# --- build stage: dependencies (CPU-only PyTorch) and MNIST test set --------------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# Precompiling .pyc speeds up container start (Cloud Run cold starts) but makes the
# build slower; skip it for local test builds with --build-arg COMPILE_BYTECODE=0
ARG COMPILE_BYTECODE=1

# UV_LINK_MODE=copy: the uv cache is a separate mount, so files cannot be hard-linked
ENV UV_COMPILE_BYTECODE=${COMPILE_BYTECODE} \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=120 \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app
# Install the exact versions pinned in uv.lock (CPU-only torch included); --frozen
# fails instead of re-resolving if the lock is out of date with pyproject.toml.
# The demo needs the core dependencies plus the "web" extra; training-only packages
# it never imports are skipped to keep the image small.
# The cache mount keeps downloaded wheels between local builds (BuildKit only); it is
# not part of the image
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra web --no-install-project \
        --no-install-package opencv-python \
        --no-install-package torchvision

# The demo only uses the test set; the loader reads the .gz files directly
COPY misc/download_mnist.py misc/
RUN .venv/bin/python misc/download_mnist.py --path ./data \
    && rm -f data/MNIST/raw/train-*

# --- runtime stage -----------------------------------------------------------------
FROM python:3.13-slim-bookworm

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/data /app/data
COPY models/ models/
COPY app/ app/
COPY model/vae.pth model/vae.pth

# Cloud Run sets PORT (8080 by default); HOST must be 0.0.0.0 inside a container
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080 \
    GRADIO_ANALYTICS_ENABLED=False \
    MPLCONFIGDIR=/tmp/matplotlib

RUN useradd --create-home appuser
USER appuser

EXPOSE 8080
CMD ["python", "app/web_vae_demo.py", "--ckpt", "model/vae.pth", "--dataset_dir", "data/MNIST/"]
