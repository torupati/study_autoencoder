# study_autoencoder

![CI](https://github.com/yourusername/study_autoencoder/workflows/CI/badge.svg)

These code are prepared for my learning of variational autoencoder.
Code work with MNIST dataset and are specialised for this dataset.

For comparison, autoencoder is also prepared. Each sample of MNIST can be converted to PNG file.

## Features

- **Custom MNIST Dataset**: Pure PyTorch implementation without torchvision dependency
- **CPU-only PyTorch**: Optimized for CPU-only environments, no CUDA dependencies
- **Modern tooling**: Uses uv for package management, ruff for linting/formatting
- **Comprehensive CI**: GitHub Actions with testing and code quality checks

### Custom MNIST Implementation

This project includes a custom MNIST dataset implementation (`models/dataset_mnist.py`) that reads the original MNIST binary files directly, eliminating the need for torchvision. The implementation:

- Supports both training and test sets
- Reads compressed (.gz) and uncompressed binary files
- Provides the same interface as torchvision's MNIST dataset
- Includes proper data validation and error handling


### Usage 

This project uses [uv](https://docs.astral.sh/uv/) for Python package and dependency management.

#### Installation with uv

First, install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, create a virtual environment and install dependencies:

```bash
# For CPU-only environment (recommended for development without GPU)
make install-cpu
source .venv/bin/activate

# Or for CUDA 12.1 environment (if NVIDIA GPU available)
make install-cuda
source .venv/bin/activate
```

**Environment Options:**

- **CPU-only** (default): Lightweight, no GPU required, perfect for development/testing
  ```bash
  make install-cpu
  ```

- **CUDA 12.1**: GPU-accelerated training, requires NVIDIA GPU and CUDA toolkit
  ```bash
  make install-cuda
  ```

- **Manual installation** (using uv directly):
  ```bash
  # CPU version
  UV_INDEX_URL="https://download.pytorch.org/whl/cpu" \
  UV_EXTRA_INDEX_URL="https://pypi.org/simple" \
  uv sync

  # CUDA version
  UV_INDEX_URL="https://download.pytorch.org/whl/cu121" \
  UV_EXTRA_INDEX_URL="https://pypi.org/simple" \
  uv sync
  ```

**Note**: The `pyproject.toml` is environment-agnostic. PyTorch version is the same for both CPU and CUDA. The actual device (CPU or GPU) is determined by the PyTorch wheel index specified during installation.

#### Data Preparation

Before running the applications, download the MNIST dataset:

```bash
# Download MNIST dataset files
uv run python misc/download_mnist.py --path ./data

# Or manually download from http://yann.lecun.com/exdb/mnist/
# Files will be saved to data/MNIST/raw/
```

#### Running the code

After installation, you can run the autoencoder training:

```bash
# Run standard autoencoder
uv run python app/run_mnist_autoencoder.py

# Run variational autoencoder
uv run python app/run_mnist_v_autoencoder.py
```

#### Development

For development, install with dev dependencies:

```bash
uv sync --all-groups
```

**Important**: The code will automatically use CPU device since CUDA is not available. The device detection in the code will set `device = 'cpu'`.

#### Code Quality Tools

This project uses [ruff](https://docs.astral.sh/ruff/) for fast Python linting and formatting:

```bash
# Lint code
uv run ruff check .

# Format code
uv run ruff format .

# Lint and auto-fix issues
uv run ruff check --fix .

# Or use make commands
make lint       # Check for issues
make format     # Format code
make lint-fix   # Fix issues and format
```

Run tests:

```bash
uv run pytest tests/
```

#### Available Commands

Use the Makefile for common development tasks:

```bash
make help              # Show all available commands
make install-cpu       # Install CPU-only PyTorch (default)
make install-cuda      # Install CUDA 12.1 PyTorch for GPU
make sync              # Sync dependencies with current environment
make install-dev       # Install with dev dependencies
make format            # Format code with ruff
make lint              # Run linting checks
make lint-fix          # Auto-fix linting issues and format
make test              # Run tests
make ci-check          # Run all CI checks locally
make clean             # Clean up cache files
```

**Switching between environments:**

```bash
# From CPU to CUDA
make install-cuda
source .venv/bin/activate

# From CUDA to CPU
make install-cpu
source .venv/bin/activate
```

**Using environment files (optional):**

Pre-configured environment files are available for quick setup:

```bash
# Source the CPU environment
source .env.cpu
make install-cpu

# Or source the CUDA environment
source .env.cuda
make install-cuda
```

Setup pre-commit hooks (optional):

```bash
make setup-pre-commit
```

#### Continuous Integration

This project uses GitHub Actions for continuous integration. On every push and pull request to any branch, the following checks are automatically run:

- **Code Linting**: Flake8 checks for code quality and style
- **Code Formatting**: Black and isort check code formatting
- **Type Checking**: MyPy performs static type analysis
- **Testing**: PyTest runs unit tests
- **PyTorch Verification**: Ensures CPU-only PyTorch is working correctly

The CI pipeline tests against Python versions 3.9, 3.10, 3.11, and 3.12.

## References

### VAE

- Original Paper: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (Kingma & Welling, 2013)
- Tutorial: https://avandekleut.github.io/vae/

### VQ-VAE

- Original paper: [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) (Oord et.al., 2017)
- Tutlrial: https://github.com/praeclarumjj3/VQ-VAE-on-MNIST

### MNIST

https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html


### programming memo (TODO)

- multi-processing
  https://pytorch.org/docs/stable/notes/multiprocessing.html

- how to save and load model:
  https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

- MNIST Dataset
  https://pytorch.org/vision/0.15/datasets.html

- docker build for poetry
  https://stackoverflow.com/questions/68756419/dockerfile-multistage-python-poetry-install

