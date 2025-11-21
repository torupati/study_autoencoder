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
# Create virtual environment with Python 3.9+
uv venv --python 3.9

# Activate the virtual environment
source .venv/bin/activate

# Install CPU-only PyTorch dependencies
uv sync

# Or use make command for easier installation
make install
```

**Note**: This project is configured to use CPU-only versions of PyTorch and TorchVision to avoid CUDA dependencies. The configuration in `pyproject.toml` ensures that only CPU versions are installed from the PyTorch CPU index.

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
make help          # Show all available commands
make install       # Install dependencies (CPU-only PyTorch)
make dev-install   # Install with dev dependencies
make format        # Format code with ruff
make lint          # Run linting checks
make lint-fix      # Auto-fix linting issues and format
make test          # Run tests
make ci-check      # Run all CI checks locally
make clean         # Clean up cache files
```

```bash
make ci-check
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

- https://avandekleut.github.io/vae/

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

  