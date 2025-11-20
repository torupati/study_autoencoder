# study_autoencoder

These code are prepared for my learning of variational autoencoder.
Code are work with MNIST dataset and specialised for this dataset.

For comparison, autoencoder is also prepared. Each sample of MNIST can be converted to PNG file.


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
uv sync --all-groups --index-url https://download.pytorch.org/whl/cpu
```

**Important**: The code will automatically use CPU device since CUDA is not available. The device detection in the code will set `device = 'cpu'`.

Run tests:

```bash
uv run pytest tests/
```

Format code:

```bash
uv run black .
uv run isort .
```

#### Legacy Poetry Usage (deprecated)

If you prefer to use Poetry:

```bash
$ pyenv local 3.9.17
$ poetry env use 3.9.17
$ poetry install
$ poetry run python app/run_mnist_autoencoder.py
```


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

  