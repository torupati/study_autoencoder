.PHONY: install dev-install test format lint clean run-autoencoder run-vae help

# Default Python version
PYTHON_VERSION ?= 3.9

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies (CPU-only PyTorch)
	uv venv --python $(PYTHON_VERSION)
	uv sync

dev-install: ## Install with development dependencies (CPU-only PyTorch)
	uv venv --python $(PYTHON_VERSION)
	uv sync --all-groups

install-cpu: ## Force install CPU-only PyTorch (removes any existing CUDA packages)
	rm -rf .venv
	uv venv --python $(PYTHON_VERSION)
	uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
	uv sync

test: ## Run tests
	uv run pytest tests/ -v

format: ## Format code with black and isort
	uv run black .
	uv run isort .

lint: ## Run linting checks
	uv run flake8 .
	uv run mypy .

clean: ## Clean up cache files and virtual environment
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .venv/
	rm -rf dist/
	rm -rf build/

run-autoencoder: ## Run MNIST autoencoder
	uv run python app/run_mnist_autoencoder.py

run-vae: ## Run MNIST variational autoencoder
	uv run python app/run_mnist_v_autoencoder.py

run-test-torch: ## Run PyTorch basic test
	uv run python tests/test_torch_basic.py

shell: ## Activate virtual environment shell
	@echo "Run: source .venv/bin/activate"