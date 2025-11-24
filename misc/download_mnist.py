#!/usr/bin/env python3
"""
MNIST Dataset Downloader

This script downloads the MNIST dataset from PyTorch's official S3 mirror.

The MNIST database of handwritten digits has a training set of 60,000 examples,
and a test set of 10,000 examples.

Usage:
    python misc/download_mnist.py --path ./data
"""

import argparse
import gzip
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PyTorch's official S3 mirror (most reliable)
BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def download_mnist(data_path: Path = Path("./data")):
    """Download MNIST dataset files from PyTorch's S3 mirror."""
    mnist_path = data_path / "MNIST" / "raw"
    mnist_path.mkdir(parents=True, exist_ok=True)

    logger.info("MNIST Dataset Downloader")
    logger.info("=" * 50)
    logger.info(f"MNIST data will be stored in: {mnist_path}")

    for filename in FILES:
        file_path = mnist_path / filename

        if file_path.exists():
            logger.info(f"File already exists: {filename}")
            continue

        url = BASE_URL + filename
        logger.info(f"Downloading {url} to {file_path}")

        try:
            urlretrieve(url, file_path)
            logger.info(f"Successfully downloaded {filename}")
        except (URLError, HTTPError) as e:
            logger.error(f"Failed to download {url}: {e}")
            logger.error(f"Failed to download {filename}")
            return False

    logger.info("All MNIST files downloaded successfully!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MNIST dataset")
    parser.add_argument("--path", type=str, default="./data", help="Path to save MNIST data")
    args = parser.parse_args()

    download_mnist(Path(args.path))
