import struct
from pathlib import Path

import numpy as np

from models.datasets import get_mnist_dataset


def _write_idx_image_file(path: Path, images: np.ndarray) -> None:
    """Write MNIST IDX image file (big-endian) to path."""
    # images shape: (n, rows, cols), dtype uint8
    n, rows, cols = images.shape
    with open(path, "wb") as f:
        # magic number for images: 2051
        f.write(struct.pack(">IIII", 2051, n, rows, cols))
        f.write(images.tobytes())


def _write_idx_label_file(path: Path, labels: np.ndarray) -> None:
    """Write MNIST IDX label file (big-endian) to path."""
    n = labels.shape[0]
    with open(path, "wb") as f:
        # magic number for labels: 2049
        f.write(struct.pack(">II", 2049, n))
        f.write(labels.tobytes())


def test_mnist_smoke(tmp_path: Path) -> None:
    # Create minimal MNIST-like data
    n = 3
    rows = cols = 28
    images = (np.random.rand(n, rows, cols) * 255).astype(np.uint8)
    labels = (np.arange(n) % 10).astype(np.uint8)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    img_path = raw_dir / "train-images-idx3-ubyte"
    lbl_path = raw_dir / "train-labels-idx1-ubyte"

    _write_idx_image_file(img_path, images)
    _write_idx_label_file(lbl_path, labels)

    ds = get_mnist_dataset(root=str(tmp_path), train=True)

    assert len(ds) == n
    img, label = ds[0]
    # shape (1,28,28)
    assert hasattr(img, "shape")
    assert tuple(img.shape) == (1, rows, cols)
    # values normalized to [0,1]
    assert float(img.min()) >= 0.0
    assert float(img.max()) <= 1.0
    assert isinstance(label, int)
