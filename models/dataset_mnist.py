"""
Custom MNIST Dataset implementation without torchvision dependency.
This module reads MNIST binary files directly and provides a PyTorch Dataset interface.

MNIST binary file format:
- Images: 32-bit big-endian integers for magic number, number of images, rows, cols
  followed by unsigned bytes for pixel values
- Labels: 32-bit big-endian integers for magic number, number of labels
  followed by unsigned bytes for label values

References:
- http://yann.lecun.com/exdb/mnist/
"""

import gzip
import os
import struct
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    """
    Custom MNIST Dataset class that reads binary files directly.

    Args:
        root (str): Root directory containing MNIST data files
        train (bool): If True, load training set, otherwise test set
        transform (callable, optional): Transform to apply to images
        target_transform (callable, optional): Transform to apply to targets
        download (bool): Not used, files should already exist
    """

    # MNIST file information
    TRAIN_IMAGES = "train-images-idx3-ubyte"
    TRAIN_LABELS = "train-labels-idx1-ubyte"
    TEST_IMAGES = "t10k-images-idx3-ubyte"
    TEST_LABELS = "t10k-labels-idx1-ubyte"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # Determine file paths
        if train:
            self.images_file = os.path.join(root, "raw", self.TRAIN_IMAGES)
            self.labels_file = os.path.join(root, "raw", self.TRAIN_LABELS)
        else:
            self.images_file = os.path.join(root, "raw", self.TEST_IMAGES)
            self.labels_file = os.path.join(root, "raw", self.TEST_LABELS)

        # Check if files exist, try gzipped versions if not
        if not os.path.exists(self.images_file):
            self.images_file += ".gz"
        if not os.path.exists(self.labels_file):
            self.labels_file += ".gz"

        # Load data
        self.data, self.targets = self._load_data()

    def _read_image_file(self, path: str) -> torch.Tensor:
        """
        Read MNIST image file and return tensor.

        Args:
            path: Path to the image file

        Returns:
            torch.Tensor: Images tensor of shape (n_images, 28, 28)
        """
        if path.endswith(".gz"):
            with gzip.open(path, "rb") as f:
                data = f.read()
        else:
            with open(path, "rb") as f:
                data = f.read()

        # Parse header: magic number (4 bytes), number of images (4 bytes),
        # number of rows (4 bytes), number of columns (4 bytes)
        magic, num_images, rows, cols = struct.unpack(">IIII", data[0:16])

        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in image file. Expected 2051.")

        # Read pixel data
        images = np.frombuffer(data, dtype=np.uint8, offset=16)
        images = images.reshape(num_images, rows, cols)

        # Create a copy to make the array writable
        images = images.copy()

        return torch.from_numpy(images).float()

    def _read_label_file(self, path: str) -> torch.Tensor:
        """
        Read MNIST label file and return tensor.

        Args:
            path: Path to the label file

        Returns:
            torch.Tensor: Labels tensor of shape (n_labels,)
        """
        if path.endswith(".gz"):
            with gzip.open(path, "rb") as f:
                data = f.read()
        else:
            with open(path, "rb") as f:
                data = f.read()

        # Parse header: magic number (4 bytes), number of labels (4 bytes)
        magic, num_labels = struct.unpack(">II", data[0:8])

        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in label file. Expected 2049.")

        # Read label data
        labels = np.frombuffer(data, dtype=np.uint8, offset=8)

        # Create a copy to make the array writable
        labels = labels.copy()

        return torch.from_numpy(labels).long()

    def _load_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load images and labels from files.

        Returns:
            Tuple of (images, labels) tensors
        """
        images = self._read_image_file(self.images_file)
        labels = self._read_label_file(self.labels_file)

        if len(images) != len(labels):
            raise ValueError(
                f"Number of images ({len(images)}) != number of labels ({len(labels)})"
            )

        return images, labels

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            index: Index of the sample

        Returns:
            Tuple of (image, target) where image is a torch.Tensor and target is an int
        """
        img, target = self.data[index], int(self.targets[index])

        # Convert to PIL-like format (add channel dimension for consistency with torchvision)
        img = img.unsqueeze(0)  # Add channel dimension: (28, 28) -> (1, 28, 28)

        # Normalize to [0, 1] range
        img = img / 255.0

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __repr__(self) -> str:
        """String representation of the dataset."""
        head = "Dataset " + self.__class__.__name__
        body = f"Number of datapoints: {self.__len__()}"
        body += f"\nRoot location: {self.root}"
        body += f"\nSplit: {'Train' if self.train else 'Test'}"
        return "\n".join([head, body])


def get_mnist_dataset(
    root: str = "./data/MNIST",
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> MNISTDataset:
    """
    Convenience function to create MNIST dataset.

    Args:
        root: Root directory containing MNIST data
        train: Whether to load training or test set
        transform: Transform to apply to images
        target_transform: Transform to apply to targets

    Returns:
        MNISTDataset instance
    """
    return MNISTDataset(
        root=root, train=train, transform=transform, target_transform=target_transform
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Testing MNIST Dataset...")

    # Test training set
    train_dataset = get_mnist_dataset("./data/MNIST", train=True)
    print(f"Training set: {len(train_dataset)} samples")

    # Test test set
    test_dataset = get_mnist_dataset("./data/MNIST", train=False)
    print(f"Test set: {len(test_dataset)} samples")

    # Test sample access
    img, label = train_dataset[0]
    print(f"First sample: image shape {img.shape}, label {label}")
    print(f"Image value range: [{img.min():.3f}, {img.max():.3f}]")

    # Test with DataLoader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    batch_imgs, batch_labels = next(iter(dataloader))
    print(f"Batch: images {batch_imgs.shape}, labels {batch_labels.shape}")

    print("✓ MNIST Dataset test completed successfully!")
