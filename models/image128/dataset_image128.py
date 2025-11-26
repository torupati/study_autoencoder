import os
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class GrayscaleImage128Dataset(Dataset):
    """
    Dataset for 128x128 grayscale images
    """

    def __init__(self, image_dir: str, transform: Optional[transforms.Compose] = None):
        """
        Args:
            image_dir: Directory containing images
            transform: Optional transform to apply to images
        """
        self.image_dir = Path(image_dir)
        self.image_files = sorted(
            [
                f
                for f in self.image_dir.glob("*")
                if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
            ]
        )

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("L")  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image, 0  # Return dummy label for compatibility
