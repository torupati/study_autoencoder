import os
from pathlib import Path
from typing import Optional

import numpy as np
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
        pil_image = Image.open(img_path).convert("L")  # Convert to grayscale

        if self.transform:
            tensor_image = self.transform(pil_image)
        else:
            # If no transform, manually convert to tensor
            tensor_image = torch.from_numpy(np.array(pil_image)).float() / 255.0
            tensor_image = tensor_image.unsqueeze(0)  # Add channel dimension

        return tensor_image, 0  # Return dummy label for compatibility
