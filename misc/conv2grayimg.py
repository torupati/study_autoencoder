import sys
from pathlib import Path

import numpy as np
from PIL import Image


def convert2grayimg128x128(img: Image.Image) -> np.ndarray:
    """Convert input image to 128x128 grayscale image as NumPy array."""

    gray_img = img.convert("L")

    width, height = gray_img.size
    if width < height:
        new_width = 128
        new_height = int(height * (128 / width))
    else:
        new_height = 128
        new_width = int(width * (128 / height))
    resized_img = gray_img.resize((new_width, new_height), Image.BILINEAR)

    left = (new_width - 128) // 2
    top = (new_height - 128) // 2
    right = left + 128
    bottom = top + 128
    cropped_img = resized_img.crop((left, top, right, bottom))
    img_array = np.array(cropped_img)
    return img_array


def batch_convert_images(target_dir: str, output_dir: str):
    """Convert all JPEG images in target_dir to 128x128 grayscale and save to output_dir."""
    target_path = Path(target_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all JPEG files (case-insensitive)
    jpeg_patterns = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    image_files = []
    for pattern in jpeg_patterns:
        image_files.extend(target_path.glob(pattern))

    if not image_files:
        print(f"No JPEG files found in {target_dir}")
        return

    print(f"Found {len(image_files)} images to convert")

    # Process each image
    for i, img_path in enumerate(image_files, 1):
        output_file = output_path / img_path.name
        if output_file.exists():
            print(f"[{i}/{len(image_files)}] Skipping existing file: {output_file}")
            continue

        try:
            # Load image
            img = Image.open(img_path)

            # Convert to 128x128 grayscale
            img_array = convert2grayimg128x128(img)

            # Convert back to PIL Image for saving
            output_img = Image.fromarray(img_array)

            # Save with same filename
            output_img.save(output_file)

            print(f"[{i}/{len(image_files)}] Converted: {img_path.name} -> {output_file}")

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    print(f"\nConversion complete! {len(image_files)} images saved to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python conv2grayimg.py <target_dir> <output_dir>")
        sys.exit(1)

    target_dir = sys.argv[1]
    output_dir = sys.argv[2]

    batch_convert_images(target_dir, output_dir)
