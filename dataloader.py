from pathlib import Path
from typing import Iterator, Optional
import zipfile
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def get_transform(
    resize: Optional[int] = None, crop_size: Optional[int] = None
) -> transforms.Compose:
    transform_list = []
    if resize:
        transform_list.append(
            transforms.Resize(size=resize, interpolation=Image.Resampling.NEAREST)
        )
    if crop_size:
        transform_list.append(transforms.CenterCrop(size=crop_size))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def is_image_corrupt(image_path: str) -> bool:
    try:
        with Image.open(image_path) as img:
            img.load()  # Verifies the image data
        return False
    except (IOError, OSError) as e:
        print(f"Error opening image: {e}")
        return True


class ImageDataset(Dataset):
    def __init__(self, zip_path, transform=None):
        self.zip_path = zip_path
        self.transform = transform
        self.image_files = []
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(file)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                with zip_ref.open(self.image_files[idx]) as img_file:
                    image = Image.open(img_file).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.image_files))

        if self.transform:
            image = self.transform(image)

        return image


class InfiniteDataLoader(DataLoader):
    def __iter__(self):
        """
        Infinite iteration over the DataLoader batches.
        Restarts when the iterator is exhausted.
        """
        while True:
            for batch in super().__iter__():  # Use the DataLoader's original iterator
                # Skip the batch if it is smaller than the desired batch size
                if len(batch) == self.batch_size:
                    yield batch
            # Reset the iterator and start over when all batches are consumed