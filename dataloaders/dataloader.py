from pathlib import Path
from typing import Iterator, Optional, List, Iterable
import zipfile
import io
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
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


def is_image_corrupt(image_data: bytes) -> bool:
    try:
        Image.open(io.BytesIO(image_data)).verify()
        return False
    except (IOError, OSError) as e:
        print(f"Corrupt image detected: {e}")
        return True


class ImageDataset(Dataset):
    def __init__(self, zip_path: str, transform=None):
        self.zip_path = zip_path
        self.transform = transform
        self.image_files: List[str] = []

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            for file in all_files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        with zip_ref.open(file) as img_file:
                            img_data = img_file.read()
                            if not is_image_corrupt(img_data):
                                self.image_files.append(file)
                    except Exception as e:
                        print(f"Skipping {file}: {str(e)}")

        if not self.image_files:
            raise ValueError(f"No valid images found in {zip_path}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                with zip_ref.open(self.image_files[idx]) as img_file:
                    image = Image.open(img_file).convert("RGB")
                    if self.transform:
                        image = self.transform(image)
                    return image
        except Exception as e:
            print(f"Error loading {self.image_files[idx]}: {str(e)}")
            return torch.reflect(3, 128, 128)


class InfiniteSampler(Sampler):
    def __init__(self, dataset_length, shuffle: bool = True):
        super().__init__(None)
        self.shuffle = shuffle
        self.dataset_length = dataset_length
        self.indices = list(range(dataset_length))

    def __iter__(self) -> Iterator[int]:
        while True:
            if self.shuffle:
                indices = torch.randperm(self.dataset_length).tolist()
            else:
                indices = self.indices.copy()

            for idx in indices:
                yield idx

    def __len__(self) -> int:
        return 0


class InfiniteDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = True, **kwargs):
        sampler = InfiniteSampler(len(dataset), shuffle=shuffle)
        super().__init__(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            **kwargs
        )

    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = super().__iter__()