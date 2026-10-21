"""Generic PyTorch Dataset and DataLoader factory for the streetview dataset.

Images are streamed from disk per batch so that large splits (640x640 RGB =
1.2 M features/image) do not exhaust RAM.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    """Streams images from disk as flat uint8 pixel vectors ``(size*size*3,)``.

    At 640×640 each image is 1.2 M features, far too large to preload a whole
    split into RAM, so images are decoded per-batch by DataLoader workers.

    Args:
        items: List of ``(path, integer_label)`` tuples.
        size: Target square side length in pixels; images are resized if needed.
    """

    def __init__(self, items: list[tuple[Path, int]], size: int) -> None:
        self.items = items
        self.size = size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[i]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.size != (self.size, self.size):
                img = img.resize((self.size, self.size), Image.BILINEAR)
            return torch.from_numpy(np.array(img, dtype=np.uint8).reshape(-1)), label


def make_loader(
    items: list[tuple[Path, int]],
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool = False,
) -> DataLoader:
    """Wrap *items* in an :class:`ImageDataset` and return a configured DataLoader.

    Args:
        items: ``(path, label)`` pairs as returned by
            :func:`~scripts.common.preprocessing.split_per_country`.
        image_size: Square side length passed to :class:`ImageDataset`.
        batch_size: Number of samples per batch.
        num_workers: DataLoader worker processes (Colab has 2 CPUs).
        seed: Used to seed the shuffle generator for reproducibility.
        shuffle: Whether to shuffle the dataset each epoch.

    Returns:
        A ready-to-iterate :class:`torch.utils.data.DataLoader`.
    """
    return DataLoader(
        ImageDataset(items, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )

