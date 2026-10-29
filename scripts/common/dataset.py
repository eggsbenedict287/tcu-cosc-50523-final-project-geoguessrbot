"""Shared Dataset classes and DataLoader factories for the streetview dataset.

Two dataset variants live here so every pipeline imports from one place:

``ImageDataset``      – returns flat uint8 pixel vectors for the baseline
                        logistic regression (no transforms, low overhead).
``TransformDataset``  – returns ``(3, H, W)`` float tensors after applying a
                        torchvision transform; used by CNN pipelines.

Images are always streamed from disk per batch so that large splits do not
exhaust RAM.
"""
from pathlib import Path
from typing import Callable

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


class TransformDataset(Dataset):
    """Loads images from disk and applies a torchvision transform.

    Returns proper ``(3, H, W)`` float tensors suitable for convolutional
    layers, as opposed to the flat uint8 vectors of :class:`ImageDataset`.

    Args:
        items:     List of ``(path, integer_label)`` tuples.
        transform: Callable applied to each PIL image (e.g., a
                   :class:`torchvision.transforms.Compose` pipeline).
    """

    def __init__(self, items: list[tuple[Path, int]], transform: Callable) -> None:
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[i]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.transform(img), label


def make_cnn_loader(
    items: list[tuple[Path, int]],
    transform: Callable,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool = False,
) -> DataLoader:
    """Wrap *items* in a :class:`TransformDataset` and return a DataLoader.

    Args:
        items:       ``(path, label)`` pairs from
                     :func:`~common.preprocessing.split_per_country`.
        transform:   Torchvision transform applied to each PIL image.
        batch_size:  Samples per batch.
        num_workers: DataLoader worker processes.
        seed:        Seed for the shuffle generator (reproducibility).
        shuffle:     Whether to shuffle each epoch.

    Returns:
        A configured :class:`torch.utils.data.DataLoader`.
    """
    return DataLoader(
        TransformDataset(items, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )

