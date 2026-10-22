"""Dataset indexing, stratified splitting, and pixel-level feature statistics.

These utilities are model-agnostic and can be reused by any pipeline that
consumes the streetview-by-country dataset layout (one folder per country,
folder name = two-letter country code).
"""
import random
from collections import defaultdict
from pathlib import Path

import torch

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def index_dataset(data_dir: Path) -> dict[str, list[Path]]:
    """Return a mapping of country code -> sorted list of image paths.

    The label for each image is its parent folder name (e.g. ``FR``, ``JP``).
    Raises ``SystemExit`` when no images are found so the error surfaces
    cleanly whether the function is called from a script or a notebook.
    """
    by_country: dict[str, list[Path]] = defaultdict(list)
    for path in data_dir.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTS:
            by_country[path.parent.name].append(path)
    if not by_country:
        raise SystemExit(
            f"No images found under {data_dir}. "
            "Run the dataset cell in notebooks/init.ipynb first."
        )
    return {c: sorted(paths) for c, paths in sorted(by_country.items())}


def split_per_country(
    by_country: dict[str, list[Path]],
    countries: list[str],
    val_frac: float,
    test_frac: float,
    max_per_class: int | None,
    rng: random.Random,
) -> dict[str, list[tuple[Path, int]]]:
    """Stratified split: shuffle each country's images, cut into train/val/test.

    Args:
        by_country: Output of :func:`index_dataset`.
        countries: Ordered list of country codes to include; index == label.
        val_frac: Fraction of each country's images reserved for validation.
        test_frac: Fraction reserved for the held-out test set.
        max_per_class: If set, cap the number of images drawn per country.
        rng: Seeded ``random.Random`` instance for reproducibility.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"``; each value is a
        list of ``(path, integer_label)`` tuples.
    """
    splits: dict[str, list[tuple[Path, int]]] = {"train": [], "val": [], "test": []}
    for label, country in enumerate(countries):
        paths = list(by_country[country])
        rng.shuffle(paths)
        if max_per_class:
            paths = paths[:max_per_class]
        n_test = int(round(len(paths) * test_frac))
        n_val = int(round(len(paths) * val_frac))
        splits["test"] += [(p, label) for p in paths[:n_test]]
        splits["val"] += [(p, label) for p in paths[n_test : n_test + n_val]]
        splits["train"] += [(p, label) for p in paths[n_test + n_val :]]
    return splits


def feature_stats(
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean and std over a data loader in a single streaming pass.

    Uses only the training loader so that val/test statistics are never leaked
    into the normalization parameters.

    Args:
        loader: DataLoader whose batches are ``(uint8_pixels, label)`` tensors.
        device: Device on which to accumulate statistics.

    Returns:
        ``(mean, std)`` float32 tensors of shape ``(n_features,)``.
        ``std`` is clipped to a minimum of ``1e-6`` to avoid division by zero.
    """
    total, total_sq, n = None, None, 0
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True).double() / 255.0
        total = xb.sum(0) if total is None else total + xb.sum(0)
        total_sq = (xb**2).sum(0) if total_sq is None else total_sq + (xb**2).sum(0)
        n += len(xb)
    mean = total / n
    std = (total_sq / n - mean**2).clamp_min(0).sqrt()
    return mean.float(), std.float() + 1e-6

