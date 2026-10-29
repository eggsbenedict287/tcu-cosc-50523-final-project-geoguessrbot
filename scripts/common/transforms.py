"""Torchvision transform pipelines for the CNN training and evaluation.

Train
-----
    RandomResizedCrop(320, scale=(0.35, 1.0))
    ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
    RandomHorizontalFlip()   ← ablation switch (``use_hflip``)
    ToTensor + Normalize

Val / Test
----------
    Resize(360)   ← assumes images were pre-resized to 360×360 (or larger)
    CenterCrop(320)
    ToTensor + Normalize

Normalization uses the per-channel mean and std computed from the training
split only (see :func:`~common.preprocessing.channel_stats`).
"""
import torchvision.transforms as T


def get_train_transform(
    mean: list[float],
    std: list[float],
    crop_size: int = 320,
    use_hflip: bool = True,
) -> T.Compose:
    """Build the training augmentation pipeline.

    Args:
        mean:      Per-channel (R, G, B) mean in [0, 1].
        std:       Per-channel (R, G, B) std in [0, 1].
        crop_size: Final square side length after RandomResizedCrop.
        use_hflip: Whether to apply random horizontal flips.  Turn off to
                   ablate: flips swap driving side and mirror text.

    Returns:
        A :class:`torchvision.transforms.Compose` pipeline.
    """
    ops = [
        T.RandomResizedCrop(crop_size, scale=(0.35, 1.0)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    ]
    if use_hflip:
        ops.append(T.RandomHorizontalFlip())
    ops += [T.ToTensor(), T.Normalize(mean=mean, std=std)]
    return T.Compose(ops)


def get_eval_transform(
    mean: list[float],
    std: list[float],
    resize_to: int = 360,
    crop_size: int = 320,
) -> T.Compose:
    """Build the validation / test transform (no augmentation).

    Args:
        mean:      Per-channel (R, G, B) mean in [0, 1].
        std:       Per-channel (R, G, B) std in [0, 1].
        resize_to: Resize the shorter edge to this value before cropping.
                   Use 360 when images are the 360×360 pre-resized versions.
        crop_size: Final square side length after CenterCrop.

    Returns:
        A :class:`torchvision.transforms.Compose` pipeline.
    """
    return T.Compose([
        T.Resize(resize_to),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

