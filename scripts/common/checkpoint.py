
"""Model checkpoint helpers shared across all model pipelines.

Checkpoints use the standard PyTorch format: a dict saved with
:func:`torch.save` whose ``"model_state_dict"`` entry can be passed straight
to :meth:`torch.nn.Module.load_state_dict`.

Layout on disk::

    weights/
      <model>/
        <run_name>/              e.g. gwtnet_10cls_320px_lr0.001_bs64_wd0.0001_ep50_seed42
          checkpoint.pt          torch.save(dict) — weights + everything needed to rebuild
          config.json            human-readable copy of the run's metadata (no tensors)

Checkpoint dict keys
--------------------
    model_name        Pipeline identifier, e.g. ``"cnn1_gwtnet"``.
    model_class       Class name, e.g. ``"GWTNet"``.
    model_kwargs      Constructor kwargs, so ``model_class(**model_kwargs)`` rebuilds it.
    model_state_dict  Weights (on CPU).
    epoch             Epoch the saved weights come from.
    countries         Ordered class names (index == integer label).
    config            Full CLI config of the run.
    metrics           Summary val/test metrics.
    normalization     Mean/std used to normalize inputs (needed for inference).
    optimizer_state_dict  Only present if an optimizer was passed in.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

CHECKPOINT_FILE = "checkpoint.pt"
CONFIG_FILE = "config.json"


def build_run_name(prefix: str, hparams: dict) -> str:
    """Join a model prefix and hyperparameters into a folder-safe run name.

    Example:
        ``build_run_name("gwtnet", {"cls": 10, "lr": 1e-3})`` →
        ``"gwtnet_cls10_lr0.001"``
    """
    parts = [prefix]
    for key, value in hparams.items():
        if isinstance(value, bool):
            value = "on" if value else "off"
        elif isinstance(value, float):
            value = f"{value:g}"
        parts.append(f"{key}{value}")
    return "_".join(parts)


def _to_cpu(obj):
    """Recursively move tensors to CPU so checkpoints load on any device."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def _jsonable(obj):
    """Drop tensors (summarizing their shape) so metadata can go in config.json."""
    if isinstance(obj, torch.Tensor):
        return f"<tensor shape={list(obj.shape)}>"
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_checkpoint(
    run_dir: Path,
    model: nn.Module,
    *,
    model_name: str,
    model_kwargs: dict,
    epoch: int | None = None,
    countries: list[str] | None = None,
    config: dict | None = None,
    metrics: dict | None = None,
    normalization: dict | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    """Save *model* (and run metadata) to ``run_dir/checkpoint.pt``.

    Args:
        run_dir:       Folder for this run; created if missing.
        model:         Trained model whose current weights are saved.
        model_name:    Pipeline identifier stored for reference.
        model_kwargs:  Constructor kwargs needed to rebuild the model.
        epoch:         Epoch the weights come from.
        countries:     Ordered class names.
        config:        CLI/hyperparameter config of the run.
        metrics:       Summary metrics to store alongside the weights.
        normalization: Input normalization stats (e.g. ``{"mean": ..., "std": ...}``).
        optimizer:     Optional optimizer whose state is saved for resuming.

    Returns:
        Path to the written ``checkpoint.pt``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "model_name": model_name,
        "model_class": type(model).__name__,
        "model_kwargs": model_kwargs,
        "epoch": epoch,
        "countries": countries,
        "config": _jsonable(config),
        "metrics": metrics,
        "normalization": _to_cpu(normalization),
    }
    checkpoint = {**meta, "model_state_dict": _to_cpu(model.state_dict())}
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = _to_cpu(optimizer.state_dict())

    path = run_dir / CHECKPOINT_FILE
    torch.save(checkpoint, path)
    (run_dir / CONFIG_FILE).write_text(json.dumps(_jsonable(meta), indent=2))
    return path


def load_checkpoint(
    path: Path,
    model: nn.Module | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load a checkpoint saved by :func:`save_checkpoint`.

    Args:
        path:         ``checkpoint.pt`` file, or the run folder containing it.
        model:        If given, its weights are loaded from the checkpoint in place.
        map_location: Device to map tensors onto.

    Returns:
        The full checkpoint dict.
    """
    path = Path(path)
    if path.is_dir():
        path = path / CHECKPOINT_FILE
    # weights_only=False: the checkpoint holds plain Python metadata alongside tensors.
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
