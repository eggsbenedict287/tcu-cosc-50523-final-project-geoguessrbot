"""Training and evaluation loop for the ResNet-18 CNN.

Features
--------
* **Mixed precision** via ``torch.amp.autocast`` + ``GradScaler`` (GPU only).
* **Channels-last** memory format for better GPU tensor core utilization.
* Per-step **OneCycleLR** scheduler support (step called after every batch).
* **Label smoothing** is baked into the loss criterion passed in from outside.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cnn_1.model import ResNet18Country


def run_epoch(
    model: ResNet18Country,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    scheduler=None,
) -> tuple[float, torch.Tensor | None, torch.Tensor | None]:
    """Run one full pass over *loader* in train or eval mode.

    Args:
        model:     The CNN to run.
        loader:    DataLoader of ``(image_tensor, label)`` batches.
        device:    Torch device.
        criterion: Loss function (e.g., ``CrossEntropyLoss`` with
                   ``label_smoothing=0.1``).
        optimizer: If provided, runs in *training* mode with backprop.
                   Pass ``None`` for validation/test.
        scaler:    :class:`~torch.cuda.amp.GradScaler` for mixed-precision.
                   Pass ``None`` for CPU or full-precision runs.
        scheduler: Per-step LR scheduler (e.g., ``OneCycleLR``).
                   ``.step()`` is called after every training batch.

    Returns:
        ``(mean_loss, probs, labels)`` — *probs* and *labels* are ``None``
        during training.
    """
    training = optimizer is not None
    model.train(training)
    amp_enabled = scaler is not None and device.type == "cuda"
    total_loss, all_probs, all_labels = 0.0, [], []

    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            # channels_last improves GPU memory throughput for conv-heavy nets
            xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(xb)
                loss = criterion(logits, yb)

            if training:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            else:
                all_probs.append(torch.softmax(logits.float(), dim=1).cpu())
                all_labels.append(yb.cpu())

            total_loss += loss.item() * len(yb)

    if training:
        return total_loss / len(loader.dataset), None, None
    return total_loss / len(loader.dataset), torch.cat(all_probs), torch.cat(all_labels)


def topk_accuracy(probs: torch.Tensor, y: torch.Tensor, k: int) -> float:
    """Fraction of samples where the true label appears in the top-*k* predictions.

    Args:
        probs: Per-class probability tensor of shape ``(N, num_classes)``.
        y:     Ground-truth integer labels of shape ``(N,)``.
        k:     Number of top predictions; clamped to ``num_classes``.

    Returns:
        Scalar accuracy in ``[0, 1]``.
    """
    k = min(k, probs.shape[1])
    return (probs.topk(k, dim=1).indices == y.unsqueeze(1)).any(dim=1).float().mean().item()

