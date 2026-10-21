"""Training and evaluation loop for the logistic regression baseline."""
import torch
from torch.utils.data import DataLoader

from baseline.model import LogisticRegression


def run_epoch(
    model: LogisticRegression,
    loader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, torch.Tensor | None, torch.Tensor | None]:
    """Run one full pass over *loader* in train or eval mode.

    Args:
        model: The :class:`~baseline.model.LogisticRegression` to run.
        loader: DataLoader over ``(uint8_pixels, label)`` batches.
        mean: Per-feature mean from the training split (shape ``(n_features,)``).
        std: Per-feature std from the training split (shape ``(n_features,)``).
        device: Device on which to perform computation.
        optimizer: If provided the function runs in *training* mode and updates
            model weights; otherwise it runs in eval mode.

    Returns:
        ``(mean_loss, probs, labels)`` where *probs* and *labels* are
        ``None`` during training.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, all_probs, all_labels = 0.0, [], []
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb = (xb.to(device, non_blocking=True).float() / 255.0 - mean) / std
            yb = yb.to(device)
            logits = model(xb)
            loss = model.loss(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                all_probs.append(model.probs(logits).cpu())
                all_labels.append(yb.cpu())
            total_loss += loss.item() * len(yb)
    if training:
        return total_loss / len(loader.dataset), None, None
    return total_loss / len(loader.dataset), torch.cat(all_probs), torch.cat(all_labels)


def topk_accuracy(probs: torch.Tensor, y: torch.Tensor, k: int) -> float:
    """Fraction of samples where the true label appears in the top-*k* predictions.

    Args:
        probs: Per-class probability tensor of shape ``(N, num_classes)``.
        y: Ground-truth integer labels of shape ``(N,)``.
        k: Number of top predictions to consider; clamped to ``num_classes``.

    Returns:
        Scalar accuracy in ``[0, 1]``.
    """
    k = min(k, probs.shape[1])
    return (probs.topk(k, dim=1).indices == y.unsqueeze(1)).any(dim=1).float().mean().item()

