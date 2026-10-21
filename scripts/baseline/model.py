"""Logistic regression model definition for the GeoGuessr baseline.

A single linear layer with no hidden units — deliberately the simplest possible
classifier so that any CNN model has a clear baseline to beat.

Supports both binary (2-country) and multiclass (N-country) classification:
  * Binary     → sigmoid output + binary cross-entropy loss
  * Multiclass → softmax output + cross-entropy loss
"""
import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """Single linear layer classifier (the \"skeleton\" baseline).

    Args:
        in_features: Number of input features (``size * size * 3`` for flat pixels).
        num_classes: Number of country classes. Pass ``2`` for binary mode.
    """

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.binary = num_classes == 2
        self.linear = nn.Linear(in_features, 1 if self.binary else num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        return out.squeeze(1) if self.binary else out

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the appropriate loss for binary or multiclass output."""
        if self.binary:
            return nn.functional.binary_cross_entropy_with_logits(logits, y.float())
        return nn.functional.cross_entropy(logits, y)

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Per-class probabilities, shape ``(N, num_classes)``, for both modes."""
        if self.binary:
            p1 = torch.sigmoid(logits)
            return torch.stack([1 - p1, p1], dim=1)
        return torch.softmax(logits, dim=1)

