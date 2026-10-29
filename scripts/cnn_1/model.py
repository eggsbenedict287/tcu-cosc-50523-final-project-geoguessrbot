"""ResNet-18 style CNN for country classification, trained from scratch.

Architecture (input: 3 × 320 × 320):

    Stem      7×7 conv, stride 2, padding 3 → BN → ReLU → 3×3 max-pool stride 2
    Stage 1   2 residual blocks, 64 channels,  stride 1   → 64  × 80  × 80
    Stage 2   2 residual blocks, 128 channels, stride 2   → 128 × 40  × 40
    Stage 3   2 residual blocks, 256 channels, stride 2   → 256 × 20  × 20
    Stage 4   2 residual blocks, 512 channels, stride 2   → 512 × 10  × 10
    Head      Global avg-pool → Dropout(0.3) → Linear(512, num_classes)

Global average pooling fixes the head input at 512 regardless of input
resolution, so the model can be evaluated at different resolutions without
re-training.

Weight initialization
---------------------
  Conv    : Kaiming normal, fan-out mode (good for ReLU chains)
  BatchNorm: weight=1, bias=0  (identity at init)
  Linear  : PyTorch default (Kaiming uniform)

~11.2 M parameters, ~3.7 GMACs at 320×320.
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Two-conv residual block with an optional downsampling shortcut.

    Structure::

        x ──► conv3×3 → BN → ReLU → conv3×3 → BN ──► (+) → ReLU → output
        │                                               ▲
        └───────────── shortcut (identity or 1×1 conv) ─┘

    When *stride* > 1 or *in_channels* ≠ *out_channels* the shortcut
    uses a 1×1 conv (+ BN) so the shapes match before the addition.

    Args:
        in_channels:  Number of input feature maps.
        out_channels: Number of output feature maps.
        stride:       Stride for the first 3×3 conv. Pass ``2`` to halve
                      the spatial dimensions.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv_a = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=stride, padding=1, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_b = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # Identity shortcut when shape is unchanged; 1×1 conv otherwise.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_b(self.conv_a(x))
        return self.relu(out + self.shortcut(x))


def _make_stage(
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    stride: int,
) -> nn.Sequential:
    """Build one ResNet stage.

    The first block uses *stride* (may downsample); the remaining blocks
    always use stride 1.
    """
    blocks = [ResidualBlock(in_channels, out_channels, stride=stride)]
    for _ in range(1, num_blocks):
        blocks.append(ResidualBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*blocks)


class ResNet18Country(nn.Module):
    """ResNet-18 style classifier for N-country image classification.

    Args:
        num_classes: Number of output country classes.
        dropout:     Dropout probability applied before the linear head.
    """

    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()

        # --- Stem ---
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # --- Residual stages ---
        self.stage1 = _make_stage(64,  64,  num_blocks=2, stride=1)
        self.stage2 = _make_stage(64,  128, num_blocks=2, stride=2)
        self.stage3 = _make_stage(128, 256, num_blocks=2, stride=2)
        self.stage4 = _make_stage(256, 512, num_blocks=2, stride=2)

        # --- Classification head ---
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (N, 512, H, W) → (N, 512, 1, 1)
            nn.Flatten(),              # → (N, 512)
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal for all Conv2d layers; ones/zeros for BatchNorm."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)

