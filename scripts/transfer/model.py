"""Pretrained torchvision CNNs with a new country-classification head (transfer learning).

Transfer learning = start from a network already trained on ImageNet (1.28M images,
1000 classes). Its early layers detect generic things (edges, textures, shapes) that
are useful for *any* photo, so we keep them and only learn what is specific to our
task: which country a street-view photo is from. We swap the 1000-way ImageNet
classifier for a fresh ``Linear(features -> num_classes)`` and then train either
just that head or the whole network at a small learning rate.

Supported backbones (one per teammate)
--------------------------------------
    resnet18             11M params  residual blocks with skip connections
    efficientnet_b0       4M params  MBConv blocks (depthwise conv + squeeze-and-excite)
    mobilenet_v3_large    4M params  inverted residuals, h-swish, light squeeze-and-excite

Freeze modes (the ``--freeze`` flag)
------------------------------------
    none      train everything (full fine-tuning)
    partial   freeze the *early* layers (generic features), train the later layers + head
    all       freeze the whole backbone, train only the head ("linear probe")

Why BatchNorm needs special care when freezing
----------------------------------------------
``requires_grad=False`` stops the *weights* from changing, but a BatchNorm layer also
keeps running mean/variance statistics that are updated by every forward pass in
``train()`` mode. A "frozen" layer would still drift. :class:`TransferNet` therefore
puts the BatchNorm layers of frozen parts back into ``eval()`` mode every time
``train()`` is called, so frozen really means frozen.
"""
import torch
import torch.nn as nn
import torchvision
from torch.nn.modules.batchnorm import _BatchNorm

# name of the head module + which top-level modules count as "early" for --freeze partial.
# Partial freezing keeps roughly the first half of each network fixed:
#   resnet18:            stem + layer1 + layer2 frozen; layer3, layer4, fc trained
#   efficientnet_b0:     features[0..4] (stem + first 4 MBConv stages) frozen; features[5..8] + classifier trained
#   mobilenet_v3_large:  features[0..8] frozen; features[9..16] + classifier trained
BACKBONES: dict[str, dict] = {
    "resnet18": {
        "head": "fc",
        "early": ("conv1", "bn1", "layer1", "layer2"),
    },
    "efficientnet_b0": {
        "head": "classifier",
        "early": tuple(f"features.{i}" for i in range(5)),
    },
    "mobilenet_v3_large": {
        "head": "classifier",
        "early": tuple(f"features.{i}" for i in range(9)),
    },
}

FREEZE_MODES = ("none", "partial", "all")

# Statistics the torchvision ImageNet weights were trained with. Pretrained models
# expect inputs normalised exactly like this, so we use them instead of statistics
# computed from our own training images.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _in_prefixes(name: str, prefixes: tuple[str, ...]) -> bool:
    """True if parameter/module *name* is one of *prefixes* or lives inside one.

    The ``"."`` matters: ``features.1`` must not match ``features.10``.
    """
    return any(name == p or name.startswith(p + ".") for p in prefixes)


class TransferNet(nn.Module):
    """A pretrained backbone with a new ``num_classes``-way head.

    Args:
        backbone:    One of :data:`BACKBONES`.
        num_classes: Number of countries.
        pretrained:  Load ImageNet weights (True) or random init (False, for ablations).
        dropout:     Dropout probability right before the final Linear layer.
        freeze:      ``"none"``, ``"partial"`` or ``"all"`` (see module docstring).
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze: str = "none",
    ) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r}. Choices: {sorted(BACKBONES)}")
        if freeze not in FREEZE_MODES:
            raise ValueError(f"freeze must be one of {FREEZE_MODES}, got {freeze!r}")
        self.backbone_name = backbone
        self.freeze_mode = freeze

        # "DEFAULT" = the best ImageNet weights torchvision ships for this model.
        self.net = torchvision.models.get_model(backbone, weights="DEFAULT" if pretrained else None)
        self._replace_head(num_classes, dropout)
        self._apply_freeze()

    # ------------------------------------------------------------------
    def _replace_head(self, num_classes: int, dropout: float) -> None:
        """Swap the 1000-way ImageNet classifier for a fresh num_classes-way one."""
        net = self.net
        if self.backbone_name == "resnet18":
            net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, num_classes))
        elif self.backbone_name == "efficientnet_b0":
            # original: classifier = [Dropout(0.2), Linear(1280, 1000)]
            net.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(net.classifier[1].in_features, num_classes)
            )
        else:  # mobilenet_v3_large
            # original: classifier = [Linear(960,1280), Hardswish, Dropout(0.2), Linear(1280,1000)]
            # Keep the pretrained hidden Linear; replace only the dropout rate and last layer.
            net.classifier[2] = nn.Dropout(dropout)
            net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_classes)

    def _apply_freeze(self) -> None:
        """Set ``requires_grad`` per the freeze mode and remember which BN layers are frozen."""
        spec = BACKBONES[self.backbone_name]
        head, early = (spec["head"],), spec["early"]

        for name, param in self.net.named_parameters():
            if self.freeze_mode == "none":
                param.requires_grad = True
            elif self.freeze_mode == "all":
                param.requires_grad = _in_prefixes(name, head)
            else:  # partial
                param.requires_grad = not _in_prefixes(name, early)

        # A BN layer is "frozen" if none of its parameters are trained.
        # Kept in a plain list (not registered as a submodule) on purpose.
        self._frozen_bn = [
            m
            for m in self.net.modules()
            if isinstance(m, _BatchNorm) and not any(p.requires_grad for p in m.parameters())
        ]

    # ------------------------------------------------------------------
    def train(self, mode: bool = True) -> "TransferNet":
        """Like ``nn.Module.train`` but keeps frozen BatchNorm layers in eval mode."""
        super().train(mode)
        if mode:
            for m in self._frozen_bn:
                m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def head_parameters(self) -> set[int]:
        """``id`` of every parameter belonging to the new head (for a separate learning rate)."""
        head = getattr(self.net, BACKBONES[self.backbone_name]["head"])
        return {id(p) for p in head.parameters()}


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(total, trainable)`` parameter counts. Quote these in the report."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_param_groups(
    model: TransferNet,
    lr: float,
    weight_decay: float,
    head_lr_mult: float = 1.0,
) -> list[dict]:
    """Build AdamW parameter groups: (backbone | head) x (decay | no decay).

    * Frozen parameters are left out entirely.
    * **Weight decay is not applied to 1-D parameters** (biases and BatchNorm scale/shift).
      Decaying those has no regularising benefit and can hurt. ``param.ndim <= 1``
      catches them all regardless of what they are named. (The older ``cnn_1`` code
      matched the substring ``"bn"`` in the name, which missed layers named like
      ``stem.1.weight``.)
    * The head can use a larger learning rate (``head_lr_mult``) because it starts from
      random weights while the backbone only needs gentle adjustment.

    Each group carries its own ``lr``, which OneCycleLR uses as that group's peak LR.
    """
    head_ids = model.head_parameters()
    buckets: dict[tuple[bool, bool], list] = {}
    for param in model.parameters():
        if not param.requires_grad:
            continue
        key = (id(param) in head_ids, param.ndim <= 1)  # (is_head, is_no_decay)
        buckets.setdefault(key, []).append(param)

    groups = []
    for (is_head, no_decay), params in sorted(buckets.items()):
        groups.append(
            {
                "params": params,
                "lr": lr * (head_lr_mult if is_head else 1.0),
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        )
    return groups
