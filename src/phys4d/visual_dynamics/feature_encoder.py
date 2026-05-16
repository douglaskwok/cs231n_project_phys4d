"""CNN encoder for per-object appearance (ResNet18)."""

from __future__ import annotations

import torch.nn as nn


class SimpleCropEncoder(nn.Module):
    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_feature_encoder(out_dim: int = 64) -> nn.Module:
    try:
        from torchvision.models import resnet18

        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        modules = list(backbone.children())[:-1]
        return nn.Sequential(*modules, nn.Flatten(), nn.Linear(512, out_dim))
    except ImportError:
        return SimpleCropEncoder(out_dim=out_dim)
