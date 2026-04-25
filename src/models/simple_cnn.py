"""Simple CNN architecture for from-scratch training."""

from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """A lightweight CNN for image classification baselines."""

    def __init__(
        self,
        num_classes: int,
        channels: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        channels = channels or [16, 32, 64]

        self.features = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[2], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        return self.classifier(x)
