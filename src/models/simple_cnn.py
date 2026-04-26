"""Simple CNN architecture and HPO helpers."""

from __future__ import annotations

import itertools
import random
from typing import Any

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


def _ensure_sequence(value: Any) -> list[Any]:
    """Ensure a value is represented as a list for search spaces."""
    if isinstance(value, list):
        return value
    return [value]


def generate_hparam_candidates(
    *,
    base_hparams: dict[str, Any],
    search_config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Create hyperparameter candidates via grid or random search."""
    if not search_config or not search_config.get("enabled", False):
        return [dict(base_hparams)]

    strategy = str(search_config.get("strategy", "grid")).lower()
    param_grid = dict(search_config.get("param_grid", {}))
    max_trials = search_config.get("max_trials")

    if not param_grid:
        return [dict(base_hparams)]

    grid_keys = list(param_grid.keys())
    grid_values = [_ensure_sequence(param_grid[key]) for key in grid_keys]
    product = list(itertools.product(*grid_values))

    all_candidates: list[dict[str, Any]] = []
    for values in product:
        trial_hparams = dict(base_hparams)
        for key, value in zip(grid_keys, values):
            trial_hparams[key] = value
        all_candidates.append(trial_hparams)

    if strategy == "random":
        n_trials = int(search_config.get("n_trials", min(10, len(all_candidates))))
        n_trials = max(1, min(n_trials, len(all_candidates)))
        rng = random.Random(int(search_config.get("seed", 42)))
        all_candidates = rng.sample(all_candidates, k=n_trials)
    elif strategy != "grid":
        raise ValueError(f"Unsupported hparam search strategy: {strategy}")

    if max_trials is not None:
        all_candidates = all_candidates[: max(1, int(max_trials))]

    return all_candidates
