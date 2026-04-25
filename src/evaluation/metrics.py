"""Evaluation metric helpers."""

from __future__ import annotations

import torch


def topk_from_logits(logits: torch.Tensor, k: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k probabilities and indices from raw logits."""
    probabilities = torch.softmax(logits, dim=1)
    return torch.topk(probabilities, k=min(k, probabilities.shape[1]), dim=1)
