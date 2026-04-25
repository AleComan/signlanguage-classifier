"""Inference utilities for Torch checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from src.evaluation.metrics import topk_from_logits
from src.models.simple_cnn import SimpleCNN
from src.models.transfer import build_resnet18_finetune


def load_torch_model(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[torch.nn.Module, list[str], int]:
    """Load a torch model checkpoint and rebuild architecture from metadata."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload: dict[str, Any] = torch.load(path, map_location=device)
    model_type = payload.get("model_type")
    class_names = payload.get("class_names", [])
    image_size = int(payload.get("image_size", 224))
    num_classes = len(class_names)

    if model_type == "simple_cnn":
        channels = payload.get("channels", [16, 32, 64])
        dropout = payload.get("dropout", 0.2)
        model = SimpleCNN(num_classes=num_classes, channels=channels, dropout=dropout)
    elif model_type == "resnet18_finetune":
        model = build_resnet18_finetune(
            num_classes=num_classes,
            freeze_backbone=False,
            unfreeze_last_n_layers=4,
        )
    else:
        raise ValueError(f"Unsupported model_type in checkpoint: {model_type}")

    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, image_size


def predict_image_topk(
    model: torch.nn.Module,
    class_names: list[str],
    image: Image.Image,
    image_size: int = 224,
    k: int = 3,
    device: torch.device | str = "cpu",
) -> list[tuple[str, float]]:
    """Predict top-k labels for a PIL image."""
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        top_probs, top_indices = topk_from_logits(logits, k=k)

    predictions: list[tuple[str, float]] = []
    for probability, index in zip(top_probs[0], top_indices[0]):
        label = class_names[int(index)] if class_names else f"class_{int(index)}"
        predictions.append((label, float(probability.item())))
    return predictions
