"""Inference utilities for Streamlit model selection and prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from src.evaluation.metrics import topk_from_logits

SUPPORTED_MODEL_EXTENSIONS = (".joblib", ".pt", ".pth", ".ckpt")
FEATURE_ARTIFACT_TOKENS = ("feature", "features", "embedding", "embeddings", "cache")


def _looks_like_feature_artifact(path: Path) -> bool:
    """Return True when filename likely corresponds to cached features, not a model."""
    stem = path.stem.lower()
    return any(token in stem for token in FEATURE_ARTIFACT_TOKENS)


@dataclass
class LoadedModel:
    """Unified container for loaded model inference artifacts."""

    kind: str
    path: Path
    class_names: list[str]
    image_size: int
    metadata: dict[str, Any]
    torch_model: torch.nn.Module | None = None
    feature_extractor: torch.nn.Module | None = None
    sklearn_model: Any | None = None
    scaler: Any | None = None


def discover_available_models(models_root: str | Path) -> list[Path]:
    """List supported model files under the provided root."""
    root = Path(models_root)
    if not root.exists():
        return []
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_MODEL_EXTENSIONS
            and not _looks_like_feature_artifact(path)
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


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
        try:
            from src.models.simple_cnn import SimpleCNN
        except ImportError as error:
            raise ImportError("Cannot import src.models.simple_cnn for this checkpoint.") from error
        channels = payload.get("channels", [16, 32, 64])
        dropout = payload.get("dropout", 0.2)
        model = SimpleCNN(num_classes=num_classes, channels=channels, dropout=dropout)
    elif model_type == "resnet18_finetune":
        try:
            from src.models.transfer import build_resnet18_finetune
        except ImportError as error:
            raise ImportError("Cannot import src.models.transfer for this checkpoint.") from error
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


def _get_preprocess(image_size: int) -> transforms.Compose:
    """Create a standard torchvision preprocessing pipeline."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _build_resnet18_feature_extractor(device: torch.device | str) -> torch.nn.Module:
    """Create ResNet18 feature extractor (frozen, fc removed)."""
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = torch.nn.Identity()
    backbone.eval().to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return backbone


def load_model_for_inference(
    model_path: str | Path,
    device: torch.device | str = "cpu",
) -> LoadedModel:
    """Load either sklearn baseline (.joblib) or torch checkpoint."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".joblib":
        payload = joblib.load(path)
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError(f"Unsupported joblib payload format: {path}")
        class_names = list(payload.get("class_names", []))
        return LoadedModel(
            kind="sklearn_baseline",
            path=path,
            class_names=class_names,
            image_size=224,
            metadata={
                "file_type": suffix,
                "model_family": "sklearn",
                "best_name": payload.get("best_name"),
                "num_classes": len(class_names),
            },
            sklearn_model=payload.get("model"),
            scaler=payload.get("scaler"),
            feature_extractor=_build_resnet18_feature_extractor(device),
        )

    torch_model, class_names, image_size = load_torch_model(checkpoint_path=path, device=device)
    return LoadedModel(
        kind="torch_checkpoint",
        path=path,
        class_names=class_names,
        image_size=image_size,
        metadata={
            "file_type": suffix,
            "model_family": "torch",
            "num_classes": len(class_names),
        },
        torch_model=torch_model,
    )


def predict_image_topk_torch(
    model: torch.nn.Module,
    class_names: list[str],
    image: Image.Image,
    image_size: int = 224,
    k: int = 3,
    device: torch.device | str = "cpu",
) -> list[tuple[str, float]]:
    """Predict top-k labels for a PIL image."""
    preprocess = _get_preprocess(image_size=image_size)
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        top_probs, top_indices = topk_from_logits(logits, k=k)

    predictions: list[tuple[str, float]] = []
    for probability, index in zip(top_probs[0], top_indices[0]):
        label = class_names[int(index)] if class_names else f"class_{int(index)}"
        predictions.append((label, float(probability.item())))
    return predictions


def predict_image_topk_baseline(
    *,
    sklearn_model: Any,
    scaler: Any,
    feature_extractor: torch.nn.Module,
    class_names: list[str],
    image: Image.Image,
    image_size: int = 224,
    k: int = 3,
    device: torch.device | str = "cpu",
) -> list[tuple[str, float]]:
    """Predict top-k using baseline sklearn model with ResNet18 features."""
    preprocess = _get_preprocess(image_size=image_size)
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        features = feature_extractor(tensor).cpu().numpy()

    if scaler is not None:
        features = scaler.transform(features)

    probabilities: np.ndarray | None = None
    if hasattr(sklearn_model, "predict_proba"):
        probabilities = sklearn_model.predict_proba(features)[0]
    else:
        scores = sklearn_model.decision_function(features)[0]
        scores = np.asarray(scores, dtype=np.float64)
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        probabilities = exp_scores / np.sum(exp_scores)

    top_indices = np.argsort(probabilities)[::-1][: min(k, len(probabilities))]
    return [
        (
            class_names[int(index)] if class_names else f"class_{int(index)}",
            float(probabilities[int(index)]),
        )
        for index in top_indices
    ]


def predict_image_topk(
    loaded_model: LoadedModel,
    image: Image.Image,
    k: int = 5,
    device: torch.device | str = "cpu",
) -> list[tuple[str, float]]:
    """Dispatch top-k prediction depending on loaded model family."""
    if loaded_model.kind == "torch_checkpoint":
        if loaded_model.torch_model is None:
            raise ValueError("Loaded torch checkpoint without torch_model.")
        return predict_image_topk_torch(
            model=loaded_model.torch_model,
            class_names=loaded_model.class_names,
            image=image,
            image_size=loaded_model.image_size,
            k=k,
            device=device,
        )

    if loaded_model.kind == "sklearn_baseline":
        if loaded_model.sklearn_model is None or loaded_model.feature_extractor is None:
            raise ValueError("Loaded sklearn baseline without required artifacts.")
        return predict_image_topk_baseline(
            sklearn_model=loaded_model.sklearn_model,
            scaler=loaded_model.scaler,
            feature_extractor=loaded_model.feature_extractor,
            class_names=loaded_model.class_names,
            image=image,
            image_size=loaded_model.image_size,
            k=k,
            device=device,
        )

    raise ValueError(f"Unsupported loaded model kind: {loaded_model.kind}")
