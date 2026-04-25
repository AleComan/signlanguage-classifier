"""Train a baseline classical ML classifier with optional deep features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import build_dataloaders, load_imagefolder_datasets
from src.utils.config import load_yaml_config
from src.utils.reproducibility import set_global_seed
from src.utils.wandb_utils import finish_wandb_run, init_wandb_run


def parse_args() -> argparse.Namespace:
    """Parse script arguments."""
    parser = argparse.ArgumentParser(description="Train baseline ML classifier.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    return parser.parse_args()


def _extract_deep_features(
    loader: DataLoader[Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract deep features using a pretrained ResNet18 encoder."""
    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()
    backbone.to(device)
    backbone.eval()

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            feats = backbone(images).cpu().numpy()
            features.append(feats)
            labels.append(targets.numpy())

    return np.concatenate(features), np.concatenate(labels)


def _extract_raw_features(loader: DataLoader[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Extract flattened pixel features."""
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, targets in loader:
        batch = images.view(images.size(0), -1).numpy()
        features.append(batch)
        labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)


def _build_classifier(classifier_cfg: dict[str, Any]) -> Any:
    """Instantiate a scikit-learn classifier from config."""
    cls_type = classifier_cfg["type"]
    params = classifier_cfg.get("params", {})

    if cls_type == "logistic_regression":
        return LogisticRegression(**params)
    if cls_type == "svm":
        return SVC(**params)
    if cls_type == "random_forest":
        return RandomForestClassifier(**params)
    raise ValueError(f"Unsupported classifier type: {cls_type}")


def main() -> None:
    """Entrypoint for baseline training."""
    args = parse_args()
    config = load_yaml_config(args.config)
    set_global_seed(int(config.get("seed", 42)))

    data_cfg = config["data"]
    train_dataset, val_dataset = load_imagefolder_datasets(
        root_dir=data_cfg["root_dir"],
        train_subdir=data_cfg.get("train_subdir", "train"),
        val_subdir=data_cfg.get("val_subdir", "val"),
        image_size=int(data_cfg.get("image_size", 224)),
    )
    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=int(data_cfg.get("batch_size", 32)),
        num_workers=int(data_cfg.get("num_workers", 0)),
    )

    track_cfg = config.get("tracking", {})
    run = init_wandb_run(
        config=config,
        enabled=bool(track_cfg.get("use_wandb", True)),
        project=track_cfg.get("project", "signlanguage-classifier"),
        run_name=track_cfg.get("run_name"),
        tags=track_cfg.get("tags"),
    )

    use_deep_features = bool(config["model"].get("use_deep_features", True))
    device = torch.device(config.get("device", "cpu"))
    if use_deep_features:
        try:
            x_train, y_train = _extract_deep_features(train_loader, device=device)
            x_val, y_val = _extract_deep_features(val_loader, device=device)
            feature_mode = "deep_resnet18"
        except Exception as error:
            print(f"No fue posible extraer features profundas ({error}). Fallback a features crudas.")
            x_train, y_train = _extract_raw_features(train_loader)
            x_val, y_val = _extract_raw_features(val_loader)
            feature_mode = "raw_flattened"
    else:
        x_train, y_train = _extract_raw_features(train_loader)
        x_val, y_val = _extract_raw_features(val_loader)
        feature_mode = "raw_flattened"

    classifier = _build_classifier(config["classifier"])
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_val)
    val_accuracy = accuracy_score(y_val, predictions)

    print(f"Validation accuracy ({feature_mode}): {val_accuracy:.4f}")
    if run is not None:
        import wandb

        wandb.log({"val_accuracy": float(val_accuracy)})

    output_cfg = config["output"]
    artifacts_dir = PROJECT_ROOT / output_cfg.get("artifacts_dir", "artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / output_cfg.get("model_name", "baseline_model.joblib")
    joblib.dump(
        {
            "classifier": classifier,
            "class_names": train_dataset.classes,
            "feature_mode": feature_mode,
            "image_size": int(data_cfg.get("image_size", 224)),
        },
        model_path,
    )
    print(f"Modelo baseline guardado en: {model_path}")
    finish_wandb_run(run)


if __name__ == "__main__":
    main()
