"""Evaluate a torch checkpoint on the validation split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import build_dataloaders, load_imagefolder_datasets
from src.inference.predict import load_torch_model
from src.training.engine import evaluate_classification
from src.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate checkpoint.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path. If omitted, uses config output path.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate a saved model checkpoint."""
    args = parse_args()
    config = load_yaml_config(args.config)

    data_cfg = config["data"]
    _, val_dataset = load_imagefolder_datasets(
        root_dir=data_cfg["root_dir"],
        train_subdir=data_cfg.get("train_subdir", "train"),
        val_subdir=data_cfg.get("val_subdir", "val"),
        image_size=int(data_cfg.get("image_size", 224)),
    )
    _, val_loader = build_dataloaders(
        val_dataset,
        val_dataset,
        batch_size=int(data_cfg.get("batch_size", 32)),
        num_workers=int(data_cfg.get("num_workers", 0)),
    )

    output_cfg = config.get("output", {})
    default_checkpoint = PROJECT_ROOT / output_cfg.get("artifacts_dir", "artifacts") / output_cfg.get(
        "checkpoint_name",
        "scratch_cnn.pt",
    )
    checkpoint_path = args.checkpoint or default_checkpoint

    device = torch.device(config.get("device", "cpu"))
    model, _, _ = load_torch_model(checkpoint_path=checkpoint_path, device=device)
    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc = evaluate_classification(model, val_loader, criterion, device)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
