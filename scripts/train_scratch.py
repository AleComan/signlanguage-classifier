"""Train a simple CNN from scratch."""

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
from src.models.simple_cnn import SimpleCNN
from src.training.engine import evaluate_classification, train_one_epoch
from src.utils.config import load_yaml_config
from src.utils.reproducibility import set_global_seed
from src.utils.wandb_utils import finish_wandb_run, init_wandb_run


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CNN from scratch.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    """Train a lightweight CNN and save checkpoint."""
    args = parse_args()
    config = load_yaml_config(args.config)
    set_global_seed(int(config.get("seed", 42)))

    data_cfg = config["data"]
    train_dataset, val_dataset = load_imagefolder_datasets(
        root_dir=data_cfg["root_dir"],
        train_subdir=data_cfg.get("train_subdir", "train"),
        val_subdir=data_cfg.get("val_subdir", "val"),
        image_size=int(data_cfg.get("image_size", 128)),
    )
    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=int(data_cfg.get("batch_size", 32)),
        num_workers=int(data_cfg.get("num_workers", 0)),
    )

    num_classes = len(train_dataset.classes)
    model_cfg = config["model"]
    model = SimpleCNN(
        num_classes=num_classes,
        channels=model_cfg.get("channels", [16, 32, 64]),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )

    device = torch.device(config.get("device", "cpu"))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    training_cfg = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    epochs = int(training_cfg.get("epochs", 2))

    track_cfg = config.get("tracking", {})
    run = init_wandb_run(
        config=config,
        enabled=bool(track_cfg.get("use_wandb", True)),
        project=track_cfg.get("project", "signlanguage-classifier"),
        run_name=track_cfg.get("run_name"),
        tags=track_cfg.get("tags"),
    )

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_classification(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if run is not None:
            import wandb

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                }
            )

    output_cfg = config["output"]
    artifacts_dir = PROJECT_ROOT / output_cfg.get("artifacts_dir", "artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifacts_dir / output_cfg.get("checkpoint_name", "scratch_cnn.pt")

    torch.save(
        {
            "model_type": "simple_cnn",
            "state_dict": model.state_dict(),
            "class_names": train_dataset.classes,
            "image_size": int(data_cfg.get("image_size", 128)),
            "channels": model_cfg.get("channels", [16, 32, 64]),
            "dropout": float(model_cfg.get("dropout", 0.2)),
        },
        checkpoint_path,
    )
    print(f"Checkpoint guardado en: {checkpoint_path}")
    finish_wandb_run(run)


if __name__ == "__main__":
    main()
