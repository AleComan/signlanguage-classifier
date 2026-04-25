"""Train a fine-tuned pretrained model."""

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
from src.models.transfer import build_resnet18_finetune
from src.training.engine import evaluate_classification, train_one_epoch
from src.utils.config import load_yaml_config
from src.utils.reproducibility import set_global_seed
from src.utils.wandb_utils import finish_wandb_run, init_wandb_run


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a fine-tuned model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    """Fine-tune a pretrained ResNet18 checkpoint."""
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

    model_cfg = config["model"]
    model = build_resnet18_finetune(
        num_classes=len(train_dataset.classes),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
        unfreeze_last_n_layers=int(model_cfg.get("unfreeze_last_n_layers", 1)),
    )

    device = torch.device(config.get("device", "cpu"))
    model.to(device)

    training_cfg = config["training"]
    head_lr = float(training_cfg.get("learning_rate_head", 1e-3))
    backbone_lr = float(training_cfg.get("learning_rate_backbone", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    epochs = int(training_cfg.get("epochs", 2))

    head_params = []
    backbone_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    optimizer = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

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
    checkpoint_path = artifacts_dir / output_cfg.get("checkpoint_name", "finetune_resnet18.pt")

    torch.save(
        {
            "model_type": "resnet18_finetune",
            "state_dict": model.state_dict(),
            "class_names": train_dataset.classes,
            "image_size": int(data_cfg.get("image_size", 224)),
        },
        checkpoint_path,
    )
    print(f"Checkpoint guardado en: {checkpoint_path}")
    finish_wandb_run(run)


if __name__ == "__main__":
    main()
