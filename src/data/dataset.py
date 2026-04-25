"""Base dataset loaders for image classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_image_transforms(image_size: int) -> transforms.Compose:
    """Create default image transforms for training/evaluation."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_imagefolder_datasets(
    root_dir: str | Path,
    train_subdir: str = "train",
    val_subdir: str = "val",
    image_size: int = 224,
) -> tuple[datasets.ImageFolder, datasets.ImageFolder]:
    """Load train/val datasets from a torchvision ImageFolder layout."""
    root_path = Path(root_dir)
    train_dir = root_path / train_subdir
    val_dir = root_path / val_subdir

    if not train_dir.exists():
        raise FileNotFoundError(f"Train folder not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation folder not found: {val_dir}")

    transform = build_image_transforms(image_size=image_size)

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    return train_dataset, val_dataset


def build_dataloaders(
    train_dataset: datasets.ImageFolder,
    val_dataset: datasets.ImageFolder,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Build dataloaders for train and validation sets."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader
