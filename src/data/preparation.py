"""Dataset preparation helpers for ASL Alphabet."""

from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class LabeledImage:
    """Simple container representing an image and its class label."""

    source_path: Path
    label: str


def _normalize_label(label: str, class_renames: dict[str, str]) -> str:
    """Normalize label names with a configurable rename map."""
    return class_renames.get(label, label)


def _is_allowed_image(path: Path, allowed_extensions: set[str]) -> bool:
    """Return True when file extension is accepted."""
    return path.suffix.lower() in allowed_extensions


def _is_valid_image(path: Path) -> bool:
    """Validate an image by opening and verifying it with Pillow."""
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _collect_train_images(
    train_dir: Path,
    allowed_extensions: set[str],
    verify_images: bool,
    class_renames: dict[str, str],
) -> dict[str, list[Path]]:
    """Collect train images grouped by class directory."""
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    grouped: dict[str, list[Path]] = {}
    for class_dir in sorted([path for path in train_dir.iterdir() if path.is_dir()]):
        label = _normalize_label(class_dir.name, class_renames)
        files = [p for p in class_dir.iterdir() if p.is_file() and _is_allowed_image(p, allowed_extensions)]
        if verify_images:
            files = [p for p in files if _is_valid_image(p)]
        grouped[label] = sorted(files)
    return grouped


def _parse_kaggle_test_label(filename: str) -> str:
    """Extract class label from Kaggle ASL test filename (e.g., A_test.jpg)."""
    return filename.split("_test", maxsplit=1)[0]


def _collect_official_test_images(
    test_dir: Path,
    allowed_extensions: set[str],
    verify_images: bool,
    class_renames: dict[str, str],
) -> dict[str, list[Path]]:
    """Collect official Kaggle test images grouped by label."""
    if not test_dir.exists():
        return {}

    grouped: dict[str, list[Path]] = {}
    for file_path in sorted([path for path in test_dir.iterdir() if path.is_file()]):
        if not _is_allowed_image(file_path, allowed_extensions):
            continue
        if verify_images and not _is_valid_image(file_path):
            continue

        raw_label = _parse_kaggle_test_label(file_path.name)
        label = _normalize_label(raw_label, class_renames)
        grouped.setdefault(label, []).append(file_path)
    return grouped


def _split_class_images(
    paths: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split one class list into train/val/test subsets."""
    if not paths:
        return [], [], []

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be exactly 1.0")

    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)

    n_total = len(shuffled)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    n_train = n_total - n_val - n_test

    # Keep all splits non-empty whenever possible.
    if n_total >= 3:
        if n_train <= 0:
            n_train = 1
        if n_val <= 0:
            n_val = 1
        if n_test <= 0:
            n_test = 1
        overflow = (n_train + n_val + n_test) - n_total
        while overflow > 0:
            if n_train > n_val and n_train > 1:
                n_train -= 1
            elif n_val > n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            overflow -= 1

    train_part = shuffled[:n_train]
    val_part = shuffled[n_train : n_train + n_val]
    test_part = shuffled[n_train + n_val :]
    return train_part, val_part, test_part


def _copy_or_link(src: Path, dst: Path, copy_files: bool) -> None:
    """Copy file to destination or create hard link when configured."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(src, dst)
    else:
        if dst.exists():
            dst.unlink()
        dst.hardlink_to(src)


def _write_split(
    records: list[LabeledImage],
    processed_root: Path,
    split_name: str,
    copy_files: bool,
) -> list[dict[str, str]]:
    """Write one split and return metadata rows."""
    rows: list[dict[str, str]] = []
    for item in records:
        target_path = processed_root / split_name / item.label / item.source_path.name
        _copy_or_link(item.source_path, target_path, copy_files=copy_files)
        rows.append(
            {
                "source_path": str(item.source_path),
                "target_path": str(target_path),
                "label": item.label,
                "split": split_name,
            }
        )
    return rows


def _rows_count_by_split(rows: list[dict[str, str]]) -> dict[str, int]:
    """Compute number of rows per split."""
    summary: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        summary[row["split"]] = summary.get(row["split"], 0) + 1
    return summary


def _rows_count_by_label(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    """Compute count by split and by label."""
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        split = row["split"]
        label = row["label"]
        result.setdefault(split, {})
        result[split][label] = result[split].get(label, 0) + 1
    return result


def prepare_asl_alphabet_dataset(config: dict[str, Any]) -> Path:
    """Prepare ASL dataset into ImageFolder-compatible train/val/test splits."""
    paths_cfg = config["paths"]
    split_cfg = config["split"]
    proc_cfg = config["processing"]

    raw_root = Path(paths_cfg["raw_root"])
    processed_root = Path(paths_cfg["processed_root"])
    train_dir = raw_root / paths_cfg.get("train_dir_name", "asl_alphabet_train")
    test_dir = raw_root / paths_cfg.get("test_dir_name", "asl_alphabet_test")

    overwrite = bool(proc_cfg.get("overwrite_processed_dir", False))
    if processed_root.exists() and overwrite:
        shutil.rmtree(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    allowed_extensions = {ext.lower() for ext in proc_cfg.get("allowed_extensions", [".png", ".jpg", ".jpeg"])}
    verify_images = bool(proc_cfg.get("verify_images", True))
    copy_files = bool(proc_cfg.get("copy_files", True))
    class_renames = {str(k): str(v) for k, v in proc_cfg.get("class_renames", {}).items()}
    seed = int(config.get("seed", 42))

    train_grouped = _collect_train_images(
        train_dir=train_dir,
        allowed_extensions=allowed_extensions,
        verify_images=verify_images,
        class_renames=class_renames,
    )
    official_test_grouped = _collect_official_test_images(
        test_dir=test_dir,
        allowed_extensions=allowed_extensions,
        verify_images=verify_images,
        class_renames=class_renames,
    )

    all_rows: list[dict[str, str]] = []
    for label, image_paths in train_grouped.items():
        class_seed = seed + abs(hash(label)) % 1_000_000
        train_split, val_split, test_split = _split_class_images(
            paths=image_paths,
            train_ratio=float(split_cfg["train_ratio"]),
            val_ratio=float(split_cfg["val_ratio"]),
            test_ratio=float(split_cfg["test_ratio"]),
            seed=class_seed,
        )

        test_from_official = (
            official_test_grouped.get(label, [])
            if bool(split_cfg.get("include_official_test_in_test_split", True))
            else []
        )

        train_records = [LabeledImage(source_path=path, label=label) for path in train_split]
        val_records = [LabeledImage(source_path=path, label=label) for path in val_split]
        test_records = [LabeledImage(source_path=path, label=label) for path in [*test_split, *test_from_official]]

        all_rows.extend(_write_split(train_records, processed_root, "train", copy_files=copy_files))
        all_rows.extend(_write_split(val_records, processed_root, "val", copy_files=copy_files))
        all_rows.extend(_write_split(test_records, processed_root, "test", copy_files=copy_files))

    metadata_path = processed_root / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["source_path", "target_path", "label", "split"])
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "dataset_name": config.get("dataset_name", "asl_alphabet"),
        "processed_root": str(processed_root),
        "counts_by_split": _rows_count_by_split(all_rows),
        "counts_by_split_and_label": _rows_count_by_label(all_rows),
    }
    summary_path = processed_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return processed_root
