"""Prepare ASL Alphabet dataset into train/val structure."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from src.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare ASL dataset splits.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset_asl.yaml"),
        help="Path to dataset preparation config YAML.",
    )
    return parser.parse_args()


def _normalize_extensions(extensions: list[str]) -> set[str]:
    return {extension.lower() for extension in extensions}


def _resolve_class_name(raw_class_name: str, class_renames: dict[str, str]) -> str:
    return class_renames.get(raw_class_name, raw_class_name)


def _is_valid_image(image_path: Path, verify_images: bool) -> bool:
    if not verify_images:
        return True
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _prepare_output_dirs(
    processed_root: Path,
    overwrite_processed_dir: bool,
    split_names: list[str],
) -> None:
    if processed_root.exists() and overwrite_processed_dir:
        shutil.rmtree(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)
    for split_name in split_names:
        (processed_root / split_name).mkdir(parents=True, exist_ok=True)


def _copy_or_move_file(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def _split_class_files(
    files: list[Path],
    train_ratio: float,
    rng: random.Random,
) -> tuple[list[Path], list[Path]]:
    shuffled = files[:]
    rng.shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, []

    train_count = int(len(shuffled) * train_ratio)
    train_count = min(max(train_count, 1), len(shuffled) - 1)
    train_files = shuffled[:train_count]
    val_files = shuffled[train_count:]
    return train_files, val_files


def _collect_train_class_files(
    raw_train_root: Path,
    allowed_extensions: set[str],
    verify_images: bool,
    class_renames: dict[str, str],
) -> dict[str, list[Path]]:
    class_to_files: dict[str, list[Path]] = {}
    if not raw_train_root.exists():
        raise FileNotFoundError(f"No existe directorio de entrenamiento raw: {raw_train_root}")

    for class_dir in sorted(path for path in raw_train_root.iterdir() if path.is_dir()):
        resolved_class = _resolve_class_name(class_dir.name, class_renames)
        valid_files: list[Path] = []
        for file_path in sorted(path for path in class_dir.iterdir() if path.is_file()):
            if file_path.suffix.lower() not in allowed_extensions:
                continue
            if not _is_valid_image(file_path, verify_images=verify_images):
                continue
            valid_files.append(file_path)
        if valid_files:
            class_to_files[resolved_class] = valid_files

    if not class_to_files:
        raise ValueError(f"No se encontraron imagenes validas en {raw_train_root}")
    return class_to_files


def _write_metadata_csv(rows: list[dict[str, str]], destination: Path) -> None:
    fieldnames = ["split", "class_name", "src_path", "dst_path"]
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_json(summary: dict[str, Any], destination: Path) -> None:
    with destination.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2, ensure_ascii=False)
        json_file.write("\n")


def prepare_asl_alphabet_dataset_train_val_only(config: dict[str, Any]) -> Path:
    """Build processed dataset using only train and val splits."""
    paths_cfg = config.get("paths", {})
    split_cfg = config.get("split", {})
    processing_cfg = config.get("processing", {})

    raw_root = (PROJECT_ROOT / paths_cfg["raw_root"]).resolve()
    train_dir_name = paths_cfg.get("train_dir_name", "asl_alphabet_train")
    processed_root = (PROJECT_ROOT / paths_cfg["processed_root"]).resolve()

    train_ratio = float(split_cfg.get("train_ratio", 0.8))
    val_ratio = float(split_cfg.get("val_ratio", 0.2))

    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train_ratio y val_ratio deben ser > 0.")
    if abs((train_ratio + val_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio debe ser 1.0 para split train/val.")

    allowed_extensions = _normalize_extensions(
        processing_cfg.get("allowed_extensions", [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
    )
    verify_images = bool(processing_cfg.get("verify_images", True))
    copy_files = bool(processing_cfg.get("copy_files", True))
    overwrite_processed_dir = bool(processing_cfg.get("overwrite_processed_dir", False))
    class_renames = dict(processing_cfg.get("class_renames", {}))
    seed = int(config.get("seed", 42))
    rng = random.Random(seed)

    split_names = ["train", "val"]
    _prepare_output_dirs(
        processed_root=processed_root,
        overwrite_processed_dir=overwrite_processed_dir,
        split_names=split_names,
    )

    raw_train_root = raw_root / train_dir_name
    class_to_files = _collect_train_class_files(
        raw_train_root=raw_train_root,
        allowed_extensions=allowed_extensions,
        verify_images=verify_images,
        class_renames=class_renames,
    )

    metadata_rows: list[dict[str, str]] = []
    summary: dict[str, Any] = {
        "dataset_name": config.get("dataset_name"),
        "seed": seed,
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "split_ratios": {"train_ratio": train_ratio, "val_ratio": val_ratio},
        "num_classes": len(class_to_files),
        "classes": sorted(class_to_files.keys()),
        "counts": {"train": 0, "val": 0},
        "counts_by_class": {"train": {}, "val": {}},
    }

    for class_name, files in sorted(class_to_files.items()):
        train_files, val_files = _split_class_files(files=files, train_ratio=train_ratio, rng=rng)
        split_map = {"train": train_files, "val": val_files}

        for split_name, split_files in split_map.items():
            summary["counts_by_class"][split_name][class_name] = len(split_files)
            summary["counts"][split_name] += len(split_files)
            for src_file in split_files:
                dst_file = processed_root / split_name / class_name / src_file.name
                _copy_or_move_file(src_file, dst_file, copy_files=copy_files)
                metadata_rows.append(
                    {
                        "split": split_name,
                        "class_name": class_name,
                        "src_path": str(src_file),
                        "dst_path": str(dst_file),
                    }
                )

    _write_metadata_csv(metadata_rows, processed_root / "metadata.csv")
    _write_summary_json(summary, processed_root / "summary.json")
    return processed_root


def main() -> None:
    """Run dataset preparation pipeline."""
    args = parse_args()
    config = load_yaml_config(args.config)
    processed_root = prepare_asl_alphabet_dataset_train_val_only(config)
    print(f"Dataset procesado correctamente en: {processed_root}")
    print(f"Resumen: {processed_root / 'summary.json'}")
    print(f"Metadata: {processed_root / 'metadata.csv'}")


if __name__ == "__main__":
    main()
