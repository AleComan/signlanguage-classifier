"""Reusable functions for computing image-folder dataset statistics."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageStats:
    """Per-image statistics returned by :func:`analyze_image`."""

    width: int
    height: int
    mode: str
    fmt: str
    mean_rgb: tuple[float, float, float]
    std_rgb: tuple[float, float, float]
    grayscale_histogram: np.ndarray


def collect_split_files(split_root: Path) -> dict[str, list[Path]]:
    """Return a mapping of class name to image paths for a split folder."""
    per_class: dict[str, list[Path]] = {}
    for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        files = sorted(
            p
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        per_class[class_dir.name] = files
    return per_class


def analyze_image(path: Path) -> ImageStats:
    """Compute size, color and grayscale statistics for a single image."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        mean_rgb = tuple(float(arr[..., c].mean()) for c in range(3))
        std_rgb = tuple(float(arr[..., c].std()) for c in range(3))
        gray = np.asarray(rgb.convert("L"), dtype=np.uint8)
        histogram, _ = np.histogram(gray, bins=256, range=(0, 256))
        return ImageStats(
            width=width,
            height=height,
            mode=img.mode,
            fmt=(img.format or "unknown").lower(),
            mean_rgb=mean_rgb,
            std_rgb=std_rgb,
            grayscale_histogram=histogram,
        )


def basic_distribution(values: list[int | float]) -> dict[str, float]:
    """Return summary statistics for a numeric list."""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def sample_paths(paths: list[Path], limit: int, rng: random.Random) -> list[Path]:
    """Sample `limit` paths deterministically using `rng`."""
    if limit <= 0 or len(paths) <= limit:
        return list(paths)
    return rng.sample(paths, limit)


def run_eda(
    data_root: Path,
    splits: list[str],
    sample_per_split: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute an EDA summary across the requested dataset splits."""
    rng = random.Random(seed)
    split_reports: dict[str, Any] = {}

    global_mode_counter: Counter[str] = Counter()
    global_format_counter: Counter[str] = Counter()
    global_widths: list[int] = []
    global_heights: list[int] = []
    global_ratios: list[float] = []
    global_gray_hist = np.zeros(256, dtype=np.float64)
    global_rgb_means: list[tuple[float, float, float]] = []
    global_rgb_stds: list[tuple[float, float, float]] = []

    total_images = 0
    sampled_images = 0

    for split in splits:
        split_root = data_root / split
        if not split_root.exists():
            continue

        per_class = collect_split_files(split_root)
        class_counts = {label: len(paths) for label, paths in per_class.items()}
        split_total = sum(class_counts.values())
        total_images += split_total

        per_class_quota = sample_per_split // max(len(per_class), 1)
        sampled_split_paths: list[Path] = []
        for paths in per_class.values():
            sampled_split_paths.extend(sample_paths(paths, per_class_quota, rng))

        sampled_images += len(sampled_split_paths)
        split_widths: list[int] = []
        split_heights: list[int] = []
        split_ratios: list[float] = []
        split_gray_hist = np.zeros(256, dtype=np.float64)

        for image_path in sampled_split_paths:
            stats = analyze_image(image_path)
            split_widths.append(stats.width)
            split_heights.append(stats.height)
            split_ratios.append(stats.width / max(stats.height, 1))
            split_gray_hist += stats.grayscale_histogram

            global_mode_counter[stats.mode] += 1
            global_format_counter[stats.fmt] += 1
            global_widths.append(stats.width)
            global_heights.append(stats.height)
            global_ratios.append(stats.width / max(stats.height, 1))
            global_gray_hist += stats.grayscale_histogram
            global_rgb_means.append(stats.mean_rgb)
            global_rgb_stds.append(stats.std_rgb)

        split_reports[split] = {
            "num_classes": len(class_counts),
            "num_images": split_total,
            "sampled_images": len(sampled_split_paths),
            "class_counts": dict(sorted(class_counts.items())),
            "width_distribution": basic_distribution(split_widths) if split_widths else {},
            "height_distribution": basic_distribution(split_heights) if split_heights else {},
            "aspect_ratio_distribution": basic_distribution(split_ratios) if split_ratios else {},
            "grayscale_histogram": split_gray_hist.astype(int).tolist(),
        }

    class_totals: dict[str, int] = defaultdict(int)
    for split_data in split_reports.values():
        for label, count in split_data["class_counts"].items():
            class_totals[label] += count

    counts_array = np.asarray(list(class_totals.values()), dtype=np.float64)
    imbalance = float(counts_array.max() / max(counts_array.min(), 1.0)) if len(counts_array) else 0.0
    cv = float(counts_array.std() / max(counts_array.mean(), 1e-8)) if len(counts_array) else 0.0

    mean_rgb_global = np.asarray(global_rgb_means, dtype=np.float64)
    std_rgb_global = np.asarray(global_rgb_stds, dtype=np.float64)

    return {
        "dataset_root": str(data_root),
        "splits_analyzed": splits,
        "total_images": total_images,
        "sampled_images": sampled_images,
        "overall": {
            "num_classes": len(class_totals),
            "class_totals": dict(sorted(class_totals.items())),
            "class_balance": {
                "min_count": int(counts_array.min()) if len(counts_array) else 0,
                "max_count": int(counts_array.max()) if len(counts_array) else 0,
                "imbalance_ratio_max_over_min": imbalance,
                "coefficient_of_variation": cv,
            },
            "mode_counts": dict(global_mode_counter),
            "format_counts": dict(global_format_counter),
            "width_distribution": basic_distribution(global_widths) if global_widths else {},
            "height_distribution": basic_distribution(global_heights) if global_heights else {},
            "aspect_ratio_distribution": basic_distribution(global_ratios) if global_ratios else {},
            "rgb_channel_mean": {
                "r": float(mean_rgb_global[:, 0].mean()) if len(mean_rgb_global) else 0.0,
                "g": float(mean_rgb_global[:, 1].mean()) if len(mean_rgb_global) else 0.0,
                "b": float(mean_rgb_global[:, 2].mean()) if len(mean_rgb_global) else 0.0,
            },
            "rgb_channel_std": {
                "r": float(std_rgb_global[:, 0].mean()) if len(std_rgb_global) else 0.0,
                "g": float(std_rgb_global[:, 1].mean()) if len(std_rgb_global) else 0.0,
                "b": float(std_rgb_global[:, 2].mean()) if len(std_rgb_global) else 0.0,
            },
            "grayscale_histogram": global_gray_hist.astype(int).tolist(),
        },
        "splits": split_reports,
    }
