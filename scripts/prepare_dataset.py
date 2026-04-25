"""Prepare ASL Alphabet dataset into train/val/test structure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preparation import prepare_asl_alphabet_dataset
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


def main() -> None:
    """Run dataset preparation pipeline."""
    args = parse_args()
    config = load_yaml_config(args.config)
    processed_root = prepare_asl_alphabet_dataset(config)
    print(f"Dataset procesado correctamente en: {processed_root}")
    print(f"Resumen: {processed_root / 'summary.json'}")
    print(f"Metadata: {processed_root / 'metadata.csv'}")


if __name__ == "__main__":
    main()
