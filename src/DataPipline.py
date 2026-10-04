# src/pipeline.py
"""Data processing pipeline."""

import csv
from pathlib import Path


def load_raw_data(filepath: Path) -> list[dict]:
    """Ingest raw records from a CSV source."""
    if not filepath.exists():
        raise FileNotFoundError(f"Source file not found: {filepath}")

    with filepath.open(mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)
