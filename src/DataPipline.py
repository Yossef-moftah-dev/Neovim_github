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


def clean_and_normalize(records: list[dict]) -> list[dict]:
    """Strip whitespace and filter out rows with missing required fields."""
    cleaned = []
    for row in records:
        normalized = {k.strip(): v.strip() for k, v in row.items() if k}
        if any(normalized.values()):
            cleaned.append(normalized)
    return cleaned


def process_pipeline(input_path: Path) -> list[dict]:
    """Execute end-to-end data processing workflow."""
    raw = load_raw_data(input_path)
    return clean_and_normalize(raw)
