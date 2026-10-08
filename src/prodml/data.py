"""Data ingestion, validation, and dataset preparation module."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from prodml.features import normalize_arabic

logger = logging.getLogger(__name__)


class ReviewRecord(BaseModel):
    """Schema for a single customer review record."""

    text: str = Field(..., min_length=1)
    label: int | None = Field(default=None, ge=0, le=2)
    raw_text: str | None = None


def stars_to_sentiment(rating: float | str) -> int:
    """Map 1-5 star ratings to 3 sentiment categories.

    - 1-2 stars -> 0 (Negative)
    - 3 stars   -> 1 (Neutral)
    - 4-5 stars -> 2 (Positive)
    """
    try:
        val = int(float(rating))
    except (ValueError, TypeError):
        return 1

    if val <= 2:
        return 0
    elif val == 3:
        return 1
    return 2


def load_raw_csv(filepath: Path | str) -> list[dict[str, Any]]:
    """Ingest raw records from a CSV source file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")

    with path.open(mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def clean_records(records: list[dict[str, Any]]) -> list[ReviewRecord]:
    """Clean, normalize, and validate review records.

    Filters out empty rows and normalizes Arabic text.
    """
    cleaned: list[ReviewRecord] = []
    dropped_count = 0

    for row in records:
        text_val = row.get("raw_text") or row.get("text") or row.get("review") or ""
        normalized = normalize_arabic(str(text_val))
        if not normalized:
            dropped_count += 1
            continue

        raw_label = row.get("label") or row.get("sentiment") or row.get("rating")
        label = stars_to_sentiment(raw_label) if raw_label is not None else None

        cleaned.append(
            ReviewRecord(
                text=normalized,
                label=label,
                raw_text=str(text_val),
            )
        )

    logger.info("Cleaned %d records (dropped %d invalid rows)", len(cleaned), dropped_count)
    return cleaned
