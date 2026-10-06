"""Tests for data loading, schema validation, and star mapping."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from prodml.data import clean_records, load_raw_csv, stars_to_sentiment


def test_stars_to_sentiment_mapping() -> None:
    # 1-2 stars -> Negative (0)
    assert stars_to_sentiment(1) == 0
    assert stars_to_sentiment("1") == 0
    assert stars_to_sentiment(2.0) == 0

    # 3 stars -> Neutral (1)
    assert stars_to_sentiment(3) == 1
    assert stars_to_sentiment("3") == 1

    # 4-5 stars -> Positive (2)
    assert stars_to_sentiment(4) == 2
    assert stars_to_sentiment(5) == 2
    assert stars_to_sentiment("5") == 2

    # Invalid input fallback to neutral
    assert stars_to_sentiment("invalid") == 1


def test_load_raw_csv(tmp_path: Path) -> None:
    test_csv = tmp_path / "test.csv"
    with test_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["raw_text", "label"])
        writer.writeheader()
        writer.writerow({"raw_text": "منتج جميل", "label": "5"})
        writer.writerow({"raw_text": "منتج سيء", "label": "1"})

    records = load_raw_csv(test_csv)
    assert len(records) == 2
    assert records[0]["raw_text"] == "منتج جميل"


def test_load_raw_csv_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_csv(Path("/nonexistent/path/file.csv"))


def test_clean_records() -> None:
    raw_data = [
        {"raw_text": "  منتج راااائع جداً  ", "label": "5"},
        {"raw_text": "   ", "label": "1"},  # empty should be dropped
        {"raw_text": "سيء للغاية", "label": "1"},
    ]
    cleaned = clean_records(raw_data)
    assert len(cleaned) == 2
    assert cleaned[0].text == "منتج راائع جدا"
    assert cleaned[0].label == 2
    assert cleaned[1].label == 0
