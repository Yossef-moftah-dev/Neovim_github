"""DVC Stage 1: Data Preparation & Stratified Splitting.

Reads raw parquet dataset, cleans and normalizes Arabic text using prodml.data
and prodml.features, and produces reproducible train/val CSV splits.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prodml.data import clean_records
from prodml.logging import configure_logging

logger = logging.getLogger("prodml.prepare_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and split Arabic review data")
    parser.add_argument(
        "--input",
        type=str,
        default="data/Arabic_Reviews_of_SHEIN/train-00000-of-00001.parquet",
        help="Path to raw parquet dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save train.csv and val.csv",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic split",
    )
    args = parser.parse_args()

    configure_logging("INFO")
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path)
        sys.exit(1)

    logger.info("Loading raw parquet dataset from %s", input_path)
    df = pd.read_parquet(input_path)
    logger.info("Loaded %d raw rows. Running prodml cleaning and normalization...", len(df))

    records = df.to_dict(orient="records")
    cleaned = clean_records(records)
    cleaned_df = pd.DataFrame([r.model_dump() for r in cleaned])
    logger.info("Normalized dataset contains %d valid rows", len(cleaned_df))

    # Stratified train/val split
    labels = cleaned_df["label"].fillna(1).astype(int)
    train_df, val_df = train_test_split(
        cleaned_df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )

    train_out = output_dir / "train.csv"
    val_out = output_dir / "val.csv"
    train_df.to_csv(train_out, index=False, encoding="utf-8")
    val_df.to_csv(val_out, index=False, encoding="utf-8")

    logger.info("Wrote %d train records to %s", len(train_df), train_out)
    logger.info("Wrote %d val records to %s", len(val_df), val_out)
    print(
        f"Data preparation complete: {train_out} ({len(train_df)} rows), {val_out} ({len(val_df)} rows)"
    )


if __name__ == "__main__":
    main()
