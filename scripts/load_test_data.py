"""
scripts/load_test_data.py

One-time script to load data/raw/test.csv into MotherDuck.
Run this ONCE before running the batch flow.

Usage:
    python scripts/load_test_data.py
    python scripts/load_test_data.py --path data/raw/test.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from batch.motherduck import create_tables, get_connection, load_test_data


def main(csv_path: str = "data/raw/test.csv"):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Test CSV not found: {path}")

    logger.info(f"Loading {path} into MotherDuck…")

    # Read CSV
    df = pd.read_csv(path)
    logger.info(f"Read {len(df)} rows from {path}")

    # Connect and create tables
    conn = get_connection()
    create_tables(conn)

    # Load data
    count = load_test_data(conn, df)

    logger.info(f"✓ Successfully loaded {count} rows into MotherDuck")
    logger.info("You can now run the batch flow: python batch/flow.py")

    # Quick verification
    sample = conn.execute("SELECT * FROM titanic.test_data LIMIT 3").df()
    logger.info(f"\nSample data:\n{sample[['PassengerId','Pclass','Sex','Age','Fare']].to_string()}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/raw/test.csv")
    args = parser.parse_args()
    main(args.path)
