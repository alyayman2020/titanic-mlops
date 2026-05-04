"""
batch/motherduck.py — MotherDuck (DuckDB Cloud) connection and queries.

Handles:
- Connecting to MotherDuck using token
- Creating tables if they don't exist
- Reading test data
- Writing predictions back
"""
from __future__ import annotations

import os
from typing import Any

import duckdb
import pandas as pd
from loguru import logger

# ── Config ─────────────────────────────────────────────────────────────────
MOTHERDUCK_TOKEN = os.getenv(
    "MOTHERDUCK_TOKEN",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImFseS5heW1hbi4yMDE4QGdtYWlsLmNvbSIsIm1kUmVnaW9uIjoiYXdzLWV1LWNlbnRyYWwtMSIsInNlc3Npb24iOiJhbHkuYXltYW4uMjAxOC5nbWFpbC5jb20iLCJwYXQiOiJfYjV3bko1VkRFTHR2SDY3czdJMHdIZG9ObDdVQUhjRGtjRzU1Y01mS2FvIiwidXNlcklkIjoiY2ZjMGFkYzYtYzZhMS00MTU5LTlkZWUtMmIxNWIzODNiNzI5IiwiaXNzIjoibWRfcGF0IiwicmVhZE9ubHkiOmZhbHNlLCJ0b2tlblR5cGUiOiJyZWFkX3dyaXRlIiwiaWF0IjoxNzc3OTE2ODk2fQ.omxVMYkTi7XBhoDzDJZH79nn00AhGJzpAg329aNKWp0",
)
DATABASE_NAME = "titanic"
TEST_TABLE    = "test_data"
PRED_TABLE    = "predictions"


def get_connection() -> duckdb.DuckDBPyConnection:
    """Open a MotherDuck connection and ensure database exists."""
    # Connect to MotherDuck root first (no db specified)
    logger.info(f"Connecting to MotherDuck…")
    conn = duckdb.connect(f"md:?motherduck_token={MOTHERDUCK_TOKEN}")

    # Create database if it doesn't exist, then use it
    conn.execute(f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}")
    conn.execute(f"USE {DATABASE_NAME}")
    logger.info(f"Connected to MotherDuck → {DATABASE_NAME} ✓")
    return conn


def create_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create tables if they don't exist."""
    conn.execute(f"USE {DATABASE_NAME}")

    # Test data table
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TEST_TABLE} (
            PassengerId  INTEGER,
            Pclass       INTEGER,
            Name         VARCHAR,
            Sex          VARCHAR,
            Age          DOUBLE,
            SibSp        INTEGER,
            Parch        INTEGER,
            Ticket       VARCHAR,
            Fare         DOUBLE,
            Cabin        VARCHAR,
            Embarked     VARCHAR
        )
    """)

    # Predictions table
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {PRED_TABLE} (
            PassengerId     INTEGER,
            Pclass          INTEGER,
            Sex             VARCHAR,
            Age             DOUBLE,
            Fare            DOUBLE,
            Survived_pred   INTEGER,
            Survival_prob   DOUBLE,
            Confidence      VARCHAR,
            model_name      VARCHAR,
            model_version   VARCHAR,
            predicted_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    logger.info("Tables ready ✓")


def load_test_data(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Load test CSV data into MotherDuck. Truncates existing data first."""
    conn.execute(f"USE {DATABASE_NAME}")
    conn.execute(f"DELETE FROM {TEST_TABLE}")
    conn.execute(f"INSERT INTO {TEST_TABLE} SELECT * FROM df")
    count = conn.execute(f"SELECT COUNT(*) FROM {TEST_TABLE}").fetchone()[0]
    logger.info(f"Loaded {count} rows into {DATABASE_NAME}.{TEST_TABLE} ✓")
    return count


def extract_test_data(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Extract test data from MotherDuck for prediction."""
    conn.execute(f"USE {DATABASE_NAME}")
    df = conn.execute(f"SELECT * FROM {TEST_TABLE}").df()
    logger.info(f"Extracted {len(df)} rows from {DATABASE_NAME}.{TEST_TABLE}")
    return df


def save_predictions(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
) -> int:
    """Save predictions back to MotherDuck. Replaces previous predictions."""
    conn.execute(f"USE {DATABASE_NAME}")
    conn.execute(f"DELETE FROM {PRED_TABLE}")
    conn.execute(f"INSERT INTO {PRED_TABLE} SELECT * FROM df")
    count = conn.execute(f"SELECT COUNT(*) FROM {PRED_TABLE}").fetchone()[0]
    logger.info(f"Saved {count} predictions to {DATABASE_NAME}.{PRED_TABLE} ✓")
    return count


def get_prediction_summary(conn: duckdb.DuckDBPyConnection) -> dict:
    """Get a summary of the latest predictions."""
    conn.execute(f"USE {DATABASE_NAME}")
    result = conn.execute(f"""
        SELECT
            COUNT(*)                                    AS total,
            SUM(Survived_pred)                          AS survived,
            COUNT(*) - SUM(Survived_pred)               AS not_survived,
            ROUND(AVG(Survival_prob) * 100, 2)          AS avg_prob_pct,
            ROUND(SUM(Survived_pred) * 100.0 / COUNT(*), 2) AS survival_rate_pct,
            MAX(predicted_at)                           AS last_run
        FROM {PRED_TABLE}
    """).fetchone()

    return {
        "total":             result[0],
        "survived":          result[1],
        "not_survived":      result[2],
        "avg_probability":   result[3],
        "survival_rate_pct": result[4],
        "last_run":          str(result[5]),
    }