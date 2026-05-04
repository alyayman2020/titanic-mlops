"""
batch/flow.py — Prefect batch serving pipeline.

Flow: Extract from MotherDuck → Transform → Predict → Load back to MotherDuck

Run manually:
    python batch/flow.py

Or deploy to Prefect Cloud:
    prefect deploy batch/flow.py:titanic_batch_prediction
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from loguru import logger
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_table_artifact, create_markdown_artifact

from batch.motherduck import (
    create_tables,
    extract_test_data,
    get_connection,
    get_prediction_summary,
    save_predictions,
)
from batch.predictor import (
    build_predictions_df,
    load_model_from_dagshub,
    predict_batch,
)

# ── Tasks ──────────────────────────────────────────────────────────────────────

@task(name="Connect to MotherDuck", retries=2, retry_delay_seconds=10)
def task_connect_motherduck():
    """Open MotherDuck connection and ensure tables exist."""
    log = get_run_logger()
    log.info("Connecting to MotherDuck…")
    conn = get_connection()
    create_tables(conn)
    log.info("MotherDuck ready ✓")
    return conn


@task(name="Extract Test Data", retries=2, retry_delay_seconds=5)
def task_extract(conn) -> pd.DataFrame:
    """Extract test passenger data from MotherDuck."""
    log = get_run_logger()
    log.info("Extracting test data from MotherDuck…")
    df = extract_test_data(conn)
    log.info(f"Extracted {len(df)} passengers ✓")
    return df


@task(name="Transform Data")
def task_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light transform: ensure correct dtypes and add any missing columns
    the model preprocessor expects.
    """
    log = get_run_logger()
    log.info("Transforming data…")

    df = df.copy()

    # Ensure all expected columns exist
    if "Name" not in df.columns:
        df["Name"] = "Doe, Mr. John"
    if "Ticket" not in df.columns:
        df["Ticket"] = "UNKNOWN"
    if "Cabin" not in df.columns:
        df["Cabin"] = None

    # Fill missing values that the preprocessor handles internally
    # (we keep NaNs intentional — preprocessor learns imputation from training)

    # Ensure correct dtypes
    df["Pclass"]      = df["Pclass"].astype(int)
    df["SibSp"]       = df["SibSp"].fillna(0).astype(int)
    df["Parch"]       = df["Parch"].fillna(0).astype(int)
    df["PassengerId"] = df["PassengerId"].astype(int)

    log.info(f"Transform complete — {len(df)} rows, {len(df.columns)} columns ✓")
    return df


@task(name="Load Model from DagsHub", retries=2, retry_delay_seconds=15)
def task_load_model():
    """Load the Production model from DagsHub MLflow registry."""
    log = get_run_logger()
    log.info("Loading model from DagsHub registry…")
    model, name, version = load_model_from_dagshub()
    log.info(f"Model loaded: {name} / {version} ✓")
    return model, name, version


@task(name="Predict Survival")
def task_predict(
    df: pd.DataFrame,
    model_artifact,
    model_name: str,
    model_version: str,
) -> pd.DataFrame:
    """Run batch prediction on all test passengers."""
    log = get_run_logger()
    log.info(f"Running predictions on {len(df)} passengers…")

    preds, probas = predict_batch(df, model_artifact)
    predictions_df = build_predictions_df(df, preds, probas, model_name, model_version)

    survived = sum(preds)
    log.info(
        f"Predictions complete: {survived}/{len(preds)} survived "
        f"({survived/len(preds)*100:.1f}%) ✓"
    )
    return predictions_df


@task(name="Save Predictions to MotherDuck", retries=2, retry_delay_seconds=10)
def task_save(conn, predictions_df: pd.DataFrame) -> int:
    """Write predictions back to MotherDuck."""
    log = get_run_logger()
    log.info("Saving predictions to MotherDuck…")
    count = save_predictions(conn, predictions_df)
    log.info(f"Saved {count} predictions ✓")
    return count


@task(name="Generate Summary Report")
def task_summary(conn, predictions_df: pd.DataFrame, model_name: str, model_version: str):
    """Create Prefect artifacts with run summary."""
    log = get_run_logger()

    summary = get_prediction_summary(conn)

    # Markdown summary artifact (shows in Prefect Cloud UI)
    markdown = f"""
# 🚢 Titanic Batch Prediction Summary

**Run time:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
**Model:** `{model_name}` / `{model_version}`

## Results

| Metric | Value |
|---|---|
| Total passengers | {summary['total']} |
| Predicted survived | {summary['survived']} |
| Predicted not survived | {summary['not_survived']} |
| Survival rate | {summary['survival_rate_pct']}% |
| Average probability | {summary['avg_probability']}% |

## Confidence Distribution

| Confidence | Count |
|---|---|
| High | {len(predictions_df[predictions_df['Confidence'] == 'High'])} |
| Medium | {len(predictions_df[predictions_df['Confidence'] == 'Medium'])} |
| Low | {len(predictions_df[predictions_df['Confidence'] == 'Low'])} |
"""
    create_markdown_artifact(
        key="batch-prediction-summary",
        markdown=markdown,
        description="Titanic batch prediction run summary",
    )

    log.info(f"Summary: {summary['survived']}/{summary['total']} survived "
             f"({summary['survival_rate_pct']}%)")
    return summary


# ── Main Flow ──────────────────────────────────────────────────────────────────

@flow(
    name="titanic-batch-prediction",
    description=(
        "ETL pipeline: extract test data from MotherDuck → "
        "predict survival using DagsHub Production model → "
        "save results back to MotherDuck"
    ),
    log_prints=True,
)
def titanic_batch_prediction():
    """
    Titanic survival batch prediction pipeline.

    Steps:
    1. Connect to MotherDuck
    2. Extract test passenger data
    3. Transform / validate data
    4. Load Production model from DagsHub
    5. Predict survival for all passengers
    6. Save predictions to MotherDuck
    7. Generate summary report
    """
    logger.info("=" * 60)
    logger.info("Titanic Batch Prediction Pipeline starting…")
    logger.info("=" * 60)

    # 1. Connect
    conn = task_connect_motherduck()

    # 2. Extract
    raw_df = task_extract(conn)

    # 3. Transform
    transformed_df = task_transform(raw_df)

    # 4. Load model
    model_artifact, model_name, model_version = task_load_model()

    # 5. Predict
    predictions_df = task_predict(
        transformed_df, model_artifact, model_name, model_version
    )

    # 6. Save
    task_save(conn, predictions_df)

    # 7. Summary
    summary = task_summary(conn, predictions_df, model_name, model_version)

    logger.info("Pipeline complete ✓")
    logger.info(
        f"Results: {summary['survived']}/{summary['total']} passengers "
        f"predicted to survive ({summary['survival_rate_pct']}%)"
    )
    return summary


if __name__ == "__main__":
    titanic_batch_prediction()
