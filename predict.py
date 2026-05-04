"""
predict.py — Load the Production model from MLflow registry and run inference.

Usage:
    python predict.py                          # runs on data/raw/test.csv
    python predict.py --input path/to/file.csv
    python predict.py --local models/titanic/xgboost_tuned.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.utils.logger import get_logger

console = Console()
log = get_logger("predict")

TRACKING_URI = "mlruns"
MODEL_NAME   = "titanic-best-model"


def load_from_registry(model_name: str = MODEL_NAME) -> object:
    """Load the Production model from MLflow registry."""
    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = f"models:/{model_name}/Production"
    log.info(f"Loading model from registry: {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)


def load_from_local(pkl_path: str) -> object:
    """Load a model artifact from a local .pkl file."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def predict_sklearn(model, X: pd.DataFrame) -> np.ndarray:
    """Handle sklearn Pipeline artifact."""
    return model.predict(X), model.predict_proba(X)[:, 1]


def predict_catboost(artifact: dict, X: pd.DataFrame):
    """Handle CatBoost artifact dict {model, preprocessor, cat_indices}."""
    prep = artifact["preprocessor"]
    cat_idx = artifact["cat_indices"]
    model = artifact["model"]
    X_clean = prep.transform(X)
    return model.predict(X_clean), model.predict_proba(X_clean)[:, 1]


def predict_tabnet(artifact: dict, X: pd.DataFrame):
    """Handle TabNet artifact dict {model, preprocessor}."""
    prep = artifact["preprocessor"]
    model = artifact["model"]
    X_np = prep.transform(X).astype(np.float32)
    return model.predict(X_np), model.predict_proba(X_np)[:, 1]


def run_inference(input_csv: str, model_source: str = "registry", local_path: str = ""):
    df = pd.read_csv(input_csv)
    passenger_ids = df.get("PassengerId", pd.Series(range(len(df))))

    # Drop target if present (inference on test set)
    if "Survived" in df.columns:
        df = df.drop(columns=["Survived"])

    # ── Load model ────────────────────────────────────────────────
    if model_source == "local":
        artifact = load_from_local(local_path)
    else:
        artifact = load_from_registry()

    # ── Infer type and predict ────────────────────────────────────
    if isinstance(artifact, dict) and "cat_indices" in artifact:
        y_pred, y_proba = predict_catboost(artifact, df)
    elif isinstance(artifact, dict) and "preprocessor" in artifact:
        y_pred, y_proba = predict_tabnet(artifact, df)
    elif hasattr(artifact, "predict"):
        # Could be sklearn pipeline or mlflow pyfunc
        try:
            y_pred = artifact.predict(df)
            y_proba = artifact.predict_proba(df)[:, 1] if hasattr(artifact, "predict_proba") else None
        except Exception:
            y_pred = np.array(artifact.predict(df))
            y_proba = None
    else:
        raise ValueError("Cannot infer model type from artifact.")

    # ── Display results ───────────────────────────────────────────
    results = pd.DataFrame({
        "PassengerId": passenger_ids.values,
        "Survived_pred": y_pred,
        "Survived_proba": y_proba if y_proba is not None else ["N/A"] * len(y_pred),
    })

    table = Table(title="Predictions", show_lines=True)
    table.add_column("PassengerId", style="cyan")
    table.add_column("Predicted Survived", style="green")
    table.add_column("Probability", style="yellow")
    for _, row in results.head(20).iterrows():
        table.add_row(
            str(int(row["PassengerId"])),
            str(int(row["Survived_pred"])),
            f"{row['Survived_proba']:.4f}" if isinstance(row["Survived_proba"], float) else str(row["Survived_proba"]),
        )
    console.print(table)
    console.print(f"Total predictions: {len(results)}")

    # Save output
    out_path = Path("reports/titanic/predictions.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    log.info(f"Predictions saved to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/test.csv")
    parser.add_argument("--source", choices=["registry", "local"], default="registry")
    parser.add_argument("--local", default="", help="Path to local .pkl file")
    args = parser.parse_args()

    run_inference(args.input, model_source=args.source, local_path=args.local)
