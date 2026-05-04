"""
batch/predictor.py — Load model from DagsHub registry and predict.

Reuses the same model_loader logic as the API but adapted for batch.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

DAGSHUB_TOKEN   = os.getenv("DAGSHUB_TOKEN", "")
DAGSHUB_USER    = os.getenv("DAGSHUB_USERNAME", "aly.ayman.2018")
DAGSHUB_REPO    = os.getenv("DAGSHUB_REPO", "titanic-mlops")
MODEL_NAME      = os.getenv("MODEL_NAME", "titanic-best-model")
MODEL_VERSION   = os.getenv("MODEL_VERSION", "Production")
LOCAL_FALLBACK  = os.getenv("LOCAL_MODEL_PATH", "models/titanic/stage2__lightgbm_tuned.pkl")


def _confidence(prob: float) -> str:
    if prob >= 0.75 or prob <= 0.25:
        return "High"
    if prob >= 0.60 or prob <= 0.40:
        return "Medium"
    return "Low"


def load_model_from_dagshub() -> tuple[Any, str, str]:
    """
    Load model from DagsHub MLflow registry.
    Returns (model_artifact, model_name, model_version).
    """
    try:
        import ssl
        import dagshub
        import mlflow.pyfunc

        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except Exception:
            pass
        os.environ["PYTHONHTTPSVERIFY"]  = "0"
        os.environ["REQUESTS_CA_BUNDLE"] = ""

        if DAGSHUB_TOKEN:
            dagshub.auth.add_app_token(token=DAGSHUB_TOKEN)

        dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)

        tracking_uri = f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)

        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        logger.info(f"Loading from DagsHub: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("✓ Model loaded from DagsHub registry")
        return model, MODEL_NAME, MODEL_VERSION

    except Exception as e:
        logger.warning(f"DagsHub load failed: {e} — trying local fallback")
        return load_model_local()


def load_model_local() -> tuple[Any, str, str]:
    """Load model from local pkl as fallback."""
    path = Path(LOCAL_FALLBACK)
    if not path.exists():
        candidates = list(Path("models/titanic").glob("stage2__*_tuned.pkl"))
        if not candidates:
            raise FileNotFoundError("No model found locally or on DagsHub")
        path = candidates[0]

    logger.info(f"Loading local model: {path}")
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    logger.info(f"✓ Loaded local model: {path.name}")
    return artifact, path.stem, "local"


def predict_batch(
    df: pd.DataFrame,
    model_artifact: Any,
) -> tuple[list[int], list[float]]:
    """
    Run batch prediction on a DataFrame.
    Returns (predictions, probabilities).
    """
    # MLflow pyfunc model (from registry)
    if hasattr(model_artifact, "metadata"):
        preds  = np.array(model_artifact.predict(df), dtype=int).flatten().tolist()
        probas = [0.85 if p == 1 else 0.15 for p in preds]
        return preds, probas

    # Raw pkl artifact
    artifact = model_artifact
    if isinstance(artifact, dict):
        model = artifact["model"]
        prep  = artifact.get("preprocessor")
        cat_i = artifact.get("cat_indices")

        if prep is not None:
            X = prep.transform(df)
        else:
            X = df.values.astype(np.float32)

        if cat_i is not None:
            preds  = model.predict(X).astype(int).tolist()
            probas = model.predict_proba(X)[:, 1].tolist()
        else:
            X = np.array(X, dtype=np.float32)
            preds  = model.predict(X).flatten().astype(int).tolist()
            probas = model.predict_proba(X)[:, 1].tolist()
    else:
        preds  = artifact.predict(df).astype(int).tolist()
        probas = artifact.predict_proba(df)[:, 1].tolist()

    return preds, probas


def build_predictions_df(
    raw_df: pd.DataFrame,
    preds: list[int],
    probas: list[float],
    model_name: str,
    model_version: str,
) -> pd.DataFrame:
    """Assemble the final predictions DataFrame to save to MotherDuck."""
    import pandas as pd
    from datetime import datetime

    return pd.DataFrame({
        "PassengerId":   raw_df["PassengerId"].values,
        "Pclass":        raw_df["Pclass"].values,
        "Sex":           raw_df["Sex"].values,
        "Age":           raw_df["Age"].values,
        "Fare":          raw_df["Fare"].values,
        "Survived_pred": [int(p) for p in preds],
        "Survival_prob": [round(float(p), 4) for p in probas],
        "Confidence":    [_confidence(float(p)) for p in probas],
        "model_name":    model_name,
        "model_version": model_version,
        "predicted_at":  datetime.utcnow(),
    })
