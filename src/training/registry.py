"""
MLflow model registry helper.

Picks the best model across all runs by test ROC-AUC and registers it.
Uses mlflow.pyfunc to wrap the pkl artifact for proper registration.
"""
from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from src.utils.logger import get_logger

log = get_logger("registry")


class PklModelWrapper(mlflow.pyfunc.PythonModel):
    """Wraps a raw pkl artifact as an MLflow pyfunc model."""

    def load_context(self, context):
        pkl_path = context.artifacts["model_pkl"]
        with open(pkl_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        import numpy as np
        obj = self.model
        # Handle dict artifacts (CatBoost, TabNet)
        if isinstance(obj, dict):
            model = obj["model"]
            prep  = obj.get("preprocessor")
            cat_i = obj.get("cat_indices")
            if prep is not None:
                X = prep.transform(model_input) if hasattr(prep, "transform") else prep.transform(model_input)
            else:
                X = model_input.values
            if cat_i is not None:
                return model.predict(X)
            return model.predict(np.array(X, dtype=np.float32) if hasattr(X, "values") else X)
        # Standard sklearn pipeline
        return obj.predict(model_input)


def register_best_model(
    experiment_id: str,
    model_registry_name: str,
    metric: str = "test_roc_auc",
) -> str:
    """
    Find the best run by metric, wrap the pkl artifact as an MLflow
    pyfunc model, register it and promote to Production.
    """
    client = MlflowClient()

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="",
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )

    if not runs:
        log.warning("No completed runs found — skipping registration.")
        return ""

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_score  = best_run.data.metrics.get(metric, "N/A")
    best_name   = best_run.data.tags.get("mlflow.runName", best_run_id)
    model_name  = best_run.data.tags.get("model_name", "unknown")

    log.info(f"Best run : {best_name}")
    log.info(f"Model    : {model_name}")
    log.info(f"{metric} : {best_score}")

    # ── Find the MLflow model artifact (logged with proper flavor) ─
    # Models are logged as "model_{name}" artifacts in each run
    all_artifacts = client.list_artifacts(best_run_id)
    model_artifact = None
    for art in all_artifacts:
        if art.path.startswith("model_") and not art.path.endswith(".pkl"):
            model_artifact = art
            break

    if model_artifact:
        # Register the properly-logged MLflow model directly
        model_uri = f"runs:/{best_run_id}/{model_artifact.path}"
        log.info(f"Registering MLflow model: {model_artifact.path}")
    else:
        # Fallback: look for pkl in models/ folder
        artifacts = client.list_artifacts(best_run_id, path="models")
        if not artifacts:
            log.warning("No model artifacts found — skipping registration.")
            return ""
        pkl_artifact = next(
            (a for a in artifacts if a.path.endswith(".pkl")), artifacts[0]
        )
        local_pkl = client.download_artifacts(best_run_id, pkl_artifact.path)
        log.info(f"Falling back to pkl: {local_pkl}")
        with mlflow.start_run(run_id=best_run_id):
            mlflow.pyfunc.log_model(
                artifact_path="registered_model",
                python_model=PklModelWrapper(),
                artifacts={"model_pkl": local_pkl},
                registered_model_name=model_registry_name,
            )
        log.info(f"Registered model '{model_registry_name}' ✓")
        versions = client.search_model_versions(f"name='{model_registry_name}'")
        if versions:
            latest = sorted(versions, key=lambda v: int(v.version))[-1]
            client.transition_model_version_stage(
                name=model_registry_name,
                version=latest.version,
                stage="Production",
                archive_existing_versions=True,
            )
            log.info(f"Version {latest.version} promoted to Production ✓")
            return latest.version
        return ""

    mv = mlflow.register_model(model_uri=model_uri, name=model_registry_name)

    log.info(f"Registered model '{model_registry_name}' ✓")

    # ── Promote latest version to Production ──────────────────────
    versions = client.search_model_versions(f"name='{model_registry_name}'")
    if versions:
        latest = sorted(versions, key=lambda v: int(v.version))[-1]
        client.transition_model_version_stage(
            name=model_registry_name,
            version=latest.version,
            stage="Production",
            archive_existing_versions=True,
        )
        log.info(f"Version {latest.version} promoted to Production ✓")
        return latest.version

    return ""
