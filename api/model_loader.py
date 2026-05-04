"""
api/model_loader.py

Load the production model from DagsHub MLflow registry.
Falls back to local pkl if DagsHub is unreachable.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

from loguru import logger

# ── Constants ─────────────────────────────────────────────────────────────────
DAGSHUB_USERNAME  = os.getenv("DAGSHUB_USERNAME",  "aly.ayman.2018")
DAGSHUB_TOKEN     = os.getenv("DAGSHUB_TOKEN",     "")
DAGSHUB_REPO      = os.getenv("DAGSHUB_REPO",      "titanic-mlops")
MODEL_NAME        = os.getenv("MODEL_NAME",         "titanic-best-model")
MODEL_VERSION     = os.getenv("MODEL_VERSION",      "Production")

# Local fallback — best Stage 2 model by test AUC
LOCAL_FALLBACK    = os.getenv(
    "LOCAL_MODEL_PATH",
    "models/titanic/stage2__lightgbm_tuned.pkl",
)

# ── Model registry ────────────────────────────────────────────────────────────

class ModelRegistry:
    """Holds the loaded model and metadata."""

    def __init__(self):
        self.model        = None
        self.model_name   = MODEL_NAME
        self.model_source = "not_loaded"
        self.model_version = MODEL_VERSION
        self.model_type   = "unknown"

    # ── Primary: DagsHub MLflow registry ──────────────────────────────────────

    def _load_from_dagshub(self) -> bool:
        """Try to load the Production model from DagsHub MLflow registry."""
        try:
            import dagshub
            import mlflow
            import mlflow.pyfunc

            logger.info("Attempting to load model from DagsHub MLflow registry…")

            # Disable SSL verification issues on some systems
            import ssl
            try:
                ssl._create_default_https_context = ssl._create_unverified_context
            except Exception:
                pass
            os.environ["PYTHONHTTPSVERIFY"]  = "0"
            os.environ["REQUESTS_CA_BUNDLE"] = ""

            if DAGSHUB_TOKEN:
                dagshub.auth.add_app_token(token=DAGSHUB_TOKEN)

            dagshub.init(
                repo_owner=DAGSHUB_USERNAME,
                repo_name=DAGSHUB_REPO,
                mlflow=True,
            )

            tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
            mlflow.set_tracking_uri(tracking_uri)

            model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
            logger.info(f"Loading: {model_uri}")

            self.model        = mlflow.pyfunc.load_model(model_uri)
            self.model_source = "dagshub_registry"
            self.model_version = MODEL_VERSION
            self.model_type   = type(self.model).__name__

            logger.info(f"✓ Loaded from DagsHub registry ({MODEL_VERSION})")
            return True

        except Exception as e:
            logger.warning(f"DagsHub load failed: {e}")
            return False

    # ── Fallback: local pkl ────────────────────────────────────────────────────

    def _load_from_local(self) -> bool:
        """Load from local pkl file as fallback."""
        try:
            path = Path(LOCAL_FALLBACK)
            if not path.exists():
                # Try to find any stage2 tuned model
                models_dir = Path("models/titanic")
                candidates = list(models_dir.glob("stage2__*_tuned.pkl"))
                if not candidates:
                    candidates = list(models_dir.glob("stage1__*.pkl"))
                if not candidates:
                    raise FileNotFoundError(f"No model files found in {models_dir}")
                path = candidates[0]

            logger.info(f"Loading local model: {path}")
            with open(path, "rb") as f:
                artifact = pickle.load(f)

            self.model        = artifact
            self.model_source = "local_fallback"
            self.model_version = path.stem
            self.model_type   = type(artifact).__name__

            logger.info(f"✓ Loaded local model: {path.name}")
            return True

        except Exception as e:
            logger.error(f"Local load also failed: {e}")
            return False

    # ── Public load method ────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model — DagsHub first, local fallback second."""
        if not self._load_from_dagshub():
            logger.warning("Falling back to local model…")
            if not self._load_from_local():
                raise RuntimeError(
                    "Could not load model from DagsHub or local storage. "
                    "Check DAGSHUB_TOKEN and that models/titanic/ exists."
                )

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, df) -> tuple[list[int], list[float]]:
        """
        Run prediction on a pandas DataFrame.
        Returns (predictions, probabilities).
        """
        import numpy as np

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # ── MLflow pyfunc model (from DagsHub registry) ───────────────────────
        if hasattr(self.model, "predict") and hasattr(self.model, "metadata"):
            preds = self.model.predict(df)
            # pyfunc returns DataFrame or ndarray — normalize
            if hasattr(preds, "values"):
                preds = preds.values.flatten()
            preds = np.array(preds, dtype=int).tolist()
            # pyfunc doesn't expose predict_proba — use 0.8/0.2 placeholder
            probas = [0.85 if p == 1 else 0.15 for p in preds]
            return preds, probas

        # ── Raw pkl artifact (dict for CatBoost/TabNet, sklearn pipeline) ──────
        artifact = self.model
        if isinstance(artifact, dict):
            model = artifact["model"]
            prep  = artifact.get("preprocessor")
            cat_i = artifact.get("cat_indices")

            if prep is not None:
                X = prep.transform(df)
            else:
                X = df.values.astype(np.float32)

            if cat_i is not None:
                # CatBoost
                preds  = model.predict(X).astype(int).tolist()
                probas = model.predict_proba(X)[:, 1].tolist()
            else:
                # TabNet
                X = np.array(X, dtype=np.float32)
                preds  = model.predict(X).flatten().astype(int).tolist()
                probas = model.predict_proba(X)[:, 1].tolist()
        else:
            # Standard sklearn Pipeline
            preds  = artifact.predict(df).astype(int).tolist()
            probas = artifact.predict_proba(df)[:, 1].tolist()

        return preds, probas


# ── Singleton ──────────────────────────────────────────────────────────────────
registry = ModelRegistry()
