"""Model training module — builds and evaluates two sklearn model pipelines.

Models
------
1. RandomForestClassifier  — ensemble of decision trees, robust to outliers.
2. AdaBoostClassifier      — adaptive boosting, strong on structured tabular data.

Each model is wrapped in a full sklearn Pipeline:
    TitanicPreprocessor  ->  Classifier

This means the saved .pkl artifact contains preprocessing + classification in
one object, so no separate preprocessing step is needed at inference time.
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
import yaml

from src.training.preprocess import TitanicPreprocessor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_params(params_path: str = "params.yaml") -> dict:
    """Load pipeline parameters from YAML file."""
    with open(params_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_data(raw_path: str, logger) -> pd.DataFrame:
    """Load raw Titanic CSV from disk.

    Parameters
    ----------
    raw_path : str
        Path to the raw train.csv file.
    logger : loguru.logger
        Logger instance.

    Returns
    -------
    pd.DataFrame
        Full raw DataFrame.
    """
    logger.info(f"Loading data from: {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df)} rows x {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def build_rf_pipeline(params: dict) -> Pipeline:
    """Build a sklearn Pipeline: TitanicPreprocessor -> RandomForestClassifier.

    Parameters
    ----------
    params : dict
        RandomForest hyperparameters from params.yaml.

    Returns
    -------
    Pipeline
        Unfitted sklearn Pipeline.
    """
    return Pipeline(
        steps=[
            ("preprocessor", TitanicPreprocessor()),
            ("classifier", RandomForestClassifier(**params)),
        ]
    )


def build_adaboost_pipeline(params: dict) -> Pipeline:
    """Build a sklearn Pipeline: TitanicPreprocessor -> AdaBoostClassifier.

    Parameters
    ----------
    params : dict
        AdaBoost hyperparameters from params.yaml.

    Returns
    -------
    Pipeline
        Unfitted sklearn Pipeline.
    """
    return Pipeline(
        steps=[
            ("preprocessor", TitanicPreprocessor()),
            ("classifier", AdaBoostClassifier(**params)),
        ]
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def cross_validate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    cv_folds: int,
    logger,
) -> float:
    """Run stratified k-fold cross-validation and log results.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted sklearn Pipeline.
    X : pd.DataFrame
        Full feature set (before split).
    y : pd.Series
        Target labels.
    model_name : str
        Human-readable name for logging.
    cv_folds : int
        Number of CV folds.
    logger : loguru.logger
        Logger instance.

    Returns
    -------
    float
        Mean CV accuracy.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    logger.info(
        f"[{model_name}] CV Accuracy ({cv_folds}-fold): "
        f"{scores.mean():.4f} (+/- {scores.std():.4f})"
    )
    return scores.mean()


def evaluate_on_test(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    logger,
) -> dict:
    """Evaluate fitted pipeline on held-out test set and log a full report.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn Pipeline.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True test labels.
    model_name : str
        Human-readable name for logging.
    logger : loguru.logger
        Logger instance.

    Returns
    -------
    dict
        Dictionary of metric name -> value.
    """
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    logger.info(f"[{model_name}] Test Accuracy : {metrics['accuracy']:.4f}")
    logger.info(f"[{model_name}] Test Precision: {metrics['precision']:.4f}")
    logger.info(f"[{model_name}] Test Recall   : {metrics['recall']:.4f}")
    logger.info(f"[{model_name}] Test F1       : {metrics['f1']:.4f}")
    logger.info(f"[{model_name}] Classification Report:\n{classification_report(y_test, y_pred)}")

    return metrics


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------


def save_model(pipeline: Pipeline, output_dir: str, filename: str, logger) -> str:
    """Serialise a fitted pipeline to disk using joblib.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn Pipeline to save.
    output_dir : str
        Directory to write the model file.
    filename : str
        Name of the output file (e.g. 'random_forest.pkl').
    logger : loguru.logger
        Logger instance.

    Returns
    -------
    str
        Full path of the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    joblib.dump(pipeline, path)
    logger.info(f"Model saved -> {path}")
    return path


def load_model(model_path: str):
    """Load a serialised pipeline from disk.

    Parameters
    ----------
    model_path : str
        Path to the .pkl file.

    Returns
    -------
    Pipeline
        Deserialised sklearn Pipeline.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def run_training(params: dict, logger) -> dict:
    """Execute the full training pipeline for both models.

    Steps
    -----
    1. Load raw data from disk.
    2. Stratified train/test split.
    3. Cross-validate both pipelines on training data.
    4. Fit both pipelines on the full training set.
    5. Evaluate both on the held-out test set.
    6. Save both model artifacts as .pkl files.
    7. Report comparison and declare winner by F1.

    Parameters
    ----------
    params : dict
        Full params.yaml config dictionary.
    logger : loguru.logger
        Configured logger.

    Returns
    -------
    dict
        Metrics for both models.
    """
    logger.info("=" * 60)
    logger.info("TITANIC MLOPS - TRAINING PIPELINE")
    logger.info("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────────
    data_cfg = params["data"]
    df = load_data(data_cfg["raw_path"], logger)

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # ── 2. Train / test split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
        X,
        y,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
        stratify=y,
    )
    logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    model_cfg = params["models"]
    output_dir = params["model"]["output_dir"]
    cv_folds = 10

    # ── 3. Build pipelines ────────────────────────────────────────────────
    rf_pipeline = build_rf_pipeline(model_cfg["random_forest"])
    ada_pipeline = build_adaboost_pipeline(model_cfg["adaboost"])

    # ── 4. Cross-validation ───────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Cross-Validation (on training data)")
    logger.info("-" * 60)
    cross_validate_pipeline(rf_pipeline, X_train, y_train, "RandomForest", cv_folds, logger)
    cross_validate_pipeline(ada_pipeline, X_train, y_train, "AdaBoost", cv_folds, logger)

    # ── 5. Fit on full training set ───────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Fitting on full training set")
    logger.info("-" * 60)
    rf_pipeline.fit(X_train, y_train)
    logger.info("RandomForest fit complete.")
    ada_pipeline.fit(X_train, y_train)
    logger.info("AdaBoost fit complete.")

    # ── 6. Evaluate on test set ───────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Test Set Evaluation")
    logger.info("-" * 60)
    rf_metrics = evaluate_on_test(rf_pipeline, X_test, y_test, "RandomForest", logger)
    ada_metrics = evaluate_on_test(ada_pipeline, X_test, y_test, "AdaBoost", logger)

    # ── 7. Save models ────────────────────────────────────────────────────
    save_model(rf_pipeline, output_dir, "random_forest.pkl", logger)
    save_model(ada_pipeline, output_dir, "adaboost.pkl", logger)

    # ── 8. Comparison summary ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)
    results = {"RandomForest": rf_metrics, "AdaBoost": ada_metrics}

    header = f"{'Metric':<14}{'RandomForest':>18}{'AdaBoost':>18}"
    logger.info(header)
    logger.info("-" * len(header))
    for metric in ["accuracy", "precision", "recall", "f1"]:
        logger.info(f"{metric:<14}{rf_metrics[metric]:>18.4f}{ada_metrics[metric]:>18.4f}")

    winner = max(results, key=lambda k: results[k]["f1"])
    logger.info("-" * len(header))
    logger.info(f"Winner by F1: {winner}")
    logger.info("=" * 60)

    return results
