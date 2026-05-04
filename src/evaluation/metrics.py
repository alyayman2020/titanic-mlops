"""
Evaluation module.

Computes all classification metrics and returns a flat dict
ready for MLflow logging.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    prefix: str = "",
) -> dict[str, float]:
    """
    Compute all classification metrics.

    Parameters
    ----------
    y_true  : ground-truth labels
    y_pred  : predicted class labels
    y_proba : predicted probabilities for the positive class (col 1)
    prefix  : optional string prepended to each metric key (e.g. "train_")

    Returns
    -------
    dict of metric_name -> float, ready for mlflow.log_metrics()
    """
    p = prefix

    metrics: dict[str, float] = {
        f"{p}accuracy":  round(accuracy_score(y_true, y_pred), 6),
        f"{p}recall":    round(recall_score(y_true, y_pred, zero_division=0), 6),
        f"{p}precision": round(precision_score(y_true, y_pred, zero_division=0), 6),
        f"{p}f1":        round(f1_score(y_true, y_pred, zero_division=0), 6),
    }

    if y_proba is not None:
        metrics[f"{p}roc_auc"] = round(roc_auc_score(y_true, y_proba), 6)
        metrics[f"{p}log_loss"] = round(log_loss(y_true, y_proba), 6)

    return metrics


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> str:
    """Return a formatted sklearn classification report string."""
    return classification_report(y_true, y_pred, target_names=["Not Survived", "Survived"])


def summarise_cv_scores(cv_scores: dict[str, list[float]]) -> dict[str, float]:
    """
    Flatten cross-validation score lists into mean/std pairs.

    Input:  {"accuracy": [0.82, 0.80, ...], ...}
    Output: {"cv_accuracy_mean": 0.81, "cv_accuracy_std": 0.01, ...}
    """
    out: dict[str, float] = {}
    for metric, values in cv_scores.items():
        arr = np.array(values)
        out[f"cv_{metric}_mean"] = round(float(arr.mean()), 6)
        out[f"cv_{metric}_std"] = round(float(arr.std()), 6)
    return out
