"""Tests for evaluation metrics."""
import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_metrics,
    get_classification_report,
    summarise_cv_scores,
)


@pytest.fixture
def binary_preds():
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 0])
    y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.85, 0.6, 0.8, 0.35, 0.15, 0.05])
    return y_true, y_pred, y_proba


class TestComputeMetrics:
    def test_returns_dict(self, binary_preds):
        y_true, y_pred, y_proba = binary_preds
        result = compute_metrics(y_true, y_pred, y_proba)
        assert isinstance(result, dict)

    def test_all_keys_present(self, binary_preds):
        y_true, y_pred, y_proba = binary_preds
        result = compute_metrics(y_true, y_pred, y_proba)
        for key in ("accuracy", "recall", "precision", "f1", "roc_auc", "log_loss"):
            assert key in result, f"Missing metric: {key}"

    def test_values_in_range(self, binary_preds):
        y_true, y_pred, y_proba = binary_preds
        result = compute_metrics(y_true, y_pred, y_proba)
        for key in ("accuracy", "recall", "precision", "f1", "roc_auc"):
            assert 0.0 <= result[key] <= 1.0, f"{key} out of range"

    def test_prefix_applied(self, binary_preds):
        y_true, y_pred, y_proba = binary_preds
        result = compute_metrics(y_true, y_pred, y_proba, prefix="test_")
        assert "test_accuracy" in result
        assert "test_roc_auc" in result

    def test_no_proba_skips_auc(self, binary_preds):
        y_true, y_pred, _ = binary_preds
        result = compute_metrics(y_true, y_pred, None)
        assert "roc_auc" not in result
        assert "log_loss" not in result
        assert "accuracy" in result


class TestClassificationReport:
    def test_returns_string(self, binary_preds):
        y_true, y_pred, _ = binary_preds
        report = get_classification_report(y_true, y_pred)
        assert isinstance(report, str)
        assert "Not Survived" in report
        assert "Survived" in report


class TestSummariseCvScores:
    def test_output_keys(self):
        cv_scores = {"accuracy": [0.8, 0.82, 0.79], "roc_auc": [0.85, 0.87, 0.84]}
        result = summarise_cv_scores(cv_scores)
        assert "cv_accuracy_mean" in result
        assert "cv_accuracy_std" in result
        assert "cv_roc_auc_mean" in result

    def test_mean_correct(self):
        cv_scores = {"f1": [0.7, 0.8, 0.9]}
        result = summarise_cv_scores(cv_scores)
        assert abs(result["cv_f1_mean"] - 0.8) < 1e-5
