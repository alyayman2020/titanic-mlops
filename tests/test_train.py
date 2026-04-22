"""Unit tests for model training, evaluation, and persistence."""

import os

import pytest
from sklearn.pipeline import Pipeline

from src.training.train import (
    build_adaboost_pipeline,
    build_rf_pipeline,
    evaluate_on_test,
    load_model,
    save_model,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RF_PARAMS = {"n_estimators": 5, "random_state": 42}
ADA_PARAMS = {"n_estimators": 5, "learning_rate": 0.5, "random_state": 42}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def trained_rf(X_y):
    X, y = X_y
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = build_rf_pipeline(RF_PARAMS)
    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test


@pytest.fixture
def trained_ada(X_y):
    X, y = X_y
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = build_adaboost_pipeline(ADA_PARAMS)
    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test


# ---------------------------------------------------------------------------
# Pipeline builder tests
# ---------------------------------------------------------------------------


class TestBuildPipelines:
    def test_rf_returns_pipeline(self):
        pipe = build_rf_pipeline(RF_PARAMS)
        assert isinstance(pipe, Pipeline)

    def test_rf_has_correct_steps(self):
        pipe = build_rf_pipeline(RF_PARAMS)
        step_names = list(pipe.named_steps.keys())
        assert step_names == ["preprocessor", "classifier"]

    def test_ada_returns_pipeline(self):
        pipe = build_adaboost_pipeline(ADA_PARAMS)
        assert isinstance(pipe, Pipeline)

    def test_ada_has_correct_steps(self):
        pipe = build_adaboost_pipeline(ADA_PARAMS)
        step_names = list(pipe.named_steps.keys())
        assert step_names == ["preprocessor", "classifier"]

    def test_rf_classifier_type(self):
        from sklearn.ensemble import RandomForestClassifier

        pipe = build_rf_pipeline(RF_PARAMS)
        assert isinstance(pipe.named_steps["classifier"], RandomForestClassifier)

    def test_ada_classifier_type(self):
        from sklearn.ensemble import AdaBoostClassifier

        pipe = build_adaboost_pipeline(ADA_PARAMS)
        assert isinstance(pipe.named_steps["classifier"], AdaBoostClassifier)


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------


class TestTrainPipelines:
    def test_rf_fits_and_predicts(self, trained_rf):
        pipe, X_test, _ = trained_rf
        preds = pipe.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1})

    def test_ada_fits_and_predicts(self, trained_ada):
        pipe, X_test, _ = trained_ada
        preds = pipe.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1})

    def test_rf_has_classes_after_fit(self, trained_rf):
        pipe, _, _ = trained_rf
        assert hasattr(pipe.named_steps["classifier"], "classes_")

    def test_ada_has_classes_after_fit(self, trained_ada):
        pipe, _, _ = trained_ada
        assert hasattr(pipe.named_steps["classifier"], "classes_")


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluateOnTest:
    def test_returns_all_metric_keys(self, trained_rf):
        from src.logger import get_logger

        logger = get_logger()
        pipe, X_test, y_test = trained_rf
        metrics = evaluate_on_test(pipe, X_test, y_test, "RF_test", logger)
        for key in ["accuracy", "precision", "recall", "f1"]:
            assert key in metrics

    def test_accuracy_in_valid_range(self, trained_rf):
        from src.logger import get_logger

        logger = get_logger()
        pipe, X_test, y_test = trained_rf
        metrics = evaluate_on_test(pipe, X_test, y_test, "RF_test", logger)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_f1_in_valid_range(self, trained_rf):
        from src.logger import get_logger

        logger = get_logger()
        pipe, X_test, y_test = trained_rf
        metrics = evaluate_on_test(pipe, X_test, y_test, "RF_test", logger)
        assert 0.0 <= metrics["f1"] <= 1.0


# ---------------------------------------------------------------------------
# Save / load tests
# ---------------------------------------------------------------------------


class TestSaveLoadModel:
    def test_save_creates_file(self, tmp_path, trained_rf):
        from src.logger import get_logger

        logger = get_logger()
        pipe, _, _ = trained_rf
        save_model(pipe, str(tmp_path), "rf.pkl", logger)
        assert (tmp_path / "rf.pkl").exists()

    def test_load_returns_pipeline(self, tmp_path, trained_rf):
        from src.logger import get_logger

        logger = get_logger()
        pipe, _, _ = trained_rf
        save_model(pipe, str(tmp_path), "rf.pkl", logger)
        loaded = load_model(str(tmp_path / "rf.pkl"))
        assert isinstance(loaded, Pipeline)

    def test_loaded_model_predicts_same(self, tmp_path, trained_rf):
        from src.logger import get_logger

        logger = get_logger()
        pipe, X_test, _ = trained_rf
        save_model(pipe, str(tmp_path), "rf.pkl", logger)
        loaded = load_model(str(tmp_path / "rf.pkl"))
        original_preds = pipe.predict(X_test)
        loaded_preds = loaded.predict(X_test)
        assert list(original_preds) == list(loaded_preds)

    def test_load_raises_on_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(str(tmp_path / "ghost.pkl"))

    def test_save_creates_output_dir(self, tmp_path, trained_rf):
        from src.logger import get_logger

        logger = get_logger()
        pipe, _, _ = trained_rf
        new_dir = str(tmp_path / "nested" / "models")
        save_model(pipe, new_dir, "rf.pkl", logger)
        assert os.path.exists(os.path.join(new_dir, "rf.pkl"))
