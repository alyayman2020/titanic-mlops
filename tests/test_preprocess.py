"""Tests for StandardPreprocessor and CatBoostPreprocessor."""
import numpy as np
import pytest

from src.preprocessing.preprocess import CatBoostPreprocessor, StandardPreprocessor


class TestStandardPreprocessor:
    def test_fit_returns_self(self, X_y):
        X, y = X_y
        prep = StandardPreprocessor()
        result = prep.fit(X, y)
        assert result is prep

    def test_learned_attributes_after_fit(self, X_y):
        X, y = X_y
        prep = StandardPreprocessor().fit(X, y)
        assert hasattr(prep, "age_medians_")
        assert hasattr(prep, "embarked_mode_")
        assert hasattr(prep, "fare_median_")
        assert hasattr(prep, "fare_bins_")

    def test_transform_returns_float32_array(self, X_y):
        X, y = X_y
        prep = StandardPreprocessor().fit(X, y)
        out = prep.transform(X)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32

    def test_no_nan_in_output(self, X_y):
        X, y = X_y
        prep = StandardPreprocessor().fit(X, y)
        out = prep.transform(X)
        assert not np.isnan(out).any(), "Output contains NaN values"

    def test_no_object_columns_in_output(self, X_y):
        X, y = X_y
        prep = StandardPreprocessor().fit(X, y)
        out = prep.transform(X)
        # result is ndarray — no object columns possible
        assert out.ndim == 2

    def test_drop_cols_removed(self, X_y):
        X, y = X_y
        # The output should NOT have the same column count as X
        # because Name, Ticket, Cabin, PassengerId are dropped
        prep = StandardPreprocessor().fit(X, y)
        out = prep.transform(X)
        assert out.shape[1] < X.shape[1]

    def test_fit_transform_consistency(self, X_y):
        X, y = X_y
        prep = StandardPreprocessor()
        out1 = prep.fit_transform(X, y)
        out2 = prep.transform(X)
        np.testing.assert_array_equal(out1, out2)

    def test_age_imputed_no_nan(self, X_y):
        X, _ = X_y
        # Artificially null out all ages
        X = X.copy()
        X["Age"] = np.nan
        prep = StandardPreprocessor().fit(X)
        out = prep.transform(X)
        assert not np.isnan(out).any()


class TestCatBoostPreprocessor:
    def test_fit_returns_self(self, X_y):
        X, y = X_y
        prep = CatBoostPreprocessor()
        assert prep.fit(X, y) is prep

    def test_transform_returns_dataframe(self, X_y):
        X, y = X_y
        prep = CatBoostPreprocessor().fit(X, y)
        out = prep.transform(X)
        import pandas as pd
        assert isinstance(out, pd.DataFrame)

    def test_sex_remains_string(self, X_y):
        X, y = X_y
        prep = CatBoostPreprocessor().fit(X, y)
        out = prep.transform(X)
        assert out["Sex"].dtype == object

    def test_embarked_remains_string(self, X_y):
        X, y = X_y
        prep = CatBoostPreprocessor().fit(X, y)
        out = prep.transform(X)
        assert out["Embarked"].dtype == object

    def test_no_nan_numerics(self, X_y):
        X, y = X_y
        prep = CatBoostPreprocessor().fit(X, y)
        out = prep.transform(X)
        numeric_cols = out.select_dtypes(include="number").columns
        assert not out[numeric_cols].isna().any().any()

    def test_cat_feature_indices_valid(self, X_y):
        X, y = X_y
        prep = CatBoostPreprocessor().fit(X, y)
        out = prep.transform(X)
        indices = prep.get_cat_feature_indices(out)
        assert isinstance(indices, list)
        assert all(isinstance(i, int) for i in indices)
        assert all(0 <= i < out.shape[1] for i in indices)
