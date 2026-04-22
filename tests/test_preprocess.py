"""Unit tests for TitanicPreprocessor custom sklearn transformer."""

import numpy as np
import pandas as pd
import pytest

from src.training.preprocess import TitanicPreprocessor


class TestTitanicPreprocessorFit:
    def test_fit_returns_self(self, X_y):
        X, _ = X_y
        prep = TitanicPreprocessor()
        result = prep.fit(X)
        assert result is prep

    def test_fit_learns_age_medians(self, X_y):
        X, _ = X_y
        prep = TitanicPreprocessor()
        prep.fit(X)
        assert hasattr(prep, "age_medians_")
        assert isinstance(prep.age_medians_, dict)
        assert len(prep.age_medians_) > 0

    def test_fit_learns_fare_median(self, X_y):
        X, _ = X_y
        prep = TitanicPreprocessor()
        prep.fit(X)
        assert hasattr(prep, "fare_median_")
        assert prep.fare_median_ > 0

    def test_fit_learns_embarked_mode(self, X_y):
        X, _ = X_y
        prep = TitanicPreprocessor()
        prep.fit(X)
        assert hasattr(prep, "embarked_mode_")
        assert prep.embarked_mode_ in {"S", "C", "Q"}


class TestTitanicPreprocessorTransform:
    @pytest.fixture(autouse=True)
    def fitted_prep(self, X_y):
        X, _ = X_y
        prep = TitanicPreprocessor()
        prep.fit(X)
        self.prep = prep
        self.X = X

    def test_transform_returns_dataframe(self):
        out = self.prep.transform(self.X)
        assert isinstance(out, pd.DataFrame)

    def test_no_nulls_after_transform(self):
        out = self.prep.transform(self.X)
        assert out.isnull().sum().sum() == 0, "NaN values remain after transform"

    def test_drop_cols_removed(self):
        out = self.prep.transform(self.X)
        for col in TitanicPreprocessor.DROP_COLS:
            assert col not in out.columns, f"Column '{col}' was not dropped"

    def test_engineered_cols_exist(self):
        out = self.prep.transform(self.X)
        for col in ["Age_band", "Family_Size", "Alone", "Fare_cat", "Initial"]:
            assert col in out.columns, f"Engineered column '{col}' is missing"

    def test_sex_encoded_binary(self):
        out = self.prep.transform(self.X)
        assert set(out["Sex"].dropna().unique()).issubset({0, 1})

    def test_alone_is_binary(self):
        out = self.prep.transform(self.X)
        assert set(out["Alone"].unique()).issubset({0, 1})

    def test_age_band_valid_range(self):
        out = self.prep.transform(self.X)
        assert out["Age_band"].between(0, 4).all()

    def test_fare_cat_valid_range(self):
        out = self.prep.transform(self.X)
        assert out["Fare_cat"].between(0, 3).all()

    def test_row_count_preserved(self):
        out = self.prep.transform(self.X)
        assert len(out) == len(self.X)

    def test_fit_transform_equivalent(self, X_y):
        X, _ = X_y
        prep1 = TitanicPreprocessor()
        out1 = prep1.fit_transform(X)
        prep2 = TitanicPreprocessor()
        prep2.fit(X)
        out2 = prep2.transform(X)
        pd.testing.assert_frame_equal(out1, out2)


class TestTitanicPreprocessorEdgeCases:
    def test_handles_all_missing_embarked(self, X_y):
        X, _ = X_y
        X = X.copy()
        X["Embarked"] = None
        prep = TitanicPreprocessor()
        prep.fit(X)
        out = prep.transform(X)
        assert out["Embarked"].isnull().sum() == 0

    def test_handles_all_missing_age(self, X_y):
        X, _ = X_y
        X = X.copy()
        X["Age"] = np.nan
        prep = TitanicPreprocessor()
        prep.fit(X)
        out = prep.transform(X)
        assert out.isnull().sum().sum() == 0
