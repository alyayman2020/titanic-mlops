"""Shared test fixtures."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_titanic_df():
    """Synthetic Titanic-like DataFrame (50 rows) for fast tests."""
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame({
        "PassengerId": range(1, n + 1),
        "Survived":    rng.integers(0, 2, n),
        "Pclass":      rng.choice([1, 2, 3], n),
        "Name":        [f"Doe, Mr. John {i}" for i in range(n)],
        "Sex":         rng.choice(["male", "female"], n),
        "Age":         np.where(rng.random(n) > 0.2, rng.integers(1, 80, n).astype(float), np.nan),
        "SibSp":       rng.integers(0, 4, n),
        "Parch":       rng.integers(0, 4, n),
        "Ticket":      [f"T{i}" for i in range(n)],
        "Fare":        rng.uniform(5, 300, n),
        "Cabin":       [None if rng.random() > 0.3 else f"C{i}" for i in range(n)],
        "Embarked":    rng.choice(["S", "C", "Q", None], n),
    })


@pytest.fixture
def X_y(sample_titanic_df):
    df = sample_titanic_df
    y = df["Survived"]
    X = df.drop(columns=["Survived"])
    return X, y
