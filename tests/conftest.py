"""Shared pytest fixtures for the Titanic MLOps test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal synthetic Titanic DataFrame that mirrors the real schema."""
    np.random.seed(42)
    n = 120

    titles = ["Mr", "Mrs", "Miss", "Master", "Other"]
    title_weights = [0.55, 0.20, 0.15, 0.07, 0.03]
    chosen_titles = np.random.choice(titles, n, p=title_weights)

    names = [f"Smith, {t}. John {i}" for i, t in enumerate(chosen_titles)]

    return pd.DataFrame(
        {
            "PassengerId": range(1, n + 1),
            "Survived": np.random.randint(0, 2, n),
            "Pclass": np.random.choice([1, 2, 3], n),
            "Name": names,
            "Sex": np.random.choice(["male", "female"], n),
            "Age": np.where(np.random.rand(n) > 0.2, np.random.uniform(1, 80, n), np.nan),
            "SibSp": np.random.randint(0, 5, n),
            "Parch": np.random.randint(0, 4, n),
            "Ticket": [f"PC{i}" for i in range(n)],
            "Fare": np.where(np.random.rand(n) > 0.1, np.random.uniform(5, 500, n), np.nan),
            "Cabin": [f"C{i}" if np.random.rand() > 0.7 else None for i in range(n)],
            "Embarked": np.random.choice(["S", "C", "Q", None], n, p=[0.72, 0.19, 0.08, 0.01]),
        }
    )


@pytest.fixture
def X_y(sample_df):
    """Return features and target from sample_df."""
    X = sample_df.drop(columns=["Survived"])
    y = sample_df["Survived"]
    return X, y
