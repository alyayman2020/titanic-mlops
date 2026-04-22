"""Custom sklearn-compatible preprocessing transformer for the Titanic dataset.

Design
------
TitanicPreprocessor is a single-step sklearn transformer that:
  1. Extracts a title (Initial) from the passenger Name column.
  2. Imputes missing Age values based on the median age per title group.
  3. Fills missing Embarked with the mode value ('S').
  4. Fills missing Fare with the median.
  5. Engineers new features: Age_band, Family_Size, Alone, Fare_cat.
  6. Encodes all categorical columns with ordinal integers.
  7. Drops raw/redundant columns that leak information or add no signal.

Wrapping all of this in BaseEstimator + TransformerMixin means the transformer
is safe inside a sklearn Pipeline — fit() learns statistics on training data
only, and transform() applies them consistently at inference time.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TitanicPreprocessor(BaseEstimator, TransformerMixin):
    """End-to-end feature engineering + encoding transformer for Titanic data.

    All imputation statistics (age medians per title, fare median) are learned
    during fit() and reused during transform() to prevent data leakage.

    Parameters
    ----------
    None — all hyperparameters are derived from the training data.
    """

    # Columns that will be dropped after feature engineering
    DROP_COLS = ["Name", "Age", "Ticket", "Fare", "Cabin", "PassengerId"]

    # Rare/foreign titles mapped to standard ones
    TITLE_MAP = {
        "Mlle": "Miss",
        "Mme": "Miss",
        "Ms": "Miss",
        "Dr": "Mr",
        "Major": "Mr",
        "Lady": "Mrs",
        "Countess": "Mrs",
        "Jonkheer": "Other",
        "Col": "Other",
        "Rev": "Other",
        "Capt": "Mr",
        "Sir": "Mr",
        "Don": "Mr",
        "Dona": "Mrs",
    }

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Fit: learn statistics from training data
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None) -> "TitanicPreprocessor":
        """Learn imputation statistics from training data.

        Parameters
        ----------
        X : pd.DataFrame
            Raw Titanic features (must include Name, Age, Fare, Embarked).
        y : ignored

        Returns
        -------
        self
        """
        df = X.copy()

        # Extract and map titles
        df["Initial"] = df["Name"].str.extract(r"([A-Za-z]+)\.")
        df["Initial"] = df["Initial"].replace(self.TITLE_MAP)

        # Learn median age per title group (for missing age imputation)
        self.age_medians_ = df.groupby("Initial")["Age"].median().to_dict()
        # Fallback if a title appears at inference that wasn't in training
        self.global_age_median_ = df["Age"].median()

        # Learn median fare (for missing fare imputation)
        self.fare_median_ = df["Fare"].median()

        # Learn mode for Embarked
        mode_series = df["Embarked"].mode()
        self.embarked_mode_ = mode_series[0] if len(mode_series) > 0 else "S"

        return self

    # ------------------------------------------------------------------
    # Transform: apply all engineering steps
    # ------------------------------------------------------------------

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Apply all feature engineering and encoding steps.

        Parameters
        ----------
        X : pd.DataFrame
            Raw Titanic features.
        y : ignored

        Returns
        -------
        pd.DataFrame
            Fully processed, model-ready DataFrame.
        """
        df = X.copy()

        # ── 1. Extract title ─────────────────────────────────────────────
        df["Initial"] = df["Name"].str.extract(r"([A-Za-z]+)\.")
        df["Initial"] = df["Initial"].replace(self.TITLE_MAP)

        # ── 2. Impute missing Age by title group ─────────────────────────
        for title, median_age in self.age_medians_.items():
            mask = df["Age"].isnull() & (df["Initial"] == title)
            df.loc[mask, "Age"] = median_age
        # Remaining NaNs (unseen titles at inference) → global median
        df["Age"] = df["Age"].fillna(self.global_age_median_)

        # ── 3. Impute missing Embarked ───────────────────────────────────
        df["Embarked"] = df["Embarked"].fillna(self.embarked_mode_)

        # ── 4. Impute missing Fare ───────────────────────────────────────
        df["Fare"] = df["Fare"].fillna(self.fare_median_)

        # ── 5. Feature engineering ───────────────────────────────────────
        # Age bands
        df["Age_band"] = 0
        df.loc[df["Age"] <= 16, "Age_band"] = 0
        df.loc[(df["Age"] > 16) & (df["Age"] <= 32), "Age_band"] = 1
        df.loc[(df["Age"] > 32) & (df["Age"] <= 48), "Age_band"] = 2
        df.loc[(df["Age"] > 48) & (df["Age"] <= 64), "Age_band"] = 3
        df.loc[df["Age"] > 64, "Age_band"] = 4

        # Family features
        df["Family_Size"] = df["SibSp"] + df["Parch"]
        df["Alone"] = (df["Family_Size"] == 0).astype(int)

        # Fare categories
        df["Fare_cat"] = 0
        df.loc[df["Fare"] <= 7.91, "Fare_cat"] = 0
        df.loc[(df["Fare"] > 7.91) & (df["Fare"] <= 14.454), "Fare_cat"] = 1
        df.loc[(df["Fare"] > 14.454) & (df["Fare"] <= 31), "Fare_cat"] = 2
        df.loc[df["Fare"] > 31, "Fare_cat"] = 3

        # ── 6. Encode categorical columns ────────────────────────────────
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
        df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
        df["Initial"] = df["Initial"].map({"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3, "Other": 4})

        # ── 7. Drop raw / redundant columns ─────────────────────────────
        cols_to_drop = [c for c in self.DROP_COLS if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        return df
