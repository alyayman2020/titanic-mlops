"""
Preprocessing for Titanic dataset.

Two paths:
  1. StandardPreprocessor  — full feature engineering + encoding (all models except CatBoost)
  2. CatBoostPreprocessor  — light clean only; Sex/Embarked remain as raw strings
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ══════════════════════════════════════════════════════════════════════════════
#  Helper
# ══════════════════════════════════════════════════════════════════════════════

TITLE_MAP = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Dr": "Officer",
    "Rev": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Mlle": "Miss",
    "Countess": "Royalty",
    "Ms": "Mrs",
    "Lady": "Royalty",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Dona": "Royalty",
    "Mme": "Mrs",
    "Capt": "Officer",
    "Sir": "Royalty",
}

INITIAL_ORDER = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Officer": 4, "Royalty": 5}
SEX_MAP = {"male": 0, "female": 1}
EMBARKED_MAP = {"S": 0, "C": 1, "Q": 2}


# ══════════════════════════════════════════════════════════════════════════════
#  1. Standard Preprocessor
# ══════════════════════════════════════════════════════════════════════════════

class StandardPreprocessor(BaseEstimator, TransformerMixin):
    """
    Full feature-engineering + ordinal encoding for all non-CatBoost models.

    Fit statistics learned only from training data (no leakage).
    """

    def __init__(self, drop_cols: list[str] | None = None):
        self.drop_cols = drop_cols or ["Name", "Ticket", "Cabin", "PassengerId"]

    # ── private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_title(name: pd.Series) -> pd.Series:
        title = name.str.extract(r" ([A-Za-z]+)\.", expand=False)
        return title.map(TITLE_MAP).fillna("Mr")

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        df["Initial"] = self._extract_title(df["Name"])

        # Per-title median Age (learned from train only)
        self.age_medians_: dict[str, float] = (
            df.groupby("Initial")["Age"].median().to_dict()
        )
        # Global fallback
        self.age_global_median_: float = df["Age"].median()

        # Mode for Embarked
        self.embarked_mode_: str = df["Embarked"].mode()[0]

        # Median for Fare
        self.fare_median_: float = df["Fare"].median()

        # Fare quantile bins (learned from train)
        self.fare_bins_ = pd.qcut(
            df["Fare"].fillna(self.fare_median_), q=4, retbins=True, duplicates="drop"
        )[1]

        return self

    # ── transform ────────────────────────────────────────────────────────────

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        df = X.copy()

        # 1. Title extraction
        df["Initial"] = self._extract_title(df["Name"])

        # 2. Impute Age using per-title median
        for title, median in self.age_medians_.items():
            mask = df["Age"].isna() & (df["Initial"] == title)
            df.loc[mask, "Age"] = median
        df["Age"] = df["Age"].fillna(self.age_global_median_)

        # 3. Impute Embarked + Fare
        df["Embarked"] = df["Embarked"].fillna(self.embarked_mode_)
        df["Fare"] = df["Fare"].fillna(self.fare_median_)

        # 4. Feature engineering
        df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
        df["Alone"] = (df["Family_Size"] == 1).astype(int)
        df["Age_band"] = pd.cut(
            df["Age"], bins=[0, 12, 18, 35, 60, 120], labels=[0, 1, 2, 3, 4]
        ).astype(int)
        df["Fare_cat"] = pd.cut(
            df["Fare"],
            bins=self.fare_bins_,
            labels=False,
            include_lowest=True,
        ).fillna(0).astype(int)

        # 5. Encode categorical columns
        df["Sex"] = df["Sex"].map(SEX_MAP).fillna(0).astype(int)
        df["Embarked"] = df["Embarked"].map(EMBARKED_MAP).fillna(0).astype(int)
        df["Initial"] = df["Initial"].map(INITIAL_ORDER).fillna(0).astype(int)

        # 6. Drop unused columns
        df.drop(columns=[c for c in self.drop_cols if c in df.columns], inplace=True)

        # 7. Drop remaining string/object columns (safety net)
        obj_cols = df.select_dtypes(include="object").columns.tolist()
        if obj_cols:
            df.drop(columns=obj_cols, inplace=True)

        return df.values.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  2. CatBoost Preprocessor  (raw categoricals — NO encoding)
# ══════════════════════════════════════════════════════════════════════════════

class CatBoostPreprocessor(BaseEstimator, TransformerMixin):
    """
    Light-clean only for CatBoost.

    Keeps Sex, Embarked, Pclass as raw strings/ints (CatBoost handles them
    natively via cat_features index list).  Drops Name, Ticket, Cabin.
    """

    def __init__(
        self,
        drop_cols: list[str] | None = None,
        cat_features: list[str] | None = None,
    ):
        self.drop_cols = drop_cols or ["Name", "Ticket", "Cabin", "PassengerId"]
        self.cat_features = cat_features or ["Sex", "Embarked", "Pclass"]

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        self.age_median_: float = df["Age"].median()
        self.fare_median_: float = df["Fare"].median()
        self.embarked_mode_: str = str(df["Embarked"].mode()[0])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Impute numerics only
        df["Age"] = df["Age"].fillna(self.age_median_)
        df["Fare"] = df["Fare"].fillna(self.fare_median_)

        # Impute Embarked (keep as string)
        df["Embarked"] = df["Embarked"].fillna(self.embarked_mode_).astype(str)

        # Keep Sex as string, Pclass as string
        df["Sex"] = df["Sex"].astype(str)
        df["Pclass"] = df["Pclass"].astype(str)

        # Engineer numeric features (safe to add)
        df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
        df["Alone"] = (df["Family_Size"] == 1).astype(int)

        # Drop unusable columns
        df.drop(columns=[c for c in self.drop_cols if c in df.columns], inplace=True)

        return df

    def get_cat_feature_indices(self, df: pd.DataFrame) -> list[int]:
        """Return column indices of categorical features after transform."""
        transformed = self.transform(df.copy())  # use a copy for index lookup
        return [
            transformed.columns.tolist().index(c)
            for c in self.cat_features
            if c in transformed.columns
        ]
