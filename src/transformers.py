"""sklearn-compatible transformer wrappers around the cleaning and feature engineering modules.

These classes allow the full preprocessing pipeline to be serialized as a single
sklearn Pipeline object, eliminating training-serving skew.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.data_cleaning import apply_cleaning, fit_cleaning_params
from src.feature_engineering import apply_features, fit_feature_params


class CleaningTransformer(BaseEstimator, TransformerMixin):
    """Wraps fit_cleaning_params / apply_cleaning as a stateful sklearn transformer."""

    def __init__(self, winsor_quantile: float = 0.995) -> None:
        self.winsor_quantile = winsor_quantile

    def fit(self, X: pd.DataFrame, y=None) -> "CleaningTransformer":
        self.params_ = fit_cleaning_params(X, winsor_quantile=self.winsor_quantile)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # drop_invalid_age=False: row count must stay stable inside a Pipeline.
        # Age >= 18 is enforced as an API precondition before calling predict_proba.
        return apply_cleaning(X, self.params_, drop_invalid_age=False)


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Wraps fit_feature_params / apply_features as a stateful sklearn transformer."""

    def __init__(self, winsor_quantile: float = 0.995) -> None:
        self.winsor_quantile = winsor_quantile

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineeringTransformer":
        self.params_ = fit_feature_params(X, winsor_quantile=self.winsor_quantile)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return apply_features(X, self.params_)
