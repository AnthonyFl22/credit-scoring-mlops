"""Feature engineering for the clean dataset.

All thresholds are *fitted on the training set only* (:func:`fit_feature_params`)
and then *applied* to any split (:func:`apply_features`).

New columns produced
* ``total_past_due``         — sum of the three delinquency counters
* ``any_past_due``           — binary flag: total_past_due > 0
* ``log_monthly_income``     — log1p of MonthlyIncome (compresses right skew)
* ``income_per_dependent``   — MonthlyIncome / (NumberOfDependents + 1)
* ``open_lines_per_age_yr``  — NumberOfOpenCreditLinesAndLoans / (age - 17), i.e. per year of eligible credit history
* ``age_bin``                — ordinal bucket: 0=young (18-35), 1=middle (36-55), 2=senior (56+)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import List

import numpy as np
import pandas as pd

PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

AGE_BIN_EDGES: List[int] = [18, 36, 56, 999]
AGE_BIN_LABELS: List[int] = [0, 1, 2]


@dataclass
class FeatureParams:
    """Parameters learned from training data; reused on every split."""

    income_per_dependent_cap: float
    open_lines_per_age_cap: float

    def to_json(self, path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path) -> "FeatureParams":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(**json.load(fh))


def fit_feature_params(df: pd.DataFrame, winsor_quantile: float = 0.995) -> FeatureParams:
    """Learn feature caps from a (training) dataframe."""
    income_per_dependent = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    open_lines_per_age = df["NumberOfOpenCreditLinesAndLoans"] / (df["age"] - 17)

    return FeatureParams(
        income_per_dependent_cap=float(income_per_dependent.quantile(winsor_quantile)),
        open_lines_per_age_cap=float(open_lines_per_age.quantile(winsor_quantile)),
    )


def apply_features(df: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered features appended."""
    out = df.copy()

    out["total_past_due"] = out[PAST_DUE_COLS].sum(axis=1)
    out["any_past_due"] = (out["total_past_due"] > 0).astype(int)

    out["log_monthly_income"] = np.log1p(out["MonthlyIncome"])

    out["income_per_dependent"] = (
        out["MonthlyIncome"] / (out["NumberOfDependents"] + 1)
    ).clip(upper=params.income_per_dependent_cap)

    out["open_lines_per_age_yr"] = (
        out["NumberOfOpenCreditLinesAndLoans"] / (out["age"] - 17)
    ).clip(upper=params.open_lines_per_age_cap)

    out["age_bin"] = pd.cut(
        out["age"],
        bins=AGE_BIN_EDGES,
        labels=AGE_BIN_LABELS,
        right=False,
    ).astype(int)

    return out


def build_features(df: pd.DataFrame, winsor_quantile: float = 0.995):
    """Convenience: fit on ``df`` and return ``(featured_df, params)``."""
    params = fit_feature_params(df, winsor_quantile=winsor_quantile)
    return apply_features(df, params), params
