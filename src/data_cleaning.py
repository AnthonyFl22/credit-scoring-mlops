"""Data cleansing for the dataset

The exploratory analysis in ``notebooks/EDA.ipynb`` surfaced several data-quality
problems that must be fixed before modeling:

* ``age`` has an impossible value of 0.
* The three past-due counters carry the sentinel codes 96 and 98, which are not
  real counts but placeholder/error codes (269 rows, ~55% default rate).
* ``RevolvingUtilizationOfUnsecuredLines``, ``DebtRatio`` and ``MonthlyIncome``
  have heavy, contaminated right tails (e.g. utilization up to 50,708).
* ``MonthlyIncome`` (~20%) and ``NumberOfDependents`` (~2.6%) have missing values.

Strategy:

#. Drop rows where ``age`` is below 18 (impossible values; negligible count).
#. Cap the past-due sentinels (96/98) to the largest *real* value seen in each
   column, keeping the "heavy delinquency" signal without the absurd magnitude.
#. Winsorize the heavy-tailed continuous features at a high percentile.
#. Impute missing ``MonthlyIncome`` with the median and add a missingness flag
   (missingness is informative here). Impute missing ``NumberOfDependents`` with 0.

All thresholds are *fitted on the training set only* (:func:`fit_cleaning_params`)
and then *applied* to any split (:func:`apply_cleaning`), so the test set never
leaks information into the cleaning parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict
import json

import pandas as pd

# column groups for cleaning steps

PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

# Heavy-tailed continuous features that get winsorized.
WINSORIZE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "MonthlyIncome",
]

# Values >= this in a past-due counter are treated as sentinel/error codes.
SENTINEL_THRESHOLD = 90

# Minimum plausible age for a credit applicant.
MIN_VALID_AGE = 18


# DebtRatio is only reliable as a ratio when MonthlyIncome is present and positive.
# High DebtRatio values are strongly associated with missing/zero income, so the cap
# is learned only from rows with valid MonthlyIncome.

DEBT_RATIO_VALID_INCOME_MIN = 1.0
DEBT_RATIO_MAX_CAP = 10.0

@dataclass
class CleaningParams:
    """Thresholds learned from the training data and reused on every split."""

    income_median: float
    dependents_fill: float
    # column -> realistic max used to cap sentinel past-due codes
    past_due_caps: Dict[str, float] = field(default_factory=dict)
    # column -> upper bound used for winsorization
    winsor_caps: Dict[str, float] = field(default_factory=dict)
    winsor_quantile: float = 0.995

    def to_json(self, path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path) -> "CleaningParams":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(**json.load(fh))


def fit_cleaning_params(df: pd.DataFrame,winsor_quantile: float = 0.995) -> CleaningParams:
    
    """Learn cleaning thresholds from a (training) dataframe."""
    past_due_caps = {
        col: float(df.loc[df[col] < SENTINEL_THRESHOLD, col].max())
        for col in PAST_DUE_COLS
    }

    winsor_caps = {
        col: float(df[col].quantile(winsor_quantile))
        for col in WINSORIZE_COLS
        if col != "DebtRatio"
    }

    valid_income_mask = (
        df["MonthlyIncome"].notna()
        & (df["MonthlyIncome"] > DEBT_RATIO_VALID_INCOME_MIN)
    )

    debt_ratio_cap = float(
        df.loc[valid_income_mask, "DebtRatio"].quantile(winsor_quantile)
    )

    winsor_caps["DebtRatio"] = min(debt_ratio_cap, DEBT_RATIO_MAX_CAP)

    return CleaningParams(
        income_median=float(df["MonthlyIncome"].median()),
        dependents_fill=0.0,
        past_due_caps=past_due_caps,
        winsor_caps=winsor_caps,
        winsor_quantile=winsor_quantile,
    )

def apply_cleaning(df: pd.DataFrame, params: CleaningParams) -> pd.DataFrame:
    """Return a cleaned copy of ``df`` using previously fitted ``params``."""
    out = df.copy()

    # 1. Drop impossible ages.
    out = out[out["age"] >= MIN_VALID_AGE]

    # 2. Cap sentinel past-due codes (96/98) to the realistic per-column max.
    for col, cap in params.past_due_caps.items():
        out[col] = out[col].clip(upper=cap)

    # 3. Flag then impute missing income; impute missing dependents with 0.
    out["MonthlyIncome_was_missing"] = out["MonthlyIncome"].isna().astype(int)
    out["MonthlyIncome"] = out["MonthlyIncome"].fillna(params.income_median)
    out["NumberOfDependents"] = out["NumberOfDependents"].fillna(params.dependents_fill)

    # 4. Winsorize heavy-tailed continuous features.
    for col, cap in params.winsor_caps.items():
        out[col] = out[col].clip(upper=cap)

    return out

def clean_training_data(df: pd.DataFrame, winsor_quantile: float = 0.995):
    """Convenience: fit on ``df`` and return ``(cleaned_df, params)``."""
    params = fit_cleaning_params(df, winsor_quantile=winsor_quantile)
    return apply_cleaning(df, params), params
