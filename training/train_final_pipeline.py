"""Build and serialize the v1 credit scoring artifact.

Creates models/credit_scoring/v1/ containing:
  model.joblib        — full sklearn Pipeline (CleaningTransformer + FeatureEngineeringTransformer + XGBoost)
  feature_schema.json — API input contract (types, ranges, descriptions)
  metrics.json        — cross-validated performance pulled from models/cv_results.json
  thresholds.json     — provisional decision bands

Note: model_card.md lives in the same directory but is maintained separately.

Usage (from repo root):
    python training/train_final_pipeline.py
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_cleaning import MIN_VALID_AGE
from src.model import CONFIGS, TARGET
from src.transformers import CleaningTransformer, FeatureEngineeringTransformer

V1_DIR = REPO_ROOT / "models" / "credit_scoring" / "v1"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_DIR = REPO_ROOT / "data" / "raw"

CONFIG_NAME = "baseline"
BEST_N_ESTIMATORS = 193
TRAINING_DATE = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

FEATURE_DESCRIPTIONS = {
    "RevolvingUtilizationOfUnsecuredLines": (
        "Total balance on credit cards and personal lines of credit divided by the sum of credit limits."
    ),
    "age": "Age of the borrower in years. Must be >= 18.",
    "NumberOfTime30-59DaysPastDueNotWorse": "Number of times 30-59 days past due in the last 2 years.",
    "DebtRatio": "Monthly debt payments (alimony, living costs) divided by monthly gross income.",
    "MonthlyIncome": (
        "Monthly gross income in USD. Null values are accepted and imputed with the training median."
    ),
    "NumberOfOpenCreditLinesAndLoans": "Number of open loans and lines of credit.",
    "NumberOfTimes90DaysLate": "Number of times 90 or more days past due.",
    "NumberRealEstateLoansOrLines": (
        "Number of mortgage and real estate loans including home equity lines of credit."
    ),
    "NumberOfTime60-89DaysPastDueNotWorse": "Number of times 60-89 days past due in the last 2 years.",
    "NumberOfDependents": (
        "Number of dependents in the household, excluding the borrower. "
        "Null values are accepted and imputed with 0."
    ),
}

NULLABLE_FEATURES = {"MonthlyIncome", "NumberOfDependents"}


FORCE_FLOAT = {"MonthlyIncome", "DebtRatio", "RevolvingUtilizationOfUnsecuredLines"}
FORCE_MIN = {"age": MIN_VALID_AGE}


def _build_feature_schema(processed_df: pd.DataFrame) -> dict:
    """Build the feature_schema.json contract from the processed training data."""
    features = []
    for name, description in FEATURE_DESCRIPTIONS.items():
        col = processed_df[name].dropna()
        is_integer = name not in FORCE_FLOAT and (col == col.astype(int)).all()
        features.append({
            "name": name,
            "dtype": "int" if is_integer else "float",
            "min": round(float(FORCE_MIN.get(name, col.min())), 6),
            "max": round(float(col.max()), 6),
            "required": name not in NULLABLE_FEATURES,
            "description": description,
        })
    return {
        "model_name": "credit_scoring",
        "version": "v1",
        "target": TARGET,
        "features": features,
    }


def _build_metrics_doc(cv_results: list) -> dict:
    """Build the metrics.json document from the saved CV results."""
    entry = next(r for r in cv_results if r["config"] == CONFIG_NAME)
    return {
        "model_name": "credit_scoring",
        "version": "v1",
        "trained_at": TRAINING_DATE,
        "config": CONFIG_NAME,
        "best_n_estimators": entry["best_n_estimators"],
        "metrics": {
            "roc_auc":       round(entry["roc_auc"], 4),
            "gini":          round(entry["gini"], 4),
            "ks":            round(entry["ks"], 4),
            "avg_precision": round(entry["avg_precision"], 4),
            "precision":     round(entry["precision"], 4),
            "recall":        round(entry["recall"], 4),
            "f1":            round(entry["f1"], 4),
        },
        "confusion_matrix": entry["confusion_matrix"],
        "cv_std_roc_auc": round(entry["std_roc_auc"], 4),
    }


def main() -> None:
    V1_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    train_raw = pd.read_csv(RAW_DIR / "cs-training.csv", index_col=0)
    train_raw = train_raw[train_raw["age"] >= MIN_VALID_AGE].copy()
    X = train_raw.drop(columns=[TARGET])
    y = train_raw[TARGET]
    print(f"  {len(X):,} rows | {y.mean():.2%} default rate")

    config = CONFIGS[CONFIG_NAME]
    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    xgb_kwargs = config.to_xgb_kwargs(scale_pos_weight)
    xgb_kwargs["n_estimators"] = BEST_N_ESTIMATORS
    xgb_kwargs.pop("eval_metric", None)

    pipeline = Pipeline([
        ("cleaning",   CleaningTransformer(winsor_quantile=0.995)),
        ("features",   FeatureEngineeringTransformer(winsor_quantile=0.995)),
        ("classifier", xgb.XGBClassifier(**xgb_kwargs)),
    ])

    print("Fitting pipeline on full training set...")
    pipeline.fit(X, y)
    print("  Done.")

    joblib.dump(pipeline, V1_DIR / "model.joblib")
    print(f"Saved: {V1_DIR / 'model.joblib'}")

    processed_df = pd.read_csv(PROCESSED_DIR / "cs-training-processed.csv", index_col=0)
    schema = _build_feature_schema(processed_df)
    (V1_DIR / "feature_schema.json").write_text(json.dumps(schema, indent=2))
    print(f"Saved: {V1_DIR / 'feature_schema.json'}")

    cv_results = json.loads((REPO_ROOT / "models" / "experiments" / "cv_results.json").read_text())
    metrics_doc = _build_metrics_doc(cv_results)
    (V1_DIR / "metrics.json").write_text(json.dumps(metrics_doc, indent=2))
    print(f"Saved: {V1_DIR / 'metrics.json'}")

    thresholds = {
        "approve_max": 0.15,
        "review_max": 0.40,
        "reject_min": 0.40,
        "notes": "Provisional thresholds. To be refined via cost-based threshold optimization.",
    }
    (V1_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    print(f"Saved: {V1_DIR / 'thresholds.json'}")

    print("\nv1 artifact complete.")


if __name__ == "__main__":
    main()
