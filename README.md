# Credit Scoring MLOps

**Last updated:** 2026-06-08

Binary credit risk classifier that predicts the probability a borrower will experience serious delinquency (90+ days past due) within two years. Built as a full MLOps project — from raw data to a production-ready inference API — following good practices around data leakage prevention, reproducibility, and model documentation.

**Dataset:** [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — 150,000 US consumer credit records.

---

## Table of contents

1. [Project structure](#1-project-structure)
2. [Architecture overview](#2-architecture-overview)
3. [Prerequisites and setup](#3-prerequisites-and-setup)
4. [Data](#4-data)
5. [Step-by-step workflow](#5-step-by-step-workflow)
   - 5.1 [Process raw data](#51-process-raw-data)
   - 5.2 [Feature engineering](#52-feature-engineering)
   - 5.3 [Model selection via cross-validation](#53-model-selection-via-cross-validation)
   - 5.4 [Build the production pipeline](#54-build-the-production-pipeline)
   - 5.5 [Inspect feature importances](#55-inspect-feature-importances)
   - 5.6 [Run the inference API](#56-run-the-inference-api)
6. [Model details](#6-model-details)
7. [API reference](#7-api-reference)
8. [Model artifacts](#8-model-artifacts)
9. [Next steps](#9-next-steps)

---

## 1. Project structure

```
credit-scoring-mlops/
│
├── src/                            # Core library — importable by scripts and training
│   ├── data_cleaning.py            # Cleaning params: fit / apply / serialize
│   ├── feature_engineering.py      # Feature params: fit / apply / serialize
│   ├── transformers.py             # sklearn-compatible wrappers (CleaningTransformer,
│   │                               #   FeatureEngineeringTransformer)
│   └── model.py                    # XGBoost configs, CV logic, training utilities
│
├── scripts/                        # One-shot data processing scripts
│   ├── clean_data.py               # Fit cleaning params on train; apply to all splits
│   ├── build_features.py           # Fit feature params on train; apply to all splits
│   └── feature_importance.py       # Extract and plot XGBoost feature importances (gain)
│
├── training/
│   ├── train_model.py              # Run stratified k-fold CV; compare config presets
│   └── train_final_pipeline.py     # Build and serialize the production sklearn Pipeline
│
├── api/
│   ├── config.py                   # Centralized artifact paths
│   ├── schemas.py                  # Pydantic v2 request / response models
│   └── main.py                     # FastAPI application (3 endpoints)
│
├── models/
│   ├── credit_scoring/v1/          # Production artifact (versioned)
│   │   ├── model.joblib            # Full sklearn Pipeline
│   │   ├── feature_schema.json     # API input contract (types, ranges, nullability)
│   │   ├── metrics.json            # CV performance metrics
│   │   ├── thresholds.json         # Decision bands: approve / review / reject
│   │   └── model_card.md           # Model documentation
│   └── experiments/                # Exploratory CV runs (not production)
│       ├── cv_results.json
│       ├── xgb_conservative.json
│       └── xgb_conservative_config.json
│
├── docs/
│   ├── feature_importance_v1.png   # Horizontal bar chart (top 15 features by gain)
│   └── feature_importance_v1.csv   # Full ranked table
│
├── notebooks/
│   └── EDA.ipynb                   # Exploratory data analysis
│
├── data/                           # Not committed — see section 4
│   ├── raw/
│   └── processed/
│
├── deployment/                     # Pending — see Next steps
├── test/                           # Pending — see Next steps
├── .gitignore
└── requirements.txt
```

---

## 2. Architecture overview

The core design decision is the **single serialized sklearn Pipeline**. Cleaning and feature engineering are implemented as stateful sklearn transformers (`CleaningTransformer`, `FeatureEngineeringTransformer`) that learn their parameters on the training set and apply them at inference time — the same object, serialized in `model.joblib`.

```
Raw input (10 columns)
        |
        v
CleaningTransformer          ← caps sentinels, winsorizes, imputes missing values
        |                       parameters learned once from training data
        v
FeatureEngineeringTransformer ← creates 7 derived features
        |                        parameters learned once from training data
        v
XGBClassifier                ← 193 trees, baseline config
        |
        v
probability_of_default (float 0–1)
        |
        v
Decision thresholds          ← approve / review / reject
```

This eliminates **training-serving skew**: the API receives raw inputs and the pipeline handles everything internally. No pre-processing code is duplicated outside the pipeline.

---

## 3. Prerequisites and setup

**Python:** 3.11+

```bash
git clone https://github.com/AnthonyFl22/credit-scoring-mlops.git
cd credit-scoring-mlops
pip install -r requirements.txt
```

Dependencies:

| Package | Version | Purpose |
|---|---|---|
| pandas | 2.3.2 | Data manipulation |
| numpy | 1.26.2 | Numerical operations |
| scikit-learn | 1.7.1 | Pipeline, transformers, CV |
| xgboost | 3.2.0 | Gradient boosted trees |
| joblib | 1.5.1 | Pipeline serialization |
| matplotlib | 3.10.5 | Feature importance plots |
| fastapi | 0.104.1 | Inference API framework |
| uvicorn | 0.24.0 | ASGI server |

---

## 4. Data

The raw data is **not committed** to the repository (covered by `.gitignore`). Download the files from [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place them as follows:

```
data/
└── raw/
    ├── cs-training.csv   ← 150,000 labeled records
    └── cs-test.csv       ← 101,503 unlabeled records (optional)
```

**Known data quality issues** (handled automatically by the pipeline):

| Issue | Columns affected | Fix |
|---|---|---|
| Impossible age values (age = 0) | `age` | Drop rows where `age < 18` |
| Sentinel codes 96 / 98 in delinquency counters | `NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfTimes90DaysLate` | Cap to the largest real value in each column |
| Heavy right tails | `RevolvingUtilizationOfUnsecuredLines`, `MonthlyIncome`, `DebtRatio` | Winsorize at the 99.5th percentile |
| Missing values (~20%) | `MonthlyIncome` | Impute with training median + add `MonthlyIncome_was_missing` flag |
| Missing values (~2.6%) | `NumberOfDependents` | Impute with 0 |

---

## 5. Step-by-step workflow

### 5.1 Process raw data

Fits cleaning parameters on the training set only, then applies them to all splits. Outputs processed CSVs and a serialized `cleaning_params.json`.

```bash
python scripts/clean_data.py
```

Output:
```
data/processed/
├── cs-training-processed.csv
├── cs-test-processed.csv       (if cs-test.csv was present)
└── cleaning_params.json
```

### 5.2 Feature engineering

Fits feature parameters on the training set only, then applies them. Creates 7 derived features.

```bash
python scripts/build_features.py
```

Output:
```
data/processed/
├── cs-training-featured.csv
├── cs-test-featured.csv        (if cs-test-processed.csv was present)
└── feature_params.json
```

**Derived features:**

| Feature | Description |
|---|---|
| `total_past_due` | Sum of all three delinquency counters |
| `any_past_due` | Binary flag: `total_past_due > 0` |
| `log_monthly_income` | `log1p(MonthlyIncome)` — compresses right skew |
| `income_per_dependent` | `MonthlyIncome / (NumberOfDependents + 1)` |
| `open_lines_per_age_yr` | `NumberOfOpenCreditLinesAndLoans / (age - 17)` — open lines per year of eligible credit history |
| `age_bin` | Ordinal bucket: 0 = young (18–35), 1 = middle (36–55), 2 = senior (56+) |
| `MonthlyIncome_was_missing` | Flag created during cleaning (1 if income was originally null) |

### 5.3 Model selection via cross-validation

Runs stratified 5-fold CV for one or more XGBoost config presets and prints a comparison table. Uses per-fold early stopping and computes all metrics including the F1-maximizing threshold.

```bash
# Compare all three presets
python training/train_model.py

# Single config
python training/train_model.py --configs baseline
```

Available presets: `conservative`, `baseline`, `deeper`.

Output saved to `models/experiments/`:
```
models/experiments/
├── cv_results.json             ← metrics for all evaluated configs
├── xgb_<best_config>.json      ← best model weights
└── xgb_<best_config>_config.json
```

### 5.4 Build the production pipeline

Retrains on the full training set using the best config and `n_estimators` from CV, wraps everything in a sklearn Pipeline, and saves all production artifacts.

```bash
python training/train_final_pipeline.py
```

Output:
```
models/credit_scoring/v1/
├── model.joblib            ← full sklearn Pipeline (use this in the API)
├── feature_schema.json     ← API input contract
├── metrics.json            ← CV performance metrics
└── thresholds.json         ← provisional decision bands
```

### 5.5 Inspect feature importances

Loads the serialized pipeline and extracts XGBoost feature importances by **gain** (total reduction in training loss per feature across all trees).

```bash
python scripts/feature_importance.py

# Custom options
python scripts/feature_importance.py --top-n 20 --out-dir docs --version v1
```

Output: `docs/feature_importance_v1.png` and `docs/feature_importance_v1.csv`.

### 5.6 Run the inference API

```bash
uvicorn api.main:app --reload
```

The API loads `model.joblib`, `thresholds.json`, and `feature_schema.json` at startup. Interactive docs available at `http://127.0.0.1:8000/docs`.

---

## 6. Model details

**Algorithm:** XGBoost gradient boosted trees — `baseline` configuration.

**Class imbalance:** handled via `scale_pos_weight = n_negatives / n_positives ≈ 13.9`. The training set has ~93.3% non-default / ~6.7% default.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| `max_depth` | 4 |
| `learning_rate` | 0.05 |
| `n_estimators` | 193 (from CV early stopping) |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 10 |
| `reg_alpha` (L1) | 0.1 |
| `reg_lambda` (L2) | 1.0 |
| `scale_pos_weight` | ~13.9 |

**Cross-validation performance** (stratified 5-fold):

| Metric | Value |
|---|---|
| ROC-AUC | 0.8661 (±0.0039) |
| Gini coefficient | 0.7322 |
| KS statistic | 0.579 |
| Average precision | 0.4031 |
| Precision | 0.4053 |
| Recall | 0.5033 |
| F1 | 0.4484 |

> Precision, recall, and F1 are computed at the **F1-maximizing threshold (~0.40)**, not at 0.5 default. ROC-AUC and Gini are threshold-independent and are the primary quality metrics. See `models/credit_scoring/v1/model_card.md` for full documentation including limitations.

**Decision thresholds** (provisional — from `thresholds.json`):

| Probability of default | Decision |
|---|---|
| < 0.15 | `approved` |
| 0.15 – 0.40 | `review` |
| >= 0.40 | `rejected` |

These thresholds have not been optimized for any specific cost function and should be tuned before production use.

---

## 7. API reference

### `GET /`

Returns a welcome message. Used to confirm the service is reachable.

```json
{ "message": "Welcome to the Credit Scoring API" }
```

### `GET /health`

Returns the load status of the model and thresholds.

```json
{ "status": "ok", "model_loaded": true, "thresholds_loaded": true }
```

### `POST /predict`

Runs the full pipeline and returns a credit decision.

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | float >= 0 | Yes | Total balance / credit limit ratio |
| `age` | int >= 18 | Yes | Age of the applicant |
| `NumberOfTime30-59DaysPastDueNotWorse` | int >= 0 | Yes | Times 30-59 days past due (last 2 years) |
| `DebtRatio` | float >= 0 | Yes | Monthly debt payments / gross income |
| `MonthlyIncome` | float >= 0 | No | Monthly gross income in USD (null → imputed with training median) |
| `NumberOfOpenCreditLinesAndLoans` | int >= 0 | Yes | Open loans and lines of credit |
| `NumberOfTimes90DaysLate` | int >= 0 | Yes | Times 90+ days past due |
| `NumberRealEstateLoansOrLines` | int >= 0 | Yes | Mortgage and real estate loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | int >= 0 | Yes | Times 60-89 days past due (last 2 years) |
| `NumberOfDependents` | int >= 0 | No | Dependents excluding borrower (null → imputed with 0) |

**Example request:**

```json
{
  "RevolvingUtilizationOfUnsecuredLines": 0.35,
  "age": 45,
  "NumberOfTime30-59DaysPastDueNotWorse": 0,
  "DebtRatio": 0.3,
  "MonthlyIncome": 6000,
  "NumberOfOpenCreditLinesAndLoans": 5,
  "NumberOfTimes90DaysLate": 0,
  "NumberRealEstateLoansOrLines": 1,
  "NumberOfTime60-89DaysPastDueNotWorse": 0,
  "NumberOfDependents": 2
}
```

**Response:**

```json
{
  "probability_of_default": 0.043,
  "decision": "approved",
  "model_version": "v1"
}
```

| Field | Type | Description |
|---|---|---|
| `probability_of_default` | float [0, 1] | Predicted probability of serious delinquency |
| `decision` | string | `approved`, `review`, or `rejected` |
| `model_version` | string | Model version used for the prediction |

---

## 8. Model artifacts

All production artifacts live in `models/credit_scoring/v1/`.

| File | Description |
|---|---|
| `model.joblib` | Serialized sklearn Pipeline. Load with `joblib.load()`. Contains fitted cleaning params, feature params, and XGBoost weights. |
| `feature_schema.json` | API input contract: column names, dtypes, min/max ranges, nullability. Source of truth for what the model accepts. |
| `metrics.json` | CV performance metrics and confusion matrix. Includes training date and config name. |
| `thresholds.json` | Decision bands with notes on how to tune them. |
| `model_card.md` | Full model documentation: intended use, training data, algorithm, metrics, limitations, and fairness considerations. |

---

## 9. Next steps

### Testing (`test/`)

The `test/` directory is currently empty. Planned test coverage:

- **Unit tests** for `src/data_cleaning.py` and `src/feature_engineering.py` — verify that fit/apply produce consistent outputs and that parameters are learned correctly
- **Pipeline integration test** — verify that `model.joblib` accepts valid inputs and returns a probability in [0, 1]
- **API tests** — verify all three endpoints, valid requests, missing optional fields, and invalid inputs (age < 18, negative values, unknown fields)
- **Schema contract test** — verify that `feature_schema.json` matches the columns expected by the pipeline

### Deployment (`deployment/`)

The `deployment/` directory is currently empty. Planned work:

- **Dockerize the API** — write a `Dockerfile` that installs dependencies and runs `uvicorn api.main:app`
- **Cloud deployment** — deploy the container to a cloud provider (e.g. AWS ECS, GCP Cloud Run, or Azure Container Apps)
- **CI/CD pipeline** — automate testing and deployment on push to `master`
- **Threshold optimization** — replace provisional thresholds with cost-based optimization using a business-defined cost matrix (cost of false approval vs. cost of false rejection)
- **Model monitoring** — track prediction distribution drift and trigger retraining when performance degrades
