# Credit Scoring MLOps

**Last updated:** 2026-06-17

Binary credit risk classifier that predicts the probability a borrower will experience serious delinquency (90+ days past due) within two years. Built as a full MLOps project — from raw data to a production-ready inference API deployed on AWS.

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
   - 5.6 [Run the API locally](#56-run-the-api-locally)
   - 5.7 [Run with Docker](#57-run-with-docker)
6. [Model details](#6-model-details)
7. [API reference](#7-api-reference)
8. [Model artifacts](#8-model-artifacts)
9. [Deployment](#9-deployment)
10. [Next steps](#10-next-steps)

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
├── test/
│   └── test_api.py                 # 29 pytest tests covering endpoints and validation
│
├── models/
│   ├── credit_scoring/v1/          # Production artifact (versioned)
│   │   ├── model.joblib            # Full sklearn Pipeline
│   │   ├── feature_schema.json     # API input contract (types, ranges, nullability)
│   │   ├── metrics.json            # CV performance metrics
│   │   ├── thresholds.json         # Decision bands: approve / review / reject
│   │   └── model_card.md           # Model documentation
│   └── experiments/                # Exploratory CV runs (not production)
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
├── Dockerfile                      # Container image definition
├── pytest.ini                      # pytest configuration
├── requirements.txt
└── .gitignore
```

---

## 2. Architecture overview

The core design decision is the **single serialized sklearn Pipeline**. Cleaning and feature engineering are implemented as stateful sklearn transformers that learn their parameters on the training set and apply them at inference time — the same object, serialized in `model.joblib`.

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

This eliminates **training-serving skew**: the API receives raw inputs and the pipeline handles everything internally.

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
| pytest | 9.1.0 | Testing framework |
| httpx | 0.27.2 | HTTP client for tests |

---

## 4. Data

The raw data is **not committed** to the repository. Download from [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place as:

```
data/
└── raw/
    ├── cs-training.csv   ← 150,000 labeled records
    └── cs-test.csv       ← 101,503 unlabeled records (optional)
```

**Known data quality issues handled automatically by the pipeline:**

| Issue | Columns affected | Fix |
|---|---|---|
| Impossible age values (age = 0) | `age` | Drop rows where `age < 18` |
| Sentinel codes 96 / 98 | Past-due counters | Cap to largest real value per column |
| Heavy right tails | `RevolvingUtilizationOfUnsecuredLines`, `MonthlyIncome`, `DebtRatio` | Winsorize at 99.5th percentile |
| Missing values (~20%) | `MonthlyIncome` | Impute with training median + missingness flag |
| Missing values (~2.6%) | `NumberOfDependents` | Impute with 0 |

---

## 5. Step-by-step workflow

### 5.1 Process raw data

```bash
python scripts/clean_data.py
```

Output: `data/processed/cs-training-processed.csv`, `cleaning_params.json`

### 5.2 Feature engineering

```bash
python scripts/build_features.py
```

Output: `data/processed/cs-training-featured.csv`, `feature_params.json`

**Derived features:**

| Feature | Description |
|---|---|
| `total_past_due` | Sum of all three delinquency counters |
| `any_past_due` | Binary flag: `total_past_due > 0` |
| `log_monthly_income` | `log1p(MonthlyIncome)` — compresses right skew |
| `income_per_dependent` | `MonthlyIncome / (NumberOfDependents + 1)` |
| `open_lines_per_age_yr` | Open credit lines per year of eligible credit history |
| `age_bin` | Ordinal bucket: 0 = young (18–35), 1 = middle (36–55), 2 = senior (56+) |
| `MonthlyIncome_was_missing` | Flag: 1 if income was originally null |

### 5.3 Model selection via cross-validation

```bash
python training/train_model.py              # compare all presets
python training/train_model.py --configs baseline  # single config
```

Output saved to `models/experiments/`.

### 5.4 Build the production pipeline

```bash
python training/train_final_pipeline.py
```

Output: `models/credit_scoring/v1/` — model.joblib, feature_schema.json, metrics.json, thresholds.json.

### 5.5 Inspect feature importances

```bash
python scripts/feature_importance.py
```

Output: `docs/feature_importance_v1.png` and `docs/feature_importance_v1.csv`.

### 5.6 Run the API locally

```bash
uvicorn api.main:app --reload
```

Interactive docs at `http://127.0.0.1:8000/docs`.

### 5.7 Run with Docker

```bash
# Build the image
docker build -t credit-scoring-api .

# Run the container
docker run -p 8080:8080 credit-scoring-api
```

Interactive docs at `http://localhost:8080/docs`.

---

## 6. Model details

**Algorithm:** XGBoost gradient boosted trees — `baseline` configuration.

**Class imbalance:** handled via `scale_pos_weight ≈ 13.9` (inverse class ratio).

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
| F1 | 0.4484 |

**Decision thresholds** (provisional):

| Probability of default | Decision |
|---|---|
| < 0.15 | `approved` |
| 0.15 – 0.40 | `review` |
| >= 0.40 | `rejected` |

---

## 7. API reference

### `GET /`
Returns a welcome message.

### `GET /health`
Returns model and threshold load status.

### `POST /predict`

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | float >= 0 | Yes | Total balance / credit limit ratio |
| `age` | int >= 18 | Yes | Age of the applicant |
| `NumberOfTime30-59DaysPastDueNotWorse` | int >= 0 | Yes | Times 30-59 days past due (last 2 years) |
| `DebtRatio` | float >= 0 | Yes | Monthly debt payments / gross income |
| `MonthlyIncome` | float >= 0 | No | Monthly gross income in USD (null → imputed) |
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

---

## 8. Model artifacts

All production artifacts live in `models/credit_scoring/v1/`.

| File | Description |
|---|---|
| `model.joblib` | Serialized sklearn Pipeline. Load with `joblib.load()`. |
| `feature_schema.json` | API input contract: column names, dtypes, ranges, nullability. |
| `metrics.json` | CV performance metrics and confusion matrix. |
| `thresholds.json` | Decision bands with notes on tuning. |
| `model_card.md` | Full model documentation including limitations and fairness considerations. |

---

## 9. Deployment

The API is containerized with Docker and deployed on **AWS ECS Express Mode** using an image stored in **Amazon ECR**.

**Infrastructure:**

| Component | Service |
|---|---|
| Container registry | Amazon ECR |
| Container orchestration | AWS ECS Express Mode (Fargate) |
| Compute | 0.5 vCPU / 1 GB RAM |
| HTTPS | Included automatically by ECS Express Mode |

**Deploy a new version:**

```bash
# 1. Rebuild the image
docker build -t credit-scoring-api .

# 2. Authenticate Docker with ECR
aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com

# 3. Tag and push
docker tag credit-scoring-api:latest <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/credit-scoring-api:latest
docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/credit-scoring-api:latest

# 4. Update the service in ECS
aws ecs update-service --cluster default --service credit-scoring-api --force-new-deployment --region <REGION>
```

**Run tests before deploying:**

```bash
python -m pytest test/ -v
```

---

## 10. Next steps

- **Threshold optimization** — replace provisional thresholds with cost-based optimization using a business-defined cost matrix
- **CI/CD pipeline** — automate testing and deployment on push to `master` using GitHub Actions
- **Model monitoring** — track prediction distribution drift and trigger retraining when performance degrades
