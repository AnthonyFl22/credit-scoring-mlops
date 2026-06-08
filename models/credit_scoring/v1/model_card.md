# Model Card — Credit Scoring v1

## Model Name and Version
- **Name:** credit_scoring
- **Version:** v1
- **Training date:** 2026-06-08

## Intended Use
Binary classifier for credit risk scoring. Outputs a default probability (probability of serious delinquency — 90+ days past due within 2 years) to support loan application decisions. Intended for use by credit analysts and automated decisioning systems.

## Training Data
- **Source:** Give Me Some Credit (Kaggle competition dataset)
- **Training rows:** 149,999 (after removing records with age < 18)
- **Class balance:** ~93.3% non-default / ~6.7% default
- **Input features:** 10 raw features (cleaning and feature engineering applied internally by the pipeline)

## Algorithm
XGBoost gradient boosted trees (`baseline` configuration).

| Hyperparameter | Value |
|---|---|
| `max_depth` | 4 |
| `learning_rate` | 0.05 |
| `n_estimators` | 193 (from CV early stopping) |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 10 |
| `reg_alpha` (L1) | 0.1 |
| `reg_lambda` (L2) | 1.0 |
| `scale_pos_weight` | ~13.9 (inverse class ratio) |

## Metrics
Evaluated via stratified 5-fold cross-validation on the full training set.

| Metric | Value | CV Std |
|---|---|---|
| ROC-AUC | 0.8661 | ±0.0039 |
| Gini | 0.7322 | — |
| KS Statistic | 0.579 | — |
| Avg Precision | 0.4031 | — |
| Precision | 0.4053 | — |
| Recall | 0.5033 | — |
| F1 | 0.4484 | — |

### Confusion Matrix 

> **Note:** Precision, recall, and the confusion matrix below are computed at the **F1-maximizing threshold** (~0.40), not at any business-tuned cutoff. All these figures change with the threshold. The ROC-AUC (0.8661) is threshold-independent and is the primary measure of model quality. See `thresholds.json` and the Decision Thresholds section for context.

| | Predicted Good | Predicted Bad |
|---|---|---|
| **Actual Good** | 132,530 | 7,443 |
| **Actual Bad** | 4,980 | 5,046 |

- False positive rate (good applicant rejected): 5.32%
- False negative rate (bad applicant approved): 49.67% — **high because F1-maximization treats false positives and false negatives as equally costly. In credit scoring, false negatives are typically 5-10x more expensive, so the operating threshold should be lower than the F1-optimal point. See thresholds.json for the provisional business-tuned cutoffs.**

## Limitations
- Trained on historical US consumer credit data (circa 2011). Applicant population and macroeconomic conditions may have shifted.
- Model is **not probability-calibrated**. Scores are suitable for ranking but should not be interpreted as literal default probabilities without calibration (e.g. Platt scaling or isotonic regression).
- **No fairness audit performed.** Impact on protected attributes (age groups, gender, race) has not been assessed.
- `MonthlyIncome` is missing for ~20% of training records and is imputed with the training median. Performance may degrade for applicant segments with systematically different income profiles.
- Decision thresholds in `thresholds.json` are provisional and have not been optimised for any specific cost function.

## Decision Thresholds
See `thresholds.json` for provisional decision bands (approve / review / reject).
Thresholds should be tuned via cost-based threshold optimization before production use.
