"""Model training and evaluation for the Give Me Some Credit dataset.

Evaluation metrics
------------------
* ``roc_auc``       — primary metric 
* ``gini``          — 2 × AUC − 1 (standard credit scorecard metric)
* ``avg_precision`` — area under precision-recall curve (robust to class imbalance)
* ``ks``            — Kolmogorov-Smirnov separation between default / non-default distributions
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "SeriousDlqin2yrs"


@dataclass
class ModelConfig:
    """XGBoost hyperparameter bundle."""

    name: str
    max_depth: int = 4
    learning_rate: float = 0.05
    n_estimators: int = 1000        # upper bound; early stopping controls actual count
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 10      # higher = more conservative splits
    reg_alpha: float = 0.1          # L1
    reg_lambda: float = 1.0         # L2
    early_stopping_rounds: int = 50

    def to_xgb_kwargs(self, scale_pos_weight: float) -> dict:
        return {
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": 42,
            "n_jobs": -1,
        }

    def to_json(self, path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(**json.load(fh))


# Three presets that trade off between regularization and expressiveness.
# conservative: safest against overfitting; good baseline on noisy data.
# baseline:     balanced starting point.
# deeper:       more expressive trees with a lower learning rate to compensate.

CONFIGS: Dict[str, ModelConfig] = {
    "conservative": ModelConfig(
        name="conservative",
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=20,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
    ),
    "baseline": ModelConfig(
        name="baseline",
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
    ),
    "deeper": ModelConfig(
        name="deeper",
        max_depth=6,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.0,
    ),
}


@dataclass
class FoldMetrics:
    roc_auc: float
    gini: float
    avg_precision: float
    ks: float
    precision: float
    recall: float
    f1: float
    threshold: float    # decision threshold used for precision/recall/confusion matrix
    tn: int
    fp: int
    fn: int
    tp: int
    best_iteration: int


@dataclass
class CVResult:
    config_name: str
    fold_metrics: List[FoldMetrics]
    # Mean best_iteration across folds — used as n_estimators for the final refit.
    best_n_estimators: int

    def mean_auc(self) -> float:
        return float(np.mean([f.roc_auc for f in self.fold_metrics]))

    def means(self) -> Dict[str, float]:
        metrics = ("roc_auc", "gini", "avg_precision", "ks", "precision", "recall", "f1")
        return {m: float(np.mean([getattr(f, m) for f in self.fold_metrics])) for m in metrics}

    def stds(self) -> Dict[str, float]:
        metrics = ("roc_auc", "gini", "avg_precision", "ks", "precision", "recall", "f1")
        return {m: float(np.std([getattr(f, m) for f in self.fold_metrics])) for m in metrics}

    def confusion_matrix_sum(self) -> Dict[str, int]:
        """Aggregate confusion matrix summed across all folds."""
        return {
            "tn": sum(f.tn for f in self.fold_metrics),
            "fp": sum(f.fp for f in self.fold_metrics),
            "fn": sum(f.fn for f in self.fold_metrics),
            "tp": sum(f.tp for f in self.fold_metrics),
        }


def _ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Max separation between cumulative default / non-default score distributions."""
    order = np.argsort(y_prob)[::-1]
    y_sorted = y_true[order]
    n_pos = max(y_sorted.sum(), 1)
    n_neg = max((1 - y_sorted).sum(), 1)
    cum_pos = np.cumsum(y_sorted) / n_pos
    cum_neg = np.cumsum(1 - y_sorted) / n_neg
    return float(np.max(np.abs(cum_pos - cum_neg)))


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold on the precision-recall curve that maximises F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Arrays have length n+1; the last element has no matching threshold.
    f1 = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    return float(thresholds[np.argmax(f1)])


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, best_iter: int) -> FoldMetrics:
    
    auc = roc_auc_score(y_true, y_prob)
    threshold = _best_f1_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return FoldMetrics(
        roc_auc=auc,
        gini=2 * auc - 1,
        avg_precision=average_precision_score(y_true, y_prob),
        ks=_ks_statistic(y_true, y_prob),
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=threshold,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        best_iteration=best_iter,
    )


def cross_validate(X: pd.DataFrame,y: pd.Series,config: ModelConfig,n_folds: int = 5,seed: int = 42,) -> CVResult:
    """Stratified k-fold CV with per-fold early stopping."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    fold_metrics: List[FoldMetrics] = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        clf = xgb.XGBClassifier(
            **config.to_xgb_kwargs(scale_pos_weight),
            early_stopping_rounds=config.early_stopping_rounds,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_prob = clf.predict_proba(X_val)[:, 1]
        fold_metrics.append(_compute_metrics(y_val.values, y_prob, clf.best_iteration))

    best_n = int(np.mean([f.best_iteration for f in fold_metrics]))
    return CVResult(config_name=config.name, fold_metrics=fold_metrics, best_n_estimators=best_n)


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    config: ModelConfig,
    n_estimators: int,
) -> xgb.XGBClassifier:
    """Refit on the full dataset using the early-stopped tree count from CV."""
    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    kwargs = config.to_xgb_kwargs(scale_pos_weight)
    kwargs["n_estimators"] = n_estimators
    clf = xgb.XGBClassifier(**kwargs)
    clf.fit(X, y, verbose=False)
    return clf


def features_and_target(df: pd.DataFrame):
    """Split a featured dataframe into (X, y)."""
    return df.drop(columns=[TARGET]), df[TARGET]
