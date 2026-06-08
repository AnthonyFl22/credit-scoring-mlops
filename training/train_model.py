"""Train and evaluate XGBoost models on the featured credit-scoring data.

Usage (from the repo root):

    python scripts/train_model.py                              # compare all presets
    python scripts/train_model.py --configs baseline           # single config
    python scripts/train_model.py --configs conservative baseline deeper

Reads ``data/processed/cs-training-featured.csv``, runs stratified k-fold CV
for each requested config, prints a comparison table, and saves the best model
(by mean CV AUC) plus all CV results to ``models/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model import CONFIGS, CVResult, cross_validate, features_and_target, train_final_model 


def _print_table(results: list[CVResult]) -> None:
    header = (
        f"{'Config':<15} {'AUC':>8} {'±':>6} {'Gini':>8} "
        f"{'AvgPrec':>9} {'KS':>8} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Trees':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        m = r.means()
        s = r.stds()
        print(
            f"{r.config_name:<15} "
            f"{m['roc_auc']:>8.4f} "
            f"{s['roc_auc']:>6.4f} "
            f"{m['gini']:>8.4f} "
            f"{m['avg_precision']:>9.4f} "
            f"{m['ks']:>8.4f} "
            f"{m['precision']:>7.4f} "
            f"{m['recall']:>7.4f} "
            f"{m['f1']:>7.4f} "
            f"{r.best_n_estimators:>7d}"
        )
    print()


def _print_confusion_matrix(result: CVResult) -> None:
    cm = result.confusion_matrix_sum()
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    total = tn + fp + fn + tp
    avg_threshold = sum(f.threshold for f in result.fold_metrics) / len(result.fold_metrics)

    print(f"Confusion matrix — '{result.config_name}' (aggregated across folds, threshold ≈ {avg_threshold:.3f})")
    print(f"                  {'Predicted 0':>14}  {'Predicted 1':>14}")
    print(f"  Actual 0 (good)  {tn:>14,}  {fp:>14,}")
    print(f"  Actual 1 (bad)   {fn:>14,}  {tp:>14,}")
    print(f"  Total: {total:,}  |  False positive rate: {fp/(fp+tn):.2%}  |  False negative rate: {fn/(fn+tp):.2%}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir", type=Path, default=REPO_ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--models-dir", type=Path, default=REPO_ROOT / "models" / "experiments"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS.keys()),
        default=list(CONFIGS.keys()),
        metavar="CONFIG",
        help=f"Presets to evaluate. Choices: {list(CONFIGS.keys())} (default: all).",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df = pd.read_csv(args.processed_dir / "cs-training-featured.csv", index_col=0)
    X, y = features_and_target(train_df)
    print(f"  {len(X):,} rows | {X.shape[1]} features | {y.mean():.2%} default rate\n")

    results: list[CVResult] = []
    for name in args.configs:
        config = CONFIGS[name]
        print(f"[{name}] Running {args.n_folds}-fold CV...")
        result = cross_validate(X, y, config, n_folds=args.n_folds, seed=args.seed)
        results.append(result)
        print(f"  AUC {result.mean_auc():.4f} | best trees {result.best_n_estimators}")

    print()
    _print_table(results)

    best = max(results, key=lambda r: r.mean_auc())
    _print_confusion_matrix(best)
    print(f"Best config: '{best.config_name}'  (AUC {best.mean_auc():.4f})")
    print(f"Refitting on full training set with {best.best_n_estimators} trees...")

    final_model = train_final_model(X, y, CONFIGS[best.config_name], best.best_n_estimators)

    model_path = args.models_dir / f"xgb_{best.config_name}.json"
    final_model.save_model(model_path)
    print(f"Model saved       -> {model_path}")

    config_path = args.models_dir / f"xgb_{best.config_name}_config.json"
    CONFIGS[best.config_name].to_json(config_path)
    print(f"Config saved      -> {config_path}")

    cv_path = args.models_dir / "cv_results.json"
    summary = []
    for r in results:
        summary.append({
            "config": r.config_name,
            "best_n_estimators": r.best_n_estimators,
            **r.means(),
            "std_roc_auc": r.stds()["roc_auc"],
            "confusion_matrix": r.confusion_matrix_sum(),
        })
    with open(cv_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"CV results saved  -> {cv_path}")


if __name__ == "__main__":
    main()
