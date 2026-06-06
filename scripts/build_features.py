"""Build feature-engineered datasets from the cleaned CSVs.

Usage (from the repo root):

    python scripts/build_features.py

Reads ``data/processed/cs-training-clean.csv`` (and ``cs-test-clean.csv`` if
present), fits feature parameters on the *training* split only, applies them to
both splits, and writes the results plus the fitted parameters to
``data/processed/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.feature_engineering import apply_features, fit_feature_params  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir", type=Path, default=REPO_ROOT / "data" / "processed"
    )
    parser.add_argument("--winsor-quantile", type=float, default=0.995)
    args = parser.parse_args()

    train = pd.read_csv(args.processed_dir / "cs-training-processed.csv", index_col=0)
    params = fit_feature_params(train, winsor_quantile=args.winsor_quantile)
    params.to_json(args.processed_dir / "feature_params.json")
    print(f"Fitted feature params -> {args.processed_dir / 'feature_params.json'}")

    splits = {"cs-training-processed.csv": train}
    test_path = args.processed_dir / "cs-test-processed.csv"
    if test_path.exists():
        splits["cs-test-processed.csv"] = pd.read_csv(test_path, index_col=0)

    for name, df in splits.items():
        featured = apply_features(df, params)
        out_name = name.replace("-processed.csv", "-featured.csv")
        featured.to_csv(args.processed_dir / out_name)
        new_cols = featured.shape[1] - df.shape[1]
        print(
            f"{name}: {featured.shape[1]} columns "
            f"(+{new_cols} new) -> {args.processed_dir / out_name}"
        )


if __name__ == "__main__":
    main()
