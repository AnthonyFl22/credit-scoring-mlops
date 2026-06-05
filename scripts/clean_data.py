"""Clean the raw Give Me Some Credit CSVs and write the processed datasets.

Usage (from the repo root):

    python scripts/clean_data.py

Reads ``data/raw/cs-training.csv`` and ``data/raw/cs-test.csv``, fits the
cleaning thresholds on the *training* split only, applies them to both splits,
and writes the results plus the fitted parameters to ``data/processed/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make ``src`` importable when the script is run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_cleaning import apply_cleaning, fit_cleaning_params  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "data" / "processed"
    )
    parser.add_argument("--winsor-quantile", type=float, default=0.995)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Fit on training, then apply to every split.
    train = pd.read_csv(args.raw_dir / "cs-training.csv", index_col=0)
    params = fit_cleaning_params(train, winsor_quantile=args.winsor_quantile)
    params.to_json(args.out_dir / "cleaning_params.json")
    print(f"Fitted cleaning params -> {args.out_dir / 'cleaning_params.json'}")

    splits = {"cs-training.csv": train}
    test_path = args.raw_dir / "cs-test.csv"
    if test_path.exists():
        splits["cs-test.csv"] = pd.read_csv(test_path, index_col=0)

    for name, df in splits.items():
        cleaned = apply_cleaning(df, params)
        out_name = name.replace(".csv", "-processed.csv")
        cleaned.to_csv(args.out_dir / out_name)
        print(
            f"{name}: {len(df):,} rows -> {args.out_dir / out_name} "
            f"({cleaned.shape[1]} columns)"
        )


if __name__ == "__main__":
    main()
