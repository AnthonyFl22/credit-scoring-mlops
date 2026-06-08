"""Extract and save feature importances from the trained pipeline.

Usage (from the repo root):

    python scripts/feature_importance.py
    python scripts/feature_importance.py --model models/credit_scoring/v1/model.joblib
    python scripts/feature_importance.py --top-n 20 --out-dir docs

Loads the serialized sklearn Pipeline, extracts the XGBoost classifier step,
and computes feature importances by *gain* (total reduction in training loss
accumulated across every split that used each feature, across all trees).

Outputs
-------
docs/feature_importance_v1.png  — horizontal bar chart (top N features)
docs/feature_importance_v1.csv  — full ranked table (all features)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "models" / "credit_scoring" / "v1" / "model.joblib",
        help="Path to a joblib-serialized sklearn Pipeline containing an XGBoost classifier.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "docs"
    )
    parser.add_argument(
        "--top-n", type=int, default=15, help="Number of top features to show in the plot."
    )
    parser.add_argument(
        "--version", type=str, default="v1", help="Version tag used in output filenames."
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline and extract the XGBoost booster
    print(f"Loading model from {args.model} ...")
    pipeline = joblib.load(args.model)
    booster = pipeline.named_steps["classifier"].get_booster()

    # Gain importance: total reduction in training loss per feature across all splits
    scores = booster.get_score(importance_type="gain")
    importance_df = (
        pd.DataFrame({"feature": list(scores.keys()), "gain": list(scores.values())})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.index += 1
    importance_df.index.name = "rank"

    print(f"\nAll features ({len(importance_df)} total):")
    print(importance_df.to_string())

    # Save full CSV
    csv_path = args.out_dir / f"feature_importance_{args.version}.csv"
    importance_df.to_csv(csv_path)
    print(f"\nSaved: {csv_path}")

    # Plot top N
    top_n = importance_df.head(args.top_n).reset_index()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(top_n["feature"][::-1], top_n["gain"][::-1], color="steelblue", edgecolor="white")
    ax.set_xlabel("Feature Importance (Gain)", fontsize=12)
    ax.set_title(
        f"Top {args.top_n} Features by Gain — XGBoost {args.version}",
        fontsize=13,
        pad=12,
    )
    x_max = top_n["gain"].max()
    for i, v in enumerate(top_n["gain"][::-1]):
        ax.text(v + x_max * 0.01, i, f"{v:,.0f}", va="center", fontsize=9)
    ax.set_xlim(0, x_max * 1.18)
    plt.tight_layout()

    png_path = args.out_dir / f"feature_importance_{args.version}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
