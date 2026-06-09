"""Centralized artifact paths for the credit scoring API."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_V1 = REPO_ROOT / 'models' / 'credit_scoring' / 'v1'
MODEL_PATH = _V1 / "model.joblib"
THRESHOLDS_PATH = _V1 / "thresholds.json"
FEATURE_SCHEMA_PATH = _V1 / "feature_schema.json"
