# Paths for model.joblib and thresholds.json 
from pathlib import Path

# Get the absolute path to the current file

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = REPO_ROOT / 'models' / 'credit_scoring' / 'v1' / "model.joblib"
THRESHOLDS_PATH = REPO_ROOT / 'models' / 'credit_scoring' / 'v1' / "thresholds.json"
