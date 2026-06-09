"""FastAPI application for the credit scoring v1 model.

Loads the sklearn Pipeline, decision thresholds, and feature column order at
startup. Exposes three endpoints: GET /, GET /health, POST /predict.
"""
import json
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from api.schemas import CreditScoringInput, CreditScoringOutput
from api.config import MODEL_PATH, THRESHOLDS_PATH, FEATURE_SCHEMA_PATH


# Load the model and thresholds at startup
model = joblib.load(MODEL_PATH)
with open(THRESHOLDS_PATH, 'r') as f:
    thresholds = json.load(f)

with open(FEATURE_SCHEMA_PATH, 'r') as f:
    _schema = json.load(f)
INPUT_COLUMNS =  [feat["name"] for feat in _schema["features"]]


app = FastAPI(
    title="Credit Scoring API",
    description="API for credit scoring predictions using a pre-trained model.",
    version="v1"
)

@app.get("/")
def root() -> dict:
    return {"message": "Welcome to the Credit Scoring API"}

@app.get("/health")
def health_check() -> dict:
    """Health check endpoint to verify that the API is running."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "thresholds_loaded": thresholds is not None
    }

@app.post("/predict", response_model=CreditScoringOutput)
def predict_credit_score(payload: CreditScoringInput) -> CreditScoringOutput:
    """Endpoint to make credit scoring predictions."""
    # Convert input data to a DataFrame
    input_data = payload.model_dump(by_alias=True)
    input_df = pd.DataFrame([input_data])

    input_df = input_df[INPUT_COLUMNS]

    # Predict probability of default
    probability_of_default = model.predict_proba(input_df)[0][1]

    # Apply decision thresholds
    decision = get_decision(
        probability_of_default=probability_of_default,
        thresholds=thresholds,
        )

    return CreditScoringOutput(
        probability_of_default=probability_of_default,
        decision=decision,
        model_version="v1",
        )


def get_decision(probability_of_default: float, thresholds: dict) -> str:
    """Determine the decision based on the predicted probability of default and thresholds."""
    if probability_of_default < thresholds["approve_max"]:
        return "approved"
    elif probability_of_default < thresholds["review_max"]:
        return "review"
    else:
        return "rejected"

