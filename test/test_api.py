"""Tests for the credit scoring API.

Run from the repo root:
    python -m pytest test/ -v
"""

from starlette.testclient import TestClient

from api.main import app, get_decision

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "RevolvingUtilizationOfUnsecuredLines": 0.35,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.30,
    "MonthlyIncome": 6000.0,
    "NumberOfOpenCreditLinesAndLoans": 5,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2,
}

THRESHOLDS = {"approve_max": 0.15, "review_max": 0.40}


# ---------------------------------------------------------------------------
# Unit tests: get_decision()
# ---------------------------------------------------------------------------

class TestGetDecision:
    def test_approved(self):
        assert get_decision(0.10, THRESHOLDS) == "approved"

    def test_review(self):
        assert get_decision(0.25, THRESHOLDS) == "review"

    def test_rejected(self):
        assert get_decision(0.50, THRESHOLDS) == "rejected"

    def test_boundary_approve_max_falls_into_review(self):
        # 0.15 is NOT < 0.15, so it goes to review
        assert get_decision(0.15, THRESHOLDS) == "review"

    def test_boundary_review_max_falls_into_rejected(self):
        # 0.40 is NOT < 0.40, so it goes to rejected
        assert get_decision(0.40, THRESHOLDS) == "rejected"


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestRoot:
    def test_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_welcome_message(self):
        response = client.get("/")
        assert response.json() == {"message": "Welcome to the Credit Scoring API"}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_status_ok(self):
        assert client.get("/health").json()["status"] == "ok"

    def test_model_loaded(self):
        assert client.get("/health").json()["model_loaded"] is True

    def test_thresholds_loaded(self):
        assert client.get("/health").json()["thresholds_loaded"] is True


# ---------------------------------------------------------------------------
# POST /predict — happy paths
# ---------------------------------------------------------------------------

class TestPredictValid:
    def test_returns_200(self):
        assert client.post("/predict", json=VALID_PAYLOAD).status_code == 200

    def test_response_has_required_fields(self):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "probability_of_default" in body
        assert "decision" in body
        assert "model_version" in body

    def test_probability_in_valid_range(self):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert 0.0 <= body["probability_of_default"] <= 1.0

    def test_decision_is_valid_literal(self):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert body["decision"] in ("approved", "review", "rejected")

    def test_model_version_is_v1(self):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert body["model_version"] == "v1"

    def test_optional_fields_sent_as_null(self):
        """MonthlyIncome and NumberOfDependents can be sent as null."""
        payload = {**VALID_PAYLOAD, "MonthlyIncome": None, "NumberOfDependents": None}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert 0.0 <= response.json()["probability_of_default"] <= 1.0

    def test_optional_fields_omitted(self):
        """MonthlyIncome and NumberOfDependents can be omitted entirely."""
        payload = {
            k: v for k, v in VALID_PAYLOAD.items()
            if k not in ("MonthlyIncome", "NumberOfDependents")
        }
        assert client.post("/predict", json=payload).status_code == 200

    def test_deterministic_output(self):
        """Same input must always produce the same prediction."""
        r1 = client.post("/predict", json=VALID_PAYLOAD).json()
        r2 = client.post("/predict", json=VALID_PAYLOAD).json()
        assert r1["probability_of_default"] == r2["probability_of_default"]
        assert r1["decision"] == r2["decision"]


# ---------------------------------------------------------------------------
# POST /predict — input validation (expect 422)
# ---------------------------------------------------------------------------

class TestPredictValidation:
    def test_age_below_minimum(self):
        """age must be >= 18."""
        response = client.post("/predict", json={**VALID_PAYLOAD, "age": 17})
        assert response.status_code == 422

    def test_age_zero(self):
        response = client.post("/predict", json={**VALID_PAYLOAD, "age": 0})
        assert response.status_code == 422

    def test_missing_required_field(self):
        """Omitting a required field must return 422."""
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_extra_field_rejected(self):
        """extra='forbid' — unknown fields must be rejected."""
        payload = {**VALID_PAYLOAD, "credit_score": 750}
        assert client.post("/predict", json=payload).status_code == 422

    def test_negative_revolving_utilization(self):
        payload = {**VALID_PAYLOAD, "RevolvingUtilizationOfUnsecuredLines": -0.1}
        assert client.post("/predict", json=payload).status_code == 422

    def test_negative_debt_ratio(self):
        payload = {**VALID_PAYLOAD, "DebtRatio": -1.0}
        assert client.post("/predict", json=payload).status_code == 422

    def test_negative_monthly_income(self):
        payload = {**VALID_PAYLOAD, "MonthlyIncome": -500.0}
        assert client.post("/predict", json=payload).status_code == 422

    def test_negative_dependents(self):
        payload = {**VALID_PAYLOAD, "NumberOfDependents": -1}
        assert client.post("/predict", json=payload).status_code == 422

    def test_wrong_type_for_age(self):
        payload = {**VALID_PAYLOAD, "age": "forty-five"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_empty_body(self):
        assert client.post("/predict", json={}).status_code == 422
