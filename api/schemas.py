"""Pydantic request and response schemas for the credit scoring API."""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal


class CreditScoringInput(BaseModel):
    """10 raw input features expected by the v1 model pipeline."""
    model_config = ConfigDict(
        extra='forbid',
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "RevolvingUtilizationOfUnsecuredLines": 0.35,
                "age": 25,
                "NumberOfTime30-59DaysPastDueNotWorse": 2,
                "DebtRatio": 0.5,
                "MonthlyIncome": 5000,
                "NumberOfOpenCreditLinesAndLoans": 5,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 1,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2
            }
        },
    )
    age: int = Field(..., ge=18, description="Age of the applicant (must be at least 18)")
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0, description="Revolving utilization of unsecured lines (must be non-negative)")
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(..., ge=0, description="Number of times 30-59 days past due not worse (must be non-negative)", alias="NumberOfTime30-59DaysPastDueNotWorse")
    DebtRatio: float = Field(..., ge=0, description="Debt ratio (must be non-negative)")
    MonthlyIncome: Optional[float] = Field(None, ge=0, description="Monthly income (must be non-negative if provided)")
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0, description="Number of open credit lines and loans (must be non-negative)")
    NumberOfTimes90DaysLate: int = Field(..., ge=0, description="Number of times 90 days late (must be non-negative)")
    NumberRealEstateLoansOrLines: int = Field(..., ge=0, description=" Number of real estate loans or lines (must be non-negative)")
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(..., ge=0, description="Number of times 60-89 days past due not worse (must be non-negative)", alias="NumberOfTime60-89DaysPastDueNotWorse")
    NumberOfDependents: Optional[int] = Field(None, ge=0, description="Number of dependents (must be non-negative if provided)")
    
class CreditScoringOutput(BaseModel):
    """Prediction result returned by the /predict endpoint."""
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "example": {
                "probability_of_default": 0.15,
                "decision": "approved",
                "model_version": "v1.0.0"
            }
        },
    )
    probability_of_default: float = Field(..., ge=0, le=1, description="Predicted probability of default (between 0 and 1)")
    decision: Literal["approved", "review" ,"rejected"] = Field(..., description="Final decision indicating whether the applicant is approved, review or rejected")
    model_version: str = Field(..., description="Version of the credit scoring model used")

