"""
Pydantic Schemas for Real-Time Fraud Scoring API.
"""

from typing import Literal

from pydantic import BaseModel, Field


class TransactionPayload(BaseModel):
    """Single transaction schema for scoring."""

    step: int = Field(..., ge=1, description="Hour index of the transaction (e.g. 1 to 744)")
    type: Literal["TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT", "PAYMENT"] = Field(
        ..., description="Transaction type"
    )
    amount: float = Field(..., ge=0, description="Amount of the transaction")
    nameOrig: str = Field(..., description="Customer ID initiating the transaction")
    oldbalanceOrg: float = Field(
        ..., ge=0, description="Initial balance of origin before transaction"
    )
    newbalanceOrig: float = Field(..., ge=0, description="New balance of origin after transaction")
    nameDest: str = Field(..., description="Recipient ID (Customer 'C...' or Merchant 'M...')")
    oldbalanceDest: float = Field(
        default=0.0, ge=0, description="Initial balance of recipient before transaction"
    )
    newbalanceDest: float = Field(
        default=0.0, ge=0, description="New balance of recipient after transaction"
    )


class DecisionFactor(BaseModel):
    feature: str
    value: float
    shap_impact: float
    direction: str


class PredictionResponse(BaseModel):
    """Response model for a single transaction scoring request."""

    fraud_probability: float
    risk_score: int = Field(..., description="Risk score on scale 0 to 1000")
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    decision: Literal["APPROVE", "MANUAL_REVIEW", "DECLINE"]
    threshold_applied: float
    top_factors: list[DecisionFactor]


class BatchTransactionPayload(BaseModel):
    transactions: list[TransactionPayload]


class BatchPredictionResponse(BaseModel):
    total_scored: int
    flagged_count: int
    results: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_name: str
    optimal_threshold: float
    feature_count: int
    version: str
