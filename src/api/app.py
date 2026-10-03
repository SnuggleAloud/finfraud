"""
FastAPI Real-Time Fraud Scoring Service.
Exposes endpoints for transaction evaluation, SHAP reasoning, and health diagnostics.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    BatchPredictionResponse,
    BatchTransactionPayload,
    DecisionFactor,
    HealthResponse,
    PredictionResponse,
    TransactionPayload,
)
from src.explainability.explainer import FraudExplainer
from src.features.pipeline import FeaturePipeline

logger = logging.getLogger(__name__)

# Global model state
state: dict[str, Any] = {"model": None, "metadata": None, "pipeline": None, "explainer": None}


def load_model_state():
    """Loads model artifact, metadata, pipeline, and SHAP explainer."""
    model_dir = "artifacts/models"
    model_path = os.path.join(model_dir, "lightgbm_best.joblib")
    meta_path = os.path.join(model_dir, "lightgbm_best_meta.json")

    if os.path.exists(model_path) and os.path.exists(meta_path):
        state["model"] = joblib.load(model_path)
        with open(meta_path, "r") as f:
            state["metadata"] = json.load(f)

        feature_cols = state["metadata"].get("feature_columns", [])
        state["pipeline"] = FeaturePipeline(feature_columns=feature_cols)
        state["explainer"] = FraudExplainer(state["model"], feature_names=feature_cols)
        logger.info(f"Loaded production model from {model_path}")
    else:
        logger.warning(
            f"No trained model found at {model_path}. API will run in mock mode until pipeline is executed."
        )


# Initialize on startup
load_model_state()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_state()
    yield


app = FastAPI(
    title="FinFraud Scoring Service",
    description="Real-Time Transaction Fraud Detection & Decision API with SHAP Explainability",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
def root():
    return {"service": "FinFraud Scoring API", "status": "online", "docs_url": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health(response: Response):
    if state["model"] is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="uninitialized",
            model_name="none",
            optimal_threshold=0.5,
            feature_count=0,
            version="1.0.0",
        )
    return HealthResponse(
        status="healthy",
        model_name=state["metadata"].get("model_name", "LightGBM"),
        optimal_threshold=state["metadata"].get("optimal_threshold", 0.5),
        feature_count=len(state["metadata"].get("feature_columns", [])),
        version="1.0.0",
    )


@app.post("/v1/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_transaction(payload: TransactionPayload):
    """
    Evaluates a single transaction payload in real time.
    """
    if state["model"] is None or state["pipeline"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not yet initialized. Please run the training pipeline first.",
        )

    # Convert payload to single row DataFrame
    df_raw = pd.DataFrame([payload.model_dump()])
    X_feat = state["pipeline"].transform(df_raw)

    # Predict probability
    prob = float(state["model"].predict_proba(X_feat)[:, 1][0])
    threshold = float(state["metadata"].get("optimal_threshold", 0.5))

    # Calculate risk score (0 - 1000)
    risk_score = int(prob * 1000)

    # Decision and Risk Tier Logic
    if prob >= 0.85:
        decision = "DECLINE"
        risk_tier = "CRITICAL"
    elif prob >= threshold:
        decision = "MANUAL_REVIEW"
        risk_tier = "HIGH"
    elif prob >= max(0.10, threshold * 0.5):
        decision = "APPROVE"
        risk_tier = "MEDIUM"
    else:
        decision = "APPROVE"
        risk_tier = "LOW"

    # Compute SHAP explanation
    explanation = state["explainer"].explain_transaction(X_feat, top_n=4)
    top_factors = [
        DecisionFactor(
            feature=f["feature"],
            value=f["value"],
            shap_impact=f["shap_impact"],
            direction=f["direction"],
        )
        for f in explanation["top_factors"]
    ]

    return PredictionResponse(
        fraud_probability=prob,
        risk_score=risk_score,
        risk_tier=risk_tier,
        decision=decision,
        threshold_applied=threshold,
        top_factors=top_factors,
    )


@app.post("/v1/predict_batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(payload: BatchTransactionPayload):
    """
    Scores a batch of transactions.
    """
    results = [predict_transaction(tx) for tx in payload.transactions]
    flagged = sum(1 for r in results if r.decision in ["DECLINE", "MANUAL_REVIEW"])

    return BatchPredictionResponse(
        total_scored=len(results), flagged_count=flagged, results=results
    )
