"""
Model Training and Benchmarking Module.
Trains Baseline, Logistic Regression, LightGBM, and XGBoost models on temporal splits.
"""

import json
import logging
import os
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.evaluate import (
    compute_metrics,
)

logger = logging.getLogger(__name__)


def train_baseline_heuristic(
    X_test: pd.DataFrame, y_test: pd.Series, raw_df_test: pd.DataFrame
) -> dict[str, Any]:
    """
    Evaluates the default simulated rule-based system (`isFlaggedFraud`: TRANSFER > 200,000).
    """
    if "isFlaggedFraud" in raw_df_test.columns:
        y_pred = raw_df_test["isFlaggedFraud"].values
        y_prob = y_pred.astype(float)
    else:
        y_pred = ((X_test["amount"] > 200000) & (X_test["type_TRANSFER"] == 1)).astype(int).values
        y_prob = y_pred.astype(float)

    metrics = compute_metrics(y_test.values, y_prob, threshold=0.5)
    metrics["model_name"] = "Rule-Based Baseline (isFlaggedFraud)"
    return metrics


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """
    Trains a balanced Logistic Regression model with feature scaling.
    """
    logger.info("Training Logistic Regression...")
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=random_state
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    y_prob_val = pipe.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val.values, y_prob_val, threshold=0.5)
    metrics["model_name"] = "Logistic Regression"
    return pipe, metrics


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> tuple[lgb.LGBMClassifier, dict[str, Any], np.ndarray]:
    """
    Trains an optimized LightGBM Classifier with scale_pos_weight.
    """
    logger.info("Training LightGBM Classifier...")
    pos_ratio = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    default_params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 7,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": pos_ratio,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    y_prob_val = model.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val.values, y_prob_val, threshold=0.5)
    metrics["model_name"] = "LightGBM"
    return model, metrics, y_prob_val


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> tuple[xgb.XGBClassifier, dict[str, Any], np.ndarray]:
    """
    Trains an optimized XGBoost Classifier with scale_pos_weight.
    """
    logger.info("Training XGBoost Classifier...")
    pos_ratio = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    default_params = {
        "n_estimators": 150,
        "learning_rate": 0.08,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": pos_ratio,
        "random_state": random_state,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "tree_method": "hist",
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_prob_val = model.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val.values, y_prob_val, threshold=0.5)
    metrics["model_name"] = "XGBoost"
    return model, metrics, y_prob_val


def save_model_artifact(
    model: Any,
    feature_columns: list,
    optimal_threshold: float,
    metrics: dict[str, Any],
    model_name: str = "lightgbm_best",
    artifacts_dir: str = "artifacts/models",
) -> str:
    """
    Serializes model and metadata to disk.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, f"{model_name}.joblib")
    meta_path = os.path.join(artifacts_dir, f"{model_name}_meta.json")

    joblib.dump(model, model_path)

    metadata = {
        "model_name": model_name,
        "feature_columns": feature_columns,
        "optimal_threshold": optimal_threshold,
        "metrics": metrics,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved model artifact to {model_path} and metadata to {meta_path}")
    return model_path
