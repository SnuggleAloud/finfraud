"""
Feature Engineering Pipeline Orchestrator.
Standardizes feature transformation across training, validation, and real-time inference.
"""

import logging

import numpy as np
import pandas as pd

from src.features.balance import compute_balance_features
from src.features.velocity import compute_temporal_and_velocity_features

logger = logging.getLogger(__name__)

# Expected model features in deterministic order
FEATURE_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
    "isMerchantDest",
    "isOrigEmptied",
    "isDestZeroBefore",
    "isDestZeroAfter",
    "origAmountRatio",
    "hourOfDay",
    "dayOfWeek",
    "isNightTime",
    "logAmount",
    "isZeroAmount",
    "origTxCount",
    "destTxCount",
    "type_TRANSFER",
]


class FeaturePipeline:
    """
    End-to-end feature pipeline ensuring consistent transformations across environments.
    """

    def __init__(self, feature_columns: list[str] | None = None):
        self.feature_columns = feature_columns or FEATURE_COLUMNS

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw transaction dataframe into a clean numeric feature matrix.
        """
        # 1. Apply domain balance features
        df_feat = compute_balance_features(df)

        # 2. Apply temporal & velocity features
        df_feat = compute_temporal_and_velocity_features(df_feat)

        # 3. Categorical encoding for 'type'
        # In Paysim TRANSFER and CASH_OUT are the two fraud-relevant types
        if "type" in df_feat.columns:
            if "type_TRANSFER" not in df_feat.columns:
                df_feat["type_TRANSFER"] = (df_feat["type"].astype(str) == "TRANSFER").astype(
                    "int8"
                )
        elif "type_TRANSFER" not in df_feat.columns:
            df_feat["type_TRANSFER"] = 0

        # 4. Fill any infinities or NaNs safely
        for col in ["origAmountRatio", "errorBalanceOrig", "errorBalanceDest", "logAmount"]:
            if col in df_feat.columns:
                df_feat[col] = df_feat[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # 5. Ensure all expected feature columns exist
        for col in self.feature_columns:
            if col not in df_feat.columns:
                df_feat[col] = 0

        # Reorder to strict schema
        return df_feat[self.feature_columns]

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
        """
        Transforms DataFrame and separates target if present.
        """
        y = df["isFraud"].copy() if "isFraud" in df.columns else None
        X = self.transform(df)
        return X, y
