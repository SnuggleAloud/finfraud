"""
Unit tests for FinFraud feature engineering and transformations.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.balance import compute_balance_features
from src.features.pipeline import FEATURE_COLUMNS, FeaturePipeline
from src.features.velocity import compute_temporal_and_velocity_features


@pytest.fixture
def sample_transaction_df():
    return pd.DataFrame(
        [
            {
                "step": 10,
                "type": "TRANSFER",
                "amount": 5000.0,
                "nameOrig": "C1001",
                "oldbalanceOrg": 5000.0,
                "newbalanceOrig": 0.0,
                "nameDest": "C2001",
                "oldbalanceDest": 1000.0,
                "newbalanceDest": 6000.0,
                "isFraud": 1,
            },
            {
                "step": 14,
                "type": "PAYMENT",
                "amount": 150.0,
                "nameOrig": "C1002",
                "oldbalanceOrg": 1000.0,
                "newbalanceOrig": 850.0,
                "nameDest": "M3001",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFraud": 0,
            },
            {
                "step": 25,
                "type": "CASH_OUT",
                "amount": 0.0,
                "nameOrig": "C1003",
                "oldbalanceOrg": 0.0,
                "newbalanceOrig": 0.0,
                "nameDest": "C2002",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFraud": 1,
            },
        ]
    )


def test_balance_features(sample_transaction_df):
    df_feat = compute_balance_features(sample_transaction_df)

    # 1. Error for normal transfer should be 0.0
    assert np.isclose(df_feat.loc[0, "errorBalanceOrig"], 0.0)
    assert np.isclose(df_feat.loc[0, "errorBalanceDest"], 0.0)

    # 2. Emptied flag should be 1 for origin 1
    assert df_feat.loc[0, "isOrigEmptied"] == 1

    # 3. Merchant destination balance error should be explicitly reset to 0.0
    assert df_feat.loc[1, "isMerchantDest"] == 1
    assert np.isclose(df_feat.loc[1, "errorBalanceDest"], 0.0)


def test_velocity_and_temporal_features(sample_transaction_df):
    df_feat = compute_temporal_and_velocity_features(sample_transaction_df)

    # 1. Hour of day
    assert df_feat.loc[0, "hourOfDay"] == 10
    assert df_feat.loc[1, "hourOfDay"] == 14
    assert df_feat.loc[2, "hourOfDay"] == 1  # 25 % 24 = 1

    # 2. Zero amount anomaly flag
    assert df_feat.loc[2, "isZeroAmount"] == 1
    assert df_feat.loc[0, "isZeroAmount"] == 0


def test_feature_pipeline_schema(sample_transaction_df):
    pipeline = FeaturePipeline()
    X, y = pipeline.fit_transform(sample_transaction_df)

    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == 3
    assert y is not None
    assert len(y) == 3
    assert not X.isnull().any().any()
