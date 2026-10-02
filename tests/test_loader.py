"""
Unit tests for data loading and temporal splitting.
"""

import numpy as np
import pandas as pd

from src.data.split import temporal_train_val_test_split


def test_temporal_split_integrity():
    # Synthetic time-series DataFrame with steps 1 to 100
    steps = np.repeat(np.arange(1, 101), 10)
    df = pd.DataFrame(
        {
            "step": steps,
            "amount": np.random.uniform(10, 1000, size=len(steps)),
            "isFraud": np.random.choice([0, 1], size=len(steps), p=[0.98, 0.02]),
        }
    )

    train, val, test = temporal_train_val_test_split(
        df, step_col="step", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    # 1. Total rows match
    assert len(train) + len(val) + len(test) == len(df)

    # 2. Strict chronological order (no temporal overlap / leakage)
    assert train["step"].max() < val["step"].min()
    assert val["step"].max() < test["step"].min()

    # 3. Steps cover full range
    assert train["step"].min() == 1
    assert test["step"].max() == 100
