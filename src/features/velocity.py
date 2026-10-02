"""
Velocity and Temporal Feature Engineering Module.
Extracts cyclical time patterns and behavioral velocity signals.
"""

import numpy as np
import pandas as pd


def compute_temporal_and_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cyclical time signals and entity velocity/frequency profiles.

    Features created:
    - hourOfDay: hour in 24-hour cycle (0-23)
    - dayOfWeek: day index in weekly cycle (0-6)
    - isNightTime: indicator for off-peak transaction hours (00:00 to 06:00)
    - isZeroAmount: exact 0.0 amount anomaly flag (100% fraud in dataset)
    - origTxCount: cumulative transaction count for origin account up to current step
    - destTxCount: cumulative transaction count for destination account up to current step
    - logAmount: log-transformed transaction amount
    """
    data = df.copy()

    # 1. Cyclical time features (1 step = 1 hour)
    data["hourOfDay"] = (data["step"] % 24).astype("int8")
    data["dayOfWeek"] = ((data["step"] // 24) % 7).astype("int8")
    data["isNightTime"] = (data["hourOfDay"].between(0, 5)).astype("int8")

    # 2. Mathematical transformations
    data["logAmount"] = np.log1p(data["amount"].clip(lower=0))
    data["isZeroAmount"] = (data["amount"] == 0.0).astype("int8")

    # 3. Entity profile & velocity features
    # Cumulative transaction counts up to current record (leak-free)
    if "nameOrig" in data.columns:
        data["origTxCount"] = data.groupby("nameOrig").cumcount().astype("int16") + 1
    else:
        data["origTxCount"] = 1

    if "nameDest" in data.columns:
        data["destTxCount"] = data.groupby("nameDest").cumcount().astype("int16") + 1
    else:
        data["destTxCount"] = 1

    return data
