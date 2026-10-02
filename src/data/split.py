"""
Temporal Data Splitting Module.
Prevents lookahead / future-to-past data leakage by partitioning on transaction time step.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def temporal_train_val_test_split(
    df: pd.DataFrame,
    step_col: str = "step",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits DataFrame chronologically into train, validation, and test partitions
    based on the time step column.

    Parameters:
    -----------
    df : pd.DataFrame
        Input transaction dataframe.
    step_col : str
        Column denoting time step (e.g., hours 1 to 744).
    train_ratio : float
        Fraction of time span for training (default 0.70).
    val_ratio : float
        Fraction of time span for validation/tuning (default 0.15).
    test_ratio : float
        Fraction of time span for final held-out test (default 0.15).

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    min_step = df[step_col].min()
    max_step = df[step_col].max()
    total_steps = max_step - min_step + 1

    train_step_cutoff = min_step + int(total_steps * train_ratio)
    val_step_cutoff = train_step_cutoff + int(total_steps * val_ratio)

    train_df = df[df[step_col] <= train_step_cutoff].copy()
    val_df = df[(df[step_col] > train_step_cutoff) & (df[step_col] <= val_step_cutoff)].copy()
    test_df = df[df[step_col] > val_step_cutoff].copy()

    logger.info("Temporal Split Summary:")
    logger.info(
        f"  - Train: steps [{min_step} - {train_step_cutoff}] | {len(train_df):,} rows | Fraud: {train_df['isFraud'].sum()} ({train_df['isFraud'].mean()*100:.3f}%)"
    )
    logger.info(
        f"  - Val:   steps [{train_step_cutoff+1} - {val_step_cutoff}] | {len(val_df):,} rows | Fraud: {val_df['isFraud'].sum()} ({val_df['isFraud'].mean()*100:.3f}%)"
    )
    logger.info(
        f"  - Test:  steps [{val_step_cutoff+1} - {max_step}] | {len(test_df):,} rows | Fraud: {test_df['isFraud'].sum()} ({test_df['isFraud'].mean()*100:.3f}%)"
    )

    return train_df, val_df, test_df
