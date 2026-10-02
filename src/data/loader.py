"""
Data Ingestion and Memory-Optimized Loading Module.
"""

import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Optimized schema for memory efficiency
DEFAULT_DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float32",
    "nameOrig": "string",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "nameDest": "string",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


def load_raw_data(
    file_path: str = "data/Synthetic_Financial_datasets_log.csv",
    filter_fraud_types: bool = True,
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Loads dataset with optimized types and optional filtering for fraud-prone transaction types.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file.
    filter_fraud_types : bool
        If True, filters to only 'TRANSFER' and 'CASH_OUT' (where 100% of fraud occurs).
    sample_size : Optional[int]
        If provided, randomly samples N rows for rapid experimentation.
    random_state : int
        Random seed for reproducibility when sampling.

    Returns:
    --------
    pd.DataFrame: Cleaned, memory-optimized DataFrame.
    """
    import os

    # Robust path resolution for root vs notebooks execution
    resolved_path = file_path
    if not os.path.exists(resolved_path):
        candidate_paths = [
            os.path.join("data", os.path.basename(file_path)),
            os.path.join("..", "data", os.path.basename(file_path)),
            os.path.basename(file_path),
        ]
        for candidate in candidate_paths:
            if os.path.exists(candidate):
                resolved_path = candidate
                break

    logger.info(f"Loading dataset from: {resolved_path}")
    df = pd.read_csv(resolved_path, dtype=DEFAULT_DTYPES)
    logger.info(
        f"Loaded {len(df):,} rows and {len(df.columns)} columns. Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB"
    )

    if filter_fraud_types:
        fraud_types = ["TRANSFER", "CASH_OUT"]
        initial_len = len(df)
        df = df[df["type"].isin(fraud_types)].copy()
        df["type"] = df["type"].cat.remove_unused_categories()
        logger.info(
            f"Filtered for {fraud_types}: {len(df):,} rows retained ({len(df)/initial_len*100:.1f}% of total)."
        )

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        logger.info(f"Subsampled to {sample_size:,} rows for rapid prototyping.")

    return df


def validate_raw_data(df: pd.DataFrame) -> bool:
    """
    Validates essential assertions on dataset integrity.
    """
    required_cols = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    if df.isnull().sum().sum() > 0:
        logger.warning("Dataset contains null values!")

    return True
