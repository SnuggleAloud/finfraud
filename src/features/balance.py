"""
Balance Discrepancy Feature Engineering Module.
Computes accounting balance inconsistencies that are primary indicators of fraudulent drain actions.
"""

import pandas as pd


def compute_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific balance discrepancy features.

    Features created:
    - errorBalanceOrig: discrepancy between expected balance change and recorded newbalanceOrig
    - errorBalanceDest: discrepancy between expected balance change and recorded newbalanceDest
    - isOrigEmptied: binary flag indicating if source account was completely drained
    - isDestZeroBefore: binary flag indicating if recipient account had zero balance before transaction
    - isMerchantDest: binary flag for merchant destination accounts
    """
    data = df.copy()

    # 1. Origin Account Balance Error
    # Expected: newbalanceOrig == oldbalanceOrg - amount
    # Error: newbalanceOrig + amount - oldbalanceOrg (should be 0 for valid transactions)
    data["errorBalanceOrig"] = data["newbalanceOrig"] + data["amount"] - data["oldbalanceOrg"]

    # 2. Destination Account Balance Error
    # Expected: newbalanceDest == oldbalanceDest + amount
    # Error: oldbalanceDest + amount - newbalanceDest (should be 0 for valid transactions)
    data["errorBalanceDest"] = data["oldbalanceDest"] + data["amount"] - data["newbalanceDest"]

    # 3. Handle Merchant Accounts (Balances are not tracked in simulation, always 0.0)
    is_merchant = data["nameDest"].astype(str).str.startswith("M")
    data["isMerchantDest"] = is_merchant.astype("int8")
    data.loc[is_merchant, "errorBalanceDest"] = 0.0

    # 4. Critical drain & zero indicators
    data["isOrigEmptied"] = ((data["newbalanceOrig"] == 0) & (data["oldbalanceOrg"] > 0)).astype(
        "int8"
    )
    data["isDestZeroBefore"] = (data["oldbalanceDest"] == 0).astype("int8")
    data["isDestZeroAfter"] = (data["newbalanceDest"] == 0).astype("int8")

    # 5. Relative amount to balance ratio
    # Avoid division by zero with small epsilon
    data["origAmountRatio"] = data["amount"] / (data["oldbalanceOrg"] + 1.0)

    return data
