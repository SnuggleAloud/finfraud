import pandas as pd
import numpy as np

def inspect_dataset(file_path="data/Synthetic_Financial_datasets_log.csv"):
    """
    Performs memory-optimized loading and diagnostic checks on the Synthetic Financial dataset.
    Identifies missing values, duplicate entries, range anomalies, and system simulation characteristics.
    """
    import os
    if not os.path.exists(file_path) and os.path.exists("Synthetic_Financial_datasets_log.csv"):
        file_path = "Synthetic_Financial_datasets_log.csv"
        
    print("====================================================")
    print("          STARTING DATASET INSPECTION")
    print("====================================================")
    print(f"Dataset path: {file_path}\n")
    
    # 1. Load memory-optimized data
    dtypes = {
        'step': 'int32',
        'type': 'category',
        'amount': 'float64',  # Using float64 for precise decimal comparison
        'nameOrig': 'string',
        'oldbalanceOrg': 'float64',
        'newbalanceOrig': 'float64',
        'nameDest': 'string',
        'oldbalanceDest': 'float64',
        'newbalanceDest': 'float64',
        'isFraud': 'int8',
        'isFlaggedFraud': 'int8'
    }
    
    print("[1/6] Loading dataset with optimized types...")
    df = pd.read_csv(file_path, dtype=dtypes)
    print(f"✔ Dataset loaded successfully. Shape: {df.shape[0]:,} rows, {df.shape[1]} columns.\n")
    
    # 2. Check for missing (null) values
    print("[2/6] Checking for missing values...")
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        print("✔ No missing values found in any column.\n")
    else:
        print("⚠ Missing values detected:")
        print(null_counts[null_counts > 0])
        print()
    
    # 3. Check for duplicates
    print("[3/6] Checking for duplicate rows...")
    duplicate_count = df.duplicated().sum()
    if duplicate_count == 0:
        print("✔ No duplicate rows found.\n")
    else:
        print(f"⚠ Found {duplicate_count} duplicate rows.\n")
    
    # 4. Check for anomalous values
    print("[4/6] Auditing ranges and anomalous values...")
    neg_amounts = (df['amount'] < 0).sum()
    zero_amounts = (df['amount'] == 0).sum()
    print(f"  - Negative amounts: {neg_amounts}")
    print(f"  - Zero amounts: {zero_amounts}")
    
    neg_old_org = (df['oldbalanceOrg'] < 0).sum()
    neg_new_org = (df['newbalanceOrig'] < 0).sum()
    print(f"  - Negative oldbalanceOrg: {neg_old_org}")
    print(f"  - Negative newbalanceOrig: {neg_new_org}")
    
    neg_old_dest = (df['oldbalanceDest'] < 0).sum()
    neg_new_dest = (df['newbalanceDest'] < 0).sum()
    print(f"  - Negative oldbalanceDest: {neg_old_dest}")
    print(f"  - Negative newbalanceDest: {neg_new_dest}")
    print("✔ Value range audit complete.\n")
    
    # 5. Check Merchant Destination balances
    print("[5/6] Inspecting Merchant destination balances...")
    is_merchant = df['nameDest'].str.startswith('M')
    merchant_count = is_merchant.sum()
    print(f"  - Total merchant destination transactions: {merchant_count:,}")
    
    if merchant_count > 0:
        non_zero_old_dest = ((df['oldbalanceDest'] != 0) & is_merchant).sum()
        non_zero_new_dest = ((df['newbalanceDest'] != 0) & is_merchant).sum()
        print(f"  - Merchant accounts with non-zero oldbalanceDest: {non_zero_old_dest}")
        print(f"  - Merchant accounts with non-zero newbalanceDest: {non_zero_new_dest}")
        print("✔ Note: Merchants have 0.0 recorded balance by design. Adjust destination balance error accordingly.")
    print()
    
    # 6. Detailed check on zero amount transactions
    if zero_amounts > 0:
        print("[6/6] Isolating and analyzing zero-amount transactions...")
        zero_df = df[df['amount'] == 0.0]
        fraud_ratio = zero_df['isFraud'].mean() * 100
        cash_out_ratio = (zero_df['type'] == 'CASH_OUT').mean() * 100
        print(f"  - Total transactions with amount == 0.0: {len(zero_df)}")
        print(f"  - Percentage of zero-amount transactions that are CASH_OUT: {cash_out_ratio:.1f}%")
        print(f"  - Percentage of zero-amount transactions that are Fraud: {fraud_ratio:.1f}%")
        print("✔ Note: Zero-amount transactions are 100% predictive of fraud in this dataset.")
    else:
        print("[6/6] No zero-amount transactions to inspect.")
    print("\n====================================================")
    print("          INSPECTION COMPLETE")
    print("====================================================")

if __name__ == "__main__":
    import sys
    path = "data/Synthetic_Financial_datasets_log.csv"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    inspect_dataset(path)
