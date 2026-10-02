"""
Model Evaluation and Cost-Sensitive Threshold Optimization Module.
Calculates PR-AUC, ROC-AUC, Brier score, and expected financial loss across decision thresholds.
"""

import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """
    Computes standard and probabilistic evaluation metrics for imbalanced fraud classification.
    """
    y_pred = (y_prob >= threshold).astype(int)

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_vals, precision_vals)
    roc_auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    return {
        "threshold": float(threshold),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "brier_score": float(brier),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def optimize_cost_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray | None = None,
    cost_fp: float = 25.0,
    cost_fn_fixed: float = 50.0,
    num_thresholds: int = 100,
) -> tuple[float, dict[str, Any], pd.DataFrame]:
    """
    Finds the optimal decision threshold that minimizes total expected dollar loss.

    Loss Formulation:
    - False Positive: cost_fp (friction of alerting / manual investigator review)
    - False Negative: cost_fn_fixed + actual stolen transaction amount
    - True Positive / True Negative: $0
    """
    if amounts is None:
        amounts = np.ones_like(y_true) * 1000.0  # default average fallback

    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    cost_records = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        # Identify indices
        fp_mask = (y_pred == 1) & (y_true == 0)
        fn_mask = (y_pred == 0) & (y_true == 1)
        tp_mask = (y_pred == 1) & (y_true == 1)

        fp_cost = fp_mask.sum() * cost_fp
        fn_cost = (cost_fn_fixed * fn_mask.sum()) + amounts[fn_mask].sum()
        total_cost = fp_cost + fn_cost

        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        cost_records.append(
            {
                "threshold": t,
                "total_cost": total_cost,
                "fp_cost": fp_cost,
                "fn_cost": fn_cost,
                "fp_count": fp_mask.sum(),
                "fn_count": fn_mask.sum(),
                "tp_count": tp_mask.sum(),
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        )

    cost_df = pd.DataFrame(cost_records)
    best_idx = cost_df["total_cost"].idxmin()
    best_row = cost_df.loc[best_idx].to_dict()
    best_threshold = float(best_row["threshold"])

    logger.info(
        f"Optimal Financial Threshold: {best_threshold:.3f} | Min Total Loss: ${best_row['total_cost']:,.2f} (Recall: {best_row['recall']:.2%}, Precision: {best_row['precision']:.2%})"
    )
    return best_threshold, best_row, cost_df


def save_evaluation_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_df: pd.DataFrame,
    best_threshold: float,
    model_name: str = "LightGBM",
    output_dir: str = "artifacts/plots",
) -> None:
    """
    Generates and saves PR curve, ROC curve, and Financial Cost curve.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_vals, precision_vals)

    plt.figure(figsize=(8, 5))
    plt.plot(
        recall_vals,
        precision_vals,
        color="#2563EB",
        lw=2.5,
        label=f"{model_name} (PR-AUC = {pr_auc:.4f})",
    )
    plt.xlabel("Recall (Fraud Detection Rate)", fontsize=12)
    plt.ylabel("Precision (True Fraud Ratio in Alerts)", fontsize=12)
    plt.title(f"Precision-Recall Curve - {model_name}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name.lower()}_pr_curve.png"), dpi=200)
    plt.close()

    # 2. Financial Cost Curve vs Threshold
    plt.figure(figsize=(8, 5))
    plt.plot(
        cost_df["threshold"],
        cost_df["total_cost"],
        color="#DC2626",
        lw=2.5,
        label="Total Financial Loss ($)",
    )
    plt.plot(
        cost_df["threshold"],
        cost_df["fn_cost"],
        color="#F59E0B",
        linestyle="--",
        label="Fraud Loss (FN)",
    )
    plt.plot(
        cost_df["threshold"],
        cost_df["fp_cost"],
        color="#10B981",
        linestyle=":",
        label="Investigation Cost (FP)",
    )
    plt.axvline(
        best_threshold,
        color="#1E293B",
        linestyle="-.",
        label=f"Optimal Threshold ({best_threshold:.2f})",
    )
    plt.xlabel("Decision Threshold", fontsize=12)
    plt.ylabel("Cost ($)", fontsize=12)
    plt.title(f"Financial Cost Optimization Curve - {model_name}", fontsize=14, fontweight="bold")
    plt.legend(loc="upper center", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name.lower()}_cost_curve.png"), dpi=200)
    plt.close()
