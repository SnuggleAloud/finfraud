"""
Model Explainability Module using TreeSHAP.
Provides global feature importance summaries and transaction-level decision breakdowns.
"""

import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


class FraudExplainer:
    """
    SHAP-based model explainer for tree models (LightGBM / XGBoost).
    """

    def __init__(self, model: Any, feature_names: list[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def generate_global_summary(
        self, X_sample: pd.DataFrame, output_path: str = "artifacts/plots/shap_summary.png"
    ) -> None:
        """
        Generates and saves a SHAP beeswarm summary plot for a sample of transactions.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shap_values = self.explainer.shap_values(X_sample)

        # LightGBM binary classification TreeExplainer returns a list of 2 arrays or 1 array
        if isinstance(shap_values, list) and len(shap_values) == 2:
            values_to_plot = shap_values[1]
        else:
            values_to_plot = shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(values_to_plot, X_sample, feature_names=self.feature_names, show=False)
        plt.title("SHAP Global Feature Importance", fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        logger.info(f"Saved SHAP summary plot to {output_path}")

    def explain_transaction(self, x_single: pd.DataFrame, top_n: int = 5) -> dict[str, Any]:
        """
        Computes local feature contributions for a single transaction.

        Returns:
        --------
        Dict with top positive (risk-increasing) and negative (risk-reducing) features.
        """
        if isinstance(x_single, pd.Series):
            x_single = x_single.to_frame().T

        shap_values = self.explainer.shap_values(x_single)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            vals = shap_values[1][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            vals = shap_values[0]
        else:
            vals = shap_values

        features_dict = []
        for name, shap_val, feat_val in zip(self.feature_names, vals, x_single.iloc[0].values):
            features_dict.append(
                {
                    "feature": name,
                    "value": float(feat_val),
                    "shap_impact": float(shap_val),
                    "direction": "RISK_INCREASING" if shap_val > 0 else "RISK_DECREASING",
                }
            )

        # Sort by absolute impact
        features_dict.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
        top_factors = features_dict[:top_n]

        return {
            "top_factors": top_factors,
            "raw_shap_values": {item["feature"]: item["shap_impact"] for item in features_dict},
        }
