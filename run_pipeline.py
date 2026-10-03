"""
FinFraud Master Pipeline Orchestrator.
Executes the full lifecycle: Loading -> Temporal Split -> Feature Pipeline -> Multi-Model Training -> Cost-Sensitive Tuning -> SHAP Explainability -> Artifact Export.
"""

import argparse
import json
import logging
import os
import sys

import yaml

from src.data.loader import load_raw_data, validate_raw_data
from src.data.split import temporal_train_val_test_split
from src.explainability.explainer import FraudExplainer
from src.features.pipeline import FeaturePipeline
from src.models.evaluate import (
    compute_metrics,
    optimize_cost_threshold,
    save_evaluation_plots,
)
from src.models.train import (
    save_model_artifact,
    train_baseline_heuristic,
    train_lightgbm,
    train_logistic_regression,
    train_xgboost,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("FinFraudPipeline")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(
    config_path: str = "configs/config.yaml",
    sample_size: int | None = None,
):
    logger.info("================================================================")
    logger.info("       STARTING FINFRAUD END-TO-END TRAINING PIPELINE           ")
    logger.info("================================================================")

    cfg = load_config(config_path)

    # 1. Ingestion
    raw_path = cfg["data"]["raw_path"]
    logger.info(f"[Step 1/6] Ingesting dataset from {raw_path}...")
    raw_df = load_raw_data(file_path=raw_path, filter_fraud_types=True, sample_size=sample_size)
    validate_raw_data(raw_df)

    # 2. Temporal Splitting
    logger.info("[Step 2/6] Executing temporal chronological partitioning...")
    split_cfg = cfg["data"]["temporal_split"]
    train_raw, val_raw, test_raw = temporal_train_val_test_split(
        raw_df,
        step_col=cfg["data"]["step_col"],
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
    )

    # 3. Feature Pipeline Transformation
    logger.info("[Step 3/6] Applying balance discrepancy and velocity feature transformations...")
    pipeline = FeaturePipeline()
    X_train, y_train = pipeline.fit_transform(train_raw)
    X_val, y_val = pipeline.fit_transform(val_raw)
    X_test, y_test = pipeline.fit_transform(test_raw)

    logger.info(f"Feature matrix ready. Total features: {X_train.shape[1]}")
    logger.info(f"Feature columns: {list(X_train.columns)}")

    # 4. Multi-Model Training & Benchmarking
    logger.info("[Step 4/6] Training and benchmarking multiple model families...")

    results = {}

    # Model 1: Heuristic Baseline
    heuristic_metrics = train_baseline_heuristic(X_test, y_test, test_raw)
    results["Rule-Based Baseline (isFlaggedFraud)"] = heuristic_metrics
    logger.info(
        f"✔ Heuristic Baseline: Recall={heuristic_metrics['recall']:.2%}, PR-AUC={heuristic_metrics['pr_auc']:.4f}"
    )

    # Model 2: Logistic Regression
    _lr_pipe, lr_metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
    results["Logistic Regression (Balanced)"] = lr_metrics
    logger.info(
        f"✔ Logistic Regression: Recall={lr_metrics['recall']:.2%}, PR-AUC={lr_metrics['pr_auc']:.4f}"
    )

    # Model 3: XGBoost
    _xgb_model, xgb_metrics, _xgb_val_prob = train_xgboost(
        X_train, y_train, X_val, y_val, cfg["models"]["xgboost"]
    )
    results["XGBoost (Hist Gradient Boosting)"] = xgb_metrics
    logger.info(
        f"✔ XGBoost: Recall={xgb_metrics['recall']:.2%}, PR-AUC={xgb_metrics['pr_auc']:.4f}"
    )

    # Model 4: LightGBM
    lgb_model, lgb_metrics, lgb_val_prob = train_lightgbm(
        X_train, y_train, X_val, y_val, cfg["models"]["lightgbm"]
    )
    results["LightGBM (Gradient Boosting)"] = lgb_metrics
    logger.info(
        f"✔ LightGBM: Recall={lgb_metrics['recall']:.2%}, PR-AUC={lgb_metrics['pr_auc']:.4f}"
    )

    # 5. Cost-Sensitive Threshold Optimization on Validation Set
    logger.info("[Step 5/6] Performing cost-sensitive decision threshold optimization...")
    val_amounts = val_raw["amount"].values
    best_thresh, _best_cost_info, cost_curve_df = optimize_cost_threshold(
        y_true=y_val.values,
        y_prob=lgb_val_prob,
        amounts=val_amounts,
        cost_fp=cfg["financial_costs"]["cost_false_positive"],
        cost_fn_fixed=cfg["financial_costs"]["cost_false_negative_fixed"],
    )

    # Evaluate final LightGBM model on held-out Test Set at optimal threshold
    logger.info("Evaluating optimal LightGBM model on held-out Test partition...")
    lgb_test_prob = lgb_model.predict_proba(X_test)[:, 1]
    final_test_metrics = compute_metrics(y_test.values, lgb_test_prob, threshold=best_thresh)
    final_test_metrics["model_name"] = "LightGBM (Production Optimized)"
    final_test_metrics["optimal_threshold"] = best_thresh
    results["LightGBM (Production Test)"] = final_test_metrics

    logger.info("================================================================")
    logger.info(f"🏆 FINAL TEST RESULTS (LightGBM @ Threshold {best_thresh:.3f}):")
    logger.info(f"   - PR-AUC:     {final_test_metrics['pr_auc']:.4f}")
    logger.info(f"   - ROC-AUC:    {final_test_metrics['roc_auc']:.4f}")
    logger.info(f"   - Recall:     {final_test_metrics['recall']:.2%}")
    logger.info(f"   - Precision:  {final_test_metrics['precision']:.2%}")
    logger.info(f"   - F1-Score:   {final_test_metrics['f1']:.4f}")
    logger.info("================================================================")

    # 6. Explainability & Artifact Export
    logger.info("[Step 6/6] Generating SHAP global explanations and serializing artifacts...")
    os.makedirs(cfg["models"]["plots_dir"], exist_ok=True)
    os.makedirs(cfg["models"]["metrics_dir"], exist_ok=True)
    os.makedirs(cfg["models"]["models_dir"], exist_ok=True)

    # Save plots
    save_evaluation_plots(
        y_true=y_val.values,
        y_prob=lgb_val_prob,
        cost_df=cost_curve_df,
        best_threshold=best_thresh,
        model_name="LightGBM",
        output_dir=cfg["models"]["plots_dir"],
    )

    # SHAP explainer
    explainer = FraudExplainer(lgb_model, feature_names=list(X_train.columns))
    sample_for_shap = X_val.sample(n=min(1000, len(X_val)), random_state=42)
    explainer.generate_global_summary(
        sample_for_shap, output_path=os.path.join(cfg["models"]["plots_dir"], "shap_summary.png")
    )

    # Save benchmark metrics
    metrics_file = os.path.join(cfg["models"]["metrics_dir"], "benchmark_results.json")
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved benchmark results to {metrics_file}")

    # Save production model artifact
    save_model_artifact(
        model=lgb_model,
        feature_columns=list(X_train.columns),
        optimal_threshold=best_thresh,
        metrics=final_test_metrics,
        model_name="lightgbm_best",
        artifacts_dir=cfg["models"]["models_dir"],
    )

    logger.info("✔ FinFraud Pipeline execution completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinFraud Master Training Pipeline")
    parser.add_argument(
        "--sample-size", type=int, default=None, help="Sample size for fast debugging"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to YAML config"
    )
    args = parser.parse_args()

    run_pipeline(config_path=args.config, sample_size=args.sample_size)
