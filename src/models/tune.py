"""
Hyperparameter Tuning Module using Optuna.
Optimizes LightGBM hyperparameters targeting PR-AUC or business loss.
"""

import logging
from typing import Any

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 15,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Executes an Optuna Bayesian search over LightGBM hyperparameter space targeting PR-AUC.
    """
    logger.info(f"Starting Optuna hyperparameter optimization ({n_trials} trials)...")
    pos_ratio = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": pos_ratio,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )

        y_prob = model.predict_proba(X_val)[:, 1]
        precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_prob)
        score = auc(recall_vals, precision_vals)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Optuna Best Trial PR-AUC: {study.best_value:.4f}")
    logger.info(f"Best Hyperparameters: {study.best_params}")
    return study.best_params
