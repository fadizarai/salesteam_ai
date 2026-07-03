"""
LAYER 2 — AI / Machine Learning
Train XGBoost regressor to predict the optimal quantity
for each recommended product.

Given a (client, product) pair that the classifier flagged as
"recommend", this model predicts the quantity to suggest.

Input features  : feature matrix from feature_engineering.py
                  (only rows where classifier predicts 1)
Output          : predicted quantity (float → rounded to int)
Saved artifact  : models/regressor_lsat.pkl
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def prepare_regressor_dataset(
    features: pd.DataFrame,
    df_history: pd.DataFrame,
    reference_date: pd.Timestamp = None,
    horizon_days: int = 30,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the training dataset for the quantity regressor.

    Target = average quantity ordered in the next `horizon_days`
    for (client, product) pairs that were actually ordered.

    Args:
        features: feature matrix from build_feature_matrix()
        df_history: cleaned main table
        reference_date: split date
        horizon_days: future window in days

    Returns:
        X: feature DataFrame
        y: quantity target Series
    """
    raise NotImplementedError("To be implemented")


def train_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict = None,
    save_path: str = "models/regressor_lsat.pkl",
) -> object:
    """
    Train and save an XGBoost regressor for quantity prediction.

    Args:
        X: training features
        y: quantity targets
        params: XGBoost hyperparameters (uses defaults if None)
        save_path: path to save the trained model

    Returns:
        Trained XGBoost regressor
    """
    raise NotImplementedError("To be implemented")


def evaluate_regressor(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate the regressor with standard regression metrics.

    Metrics computed:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - R² score

    Args:
        model: trained XGBoost regressor
        X_test: test features
        y_test: test targets

    Returns:
        dict of metric name → value
    """
    raise NotImplementedError("To be implemented")


def load_regressor(path: str = "models/regressor_lsat.pkl") -> object:
    """
    Load a trained regressor from disk.

    Args:
        path: path to the .pkl file

    Returns:
        Trained XGBoost regressor
    """
    raise NotImplementedError("To be implemented")
