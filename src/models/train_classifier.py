"""
LAYER 2 — AI / Machine Learning
Train XGBoost binary classifier to predict which products
to recommend for a given client.

For each (client, product) pair, predicts:
    - 1: this product will likely be ordered (recommend)
    - 0: this product will NOT be ordered (skip)

Input features  : feature matrix from feature_engineering.py
Output          : trained XGBoost classifier + threshold
Saved artifact  : models/classifier_lsat.pkl
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def prepare_classifier_dataset(
    features: pd.DataFrame,
    df_history: pd.DataFrame,
    reference_date: pd.Timestamp = None,
    horizon_days: int = 30,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the training dataset for the binary classifier.

    Label = 1 if the (client, product) pair was ordered
    within the next `horizon_days` days from reference_date.

    Args:
        features: feature matrix from build_feature_matrix()
        df_history: cleaned main table (used for label creation)
        reference_date: split date (default: today - horizon_days)
        horizon_days: number of days for the future window

    Returns:
        X: feature DataFrame
        y: binary target Series
    """
    raise NotImplementedError("To be implemented")


def train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict = None,
    save_path: str = "models/classifier_lsat.pkl",
) -> object:
    """
    Train and save an XGBoost binary classifier.

    Args:
        X: training features
        y: binary labels
        params: XGBoost hyperparameters (uses defaults if None)
        save_path: path to save the trained model

    Returns:
        Trained XGBoost classifier
    """
    raise NotImplementedError("To be implemented")


def evaluate_classifier(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate the classifier with standard binary metrics.

    Metrics computed:
    - accuracy, precision, recall, f1
    - ROC-AUC
    - confusion matrix

    Args:
        model: trained XGBoost classifier
        X_test: test features
        y_test: test labels

    Returns:
        dict of metric name → value
    """
    raise NotImplementedError("To be implemented")


def load_classifier(path: str = "models/classifier_lsat.pkl") -> object:
    """
    Load a trained classifier from disk.

    Args:
        path: path to the .pkl file

    Returns:
        Trained XGBoost classifier
    """
    raise NotImplementedError("To be implemented")
