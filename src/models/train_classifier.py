"""
LAYER 2 — AI / Machine Learning
Train XGBoost binary classifier to predict which products
to recommend for a given client.

Input features  : training_set.csv from target_builder.py
Output          : trained XGBoost classifier
Saved artifact  : models/classifier_lsat.pkl
"""

import pandas as pd
import numpy as np
import logging
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb

logger = logging.getLogger(__name__)


def prepare_classifier_dataset(
    training_set_path: str = "data/processed/training_set.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load the training set, separate features and target, and split into train/test sets.

    Args:
        training_set_path: Path to the generated training set CSV.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info(f"Loading training set from {training_set_path}...")
    df = pd.read_csv(training_set_path)

    # 1. Separate target and features
    y = df["target_bought"]
    
    # Drop identifier columns and targets
    cols_to_drop = [
        "code_client",
        "code_article",
        "designation",
        "target_qty",
        "target_bought"
    ]
    X = df.drop(columns=cols_to_drop, errors="ignore")

    # 2. Preprocess data types for XGBoost
    # Convert booleans to int (0/1)
    bool_cols = X.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        X[col] = X[col].astype(int)

    # Convert categorical string columns to category type
    cat_cols = X.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        logger.info(f"Converting column '{col}' to category type")
        X[col] = X[col].astype("category")

    logger.info(f"Features list: {list(X.columns)}")
    logger.info(f"Feature matrix shape: {X.shape}")

    # 3. Stratified split to maintain label distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    params: dict = None,
    save_path: str = "models/classifier_lsat.pkl",
) -> xgb.XGBClassifier:
    """
    Train and save an XGBoost binary classifier.

    Args:
        X_train: Training features.
        y_train: Training target labels.
        X_val: Optional validation features.
        y_val: Optional validation targets.
        params: XGBoost hyperparameters dict.
        save_path: Filepath to save the pickled model.

    Returns:
        Trained XGBoost classifier.
    """
    logger.info("Initializing XGBoost classifier...")

    # Default parameters optimized for tabular recommendation classification
    default_params = {
        "n_estimators": 150,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
        "enable_categorical": True,  # Allows native category support
    }

    if params:
        default_params.update(params)

    # Handle class imbalance (62% non-purchase vs 38% purchase)
    # scale_pos_weight = sum(negative) / sum(positive)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(1, pos_count)
    default_params["scale_pos_weight"] = scale_pos_weight
    logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(**default_params)

    # Fit with early stopping if validation set is provided
    if X_val is not None and y_val is not None:
        logger.info("Training with early stopping...")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=10,
        )
    else:
        logger.info("Training on full train split...")
        model.fit(X_train, y_train, verbose=True)

    # Save to disk
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {save_path}")

    return model


def evaluate_classifier(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate the classifier with standard binary metrics.

    Args:
        model: Trained XGBoost classifier.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        dict of computed metrics
    """
    logger.info("Evaluating model on test set...")
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    report = classification_report(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    print(f"ROC-AUC Score: {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("=" * 60 + "\n")

    return {
        "auc": auc,
        "confusion_matrix": cm,
    }


def load_classifier(path: str = "models/classifier_lsat.pkl") -> xgb.XGBClassifier:
    """
    Load a trained classifier from disk.

    Args:
        path: Path to the .pkl file.

    Returns:
        Trained XGBoost classifier.
    """
    logger.info(f"Loading classifier from {path}...")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 1. Prepare data
    X_train, X_test, y_train, y_test = prepare_classifier_dataset()

    # 2. Train model
    model = train_classifier(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        save_path="models/classifier_lsat.pkl"
    )

    # 3. Evaluate model
    evaluate_classifier(model, X_test, y_test)
