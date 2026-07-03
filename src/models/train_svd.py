"""
LAYER 2 — AI / Machine Learning
Train SVD (Singular Value Decomposition) collaborative filtering
model to discover "clients who bought A also bought B" patterns.

Uses the scikit-surprise library (SVD algorithm).
Builds a client × product interaction matrix from order history.

Input  : cleaned main table (client, product, quantity)
Output : trained SVD model + encoded index mappings
Saved  : models/svd_lsat.pkl
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def build_interaction_matrix(
    df: pd.DataFrame,
    rating_col: str = "quantite",
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Build the client × product interaction matrix.
    Each cell = total quantity ordered (or normalized value).

    Args:
        df: cleaned main table
        rating_col: column to use as rating (default: quantite)
        normalize: if True, apply log1p normalization to ratings

    Returns:
        Pivot table (clients as rows, products as columns)
    """
    raise NotImplementedError("To be implemented")


def train_svd(
    df: pd.DataFrame,
    n_factors: int = 50,
    n_epochs: int = 20,
    save_path: str = "models/svd_lsat.pkl",
) -> object:
    """
    Train the SVD collaborative filtering model.

    Args:
        df: cleaned main table
        n_factors: number of latent factors
        n_epochs: number of training epochs
        save_path: path to save the trained model

    Returns:
        Trained SVD model (scikit-surprise)
    """
    raise NotImplementedError("To be implemented")


def get_svd_recommendations(
    model: object,
    client_id: str,
    all_products: list,
    ordered_products: list,
    top_n: int = 10,
) -> list[dict]:
    """
    Generate top-N product recommendations for a client
    using the trained SVD model.

    Filters out products the client already ordered recently.

    Args:
        model: trained SVD model
        client_id: target client code
        all_products: list of all known product codes
        ordered_products: products already ordered by client
                          (to exclude from recommendations)
        top_n: number of recommendations to return

    Returns:
        List of dicts: [{code_article, score}, ...]
        sorted by score descending
    """
    raise NotImplementedError("To be implemented")


def load_svd(path: str = "models/svd_lsat.pkl") -> object:
    """
    Load a trained SVD model from disk.

    Args:
        path: path to the .pkl file

    Returns:
        Trained SVD model
    """
    raise NotImplementedError("To be implemented")
