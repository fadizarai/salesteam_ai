"""
LAYER 2 — AI / Machine Learning
Unified prediction interface that orchestrates all models.

This is the single entry point for making predictions.
It coordinates:
1. XGBoost classifier  → which products to recommend
2. XGBoost regressor   → optimal quantity for each product
3. SVD model           → collaborative filtering boost
4. Cold start          → fallback for new products/clients

Input : client_id + AIConfig
Output: ranked list of ProductSuggestion dicts
        (before explanation — see services/explanation.py)

Decision logic:
- If client has history    → classifier + regressor + SVD
- If client has no history → cold_start_new_client()
- If product is new        → cold_start_new_product()
- SVD score is blended with classifier score (weighted average)
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Predictor:
    """
    Unified prediction interface for all recommendation models.
    Loads all models once at API startup and keeps them in memory.
    """

    def __init__(
        self,
        models_path: str = "models",
        features_path: str = "data/processed/feature_matrix.csv",
    ):
        """
        Initialize the predictor by loading all trained models
        and the pre-computed feature matrix.

        Args:
            models_path: directory containing .pkl model files
            features_path: path to the feature matrix CSV
        """
        raise NotImplementedError("To be implemented")

    def load_models(self) -> None:
        """
        Load all models from disk into memory.
        Called once at API startup via FastAPI lifespan.

        Models loaded:
        - classifier_lsat.pkl
        - regressor_lsat.pkl
        - svd_lsat.pkl
        """
        raise NotImplementedError("To be implemented")

    def predict(
        self,
        client_id: str,
        config: dict,
        visit_date: Optional[str] = None,
    ) -> list[dict]:
        """
        Main prediction method. Orchestrates all models.

        Args:
            client_id: client code (e.g. "CLT011712")
            config: AIConfig dict (which parameters to use)
            visit_date: optional visit date string (YYYY-MM-DD)
                        defaults to today

        Returns:
            List of dicts, one per suggested product:
            {
                code_article, designation, categorie,
                quantite_suggeree, quantite_min, quantite_max,
                score_confiance, is_nouveau_produit
            }
            Sorted by score_confiance descending.
            Limited to config.nb_suggestions items.
        """
        raise NotImplementedError("To be implemented")

    def _blend_scores(
        self,
        classifier_score: float,
        svd_score: float,
        classifier_weight: float = 0.6,
        svd_weight: float = 0.4,
    ) -> float:
        """
        Blend classifier and SVD scores into a single
        confidence score.

        Args:
            classifier_score: XGBoost predicted probability [0, 1]
            svd_score: SVD predicted rating (normalized to [0, 1])
            classifier_weight: weight for classifier score
            svd_weight: weight for SVD score

        Returns:
            Blended score in [0, 1]
        """
        raise NotImplementedError("To be implemented")

    def _apply_seasonality(
        self,
        suggestions: list[dict],
        visit_month: int,
    ) -> list[dict]:
        """
        Adjust suggested quantities by the seasonal coefficient
        for the visit month.

        Args:
            suggestions: list of suggestion dicts
            visit_month: month number (1-12)

        Returns:
            Updated suggestions with seasonality-adjusted quantities
        """
        raise NotImplementedError("To be implemented")
