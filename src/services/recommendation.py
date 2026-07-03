"""
LAYER 3 — Services
Orchestrates the full recommendation pipeline.

Workflow:
1. Load pre-computed feature matrix (or compute on-the-fly)
2. Call Predictor.predict() → raw suggestion list
3. Call ExplanationService.explain() → French explanations
4. Assemble and return a RecommendResponse

This service is called by the /api/recommend route.
It is the only layer that knows about both models and API schemas.
"""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def recommend(request: object) -> object:
    """
    Full recommendation pipeline: features → predictions → explanations.

    Args:
        request: RecommendRequest (from src.api.schemas)

    Returns:
        RecommendResponse (from src.api.schemas)
    """
    raise NotImplementedError("To be implemented")


def _load_client_features(
    client_id: str,
    features_path: str = "data/processed/feature_matrix.csv",
) -> object:
    """
    Load or filter the feature matrix for a specific client.

    Args:
        client_id: client code to filter on
        features_path: path to the pre-computed feature matrix CSV

    Returns:
        DataFrame filtered for the given client
    """
    raise NotImplementedError("To be implemented")


def _assemble_response(
    client_id: str,
    commercial_id: str,
    predictions: list[dict],
    config: object,
) -> object:
    """
    Assemble a RecommendResponse from raw prediction dicts
    and explanations.

    Args:
        client_id: client code
        commercial_id: sales rep code
        predictions: list of prediction dicts from Predictor
        config: AIConfig instance

    Returns:
        RecommendResponse Pydantic model
    """
    raise NotImplementedError("To be implemented")
