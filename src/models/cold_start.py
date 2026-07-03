"""
LAYER 2 — AI / Machine Learning
Handle cold start scenarios: new products with no order history
or new clients with no purchase history.

Strategy:
1. For new PRODUCTS: content-based filtering
   - Use product category, designation embeddings
   - Find similar products that do have history
   - Propagate recommendations from similar products
2. For new CLIENTS (no history): geographic KNN
   - Find the K nearest clients with GPS coordinates
   - Aggregate their top products as suggestions
3. Combined: new product + client with GPS
   - Intersect category-similar products with
     neighborhood-popular products

Dependencies:
- sklearn BallTree for geographic KNN
- sentence-transformers for product embedding (optional)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def find_similar_products_by_category(
    target_product: str,
    product_features: pd.DataFrame,
    top_n: int = 5,
) -> list[str]:
    """
    Find products similar to the target product
    based on category matching.

    Args:
        target_product: code_article of the new product
        product_features: product features DataFrame
                          from compute_product_features()
        top_n: number of similar products to return

    Returns:
        List of similar code_article codes
    """
    raise NotImplementedError("To be implemented")


def find_nearest_clients(
    target_client: str,
    geo_features: pd.DataFrame,
    top_k: int = 5,
    max_distance_km: float = 10.0,
) -> list[str]:
    """
    Find the K nearest clients to the target client
    using haversine distance on GPS coordinates.

    Args:
        target_client: code_client of the target
        geo_features: geo features from compute_geo_features()
        top_k: maximum number of neighbors to return
        max_distance_km: maximum distance radius in km

    Returns:
        List of nearest client codes (excluding target)
    """
    raise NotImplementedError("To be implemented")


def cold_start_new_product(
    target_product: str,
    target_client: str,
    product_features: pd.DataFrame,
    history_features: pd.DataFrame,
    top_n: int = 3,
) -> list[dict]:
    """
    Generate suggestions for a new product with no order history.

    Uses products from the same category that have history
    to estimate likely quantity and confidence score.

    Args:
        target_product: code_article of the new product
        target_client: code_client requesting recommendation
        product_features: product features DataFrame
        history_features: order history features DataFrame
        top_n: max number of quantity estimates to average

    Returns:
        dict: {
            code_article, quantite_suggeree,
            score_confiance, is_nouveau_produit=True,
            source="cold_start_category"
        }
    """
    raise NotImplementedError("To be implemented")


def cold_start_new_client(
    target_client: str,
    geo_features: pd.DataFrame,
    history_features: pd.DataFrame,
    top_n: int = 5,
) -> list[dict]:
    """
    Generate suggestions for a client with no purchase history
    by aggregating top products from geographic neighbors.

    Args:
        target_client: code_client with no history
        geo_features: geo features from compute_geo_features()
        history_features: order history features DataFrame
        top_n: number of top products to suggest

    Returns:
        List of dicts: [{code_article, quantite_suggeree,
                         score_confiance, source="cold_start_geo"}, ...]
    """
    raise NotImplementedError("To be implemented")
