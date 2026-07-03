"""
LAYER 3 — Services
Generates French-language explanations for each product suggestion.

Strategy:
1. PRIMARY: Call HuggingFace Inference API (Mistral-7B-Instruct)
   with a structured prompt describing the suggestion context.
2. CACHE: Responses are cached for 24h (CACHE_TTL from .env)
   to avoid redundant API calls for identical contexts.
3. FALLBACK: If the API is unavailable or rate-limited,
   generate a rule-based template explanation in French.

Cache key = hash(client_id + code_article + month + category)
Cache storage = in-memory dict (reset on API restart)
"""

import logging
import hashlib
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# In-memory explanation cache: {cache_key: (explanation, expires_at)}
_explanation_cache: dict[str, tuple[str, datetime]] = {}


def explain_suggestion(
    client_id: str,
    code_article: str,
    designation: str,
    categorie: str,
    quantite_suggeree: int,
    score_confiance: float,
    recency_days: int,
    frequency: int,
    trend: float,
    is_new_product: bool,
    cache_ttl: int = 86400,
) -> str:
    """
    Generate a French explanation for a single product suggestion.

    Tries HuggingFace API first, falls back to rule-based template.

    Args:
        client_id: client code (for cache key)
        code_article: product code (for cache key)
        designation: product name
        categorie: product category
        quantite_suggeree: recommended quantity
        score_confiance: model confidence score [0, 1]
        recency_days: days since last order of this product
        frequency: number of times ordered historically
        trend: order trend (+/- ratio)
        is_new_product: whether this is a newly launched product
        cache_ttl: cache duration in seconds (default 24h)

    Returns:
        French explanation string (1-2 sentences)
    """
    raise NotImplementedError("To be implemented")


def _get_cache_key(
    client_id: str,
    code_article: str,
    month: int,
) -> str:
    """
    Build a deterministic cache key for the explanation.

    Args:
        client_id: client code
        code_article: product code
        month: current month number

    Returns:
        MD5 hex digest string
    """
    raise NotImplementedError("To be implemented")


def _call_huggingface_api(
    prompt: str,
    model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    max_tokens: int = 150,
) -> Optional[str]:
    """
    Call the HuggingFace Inference API to generate explanation text.

    Args:
        prompt: French instruction prompt for the model
        model: HuggingFace model ID
        max_tokens: maximum tokens to generate

    Returns:
        Generated text string, or None if API call fails
    """
    raise NotImplementedError("To be implemented")


def _rule_based_explanation(
    designation: str,
    quantite_suggeree: int,
    recency_days: int,
    frequency: int,
    trend: float,
    is_new_product: bool,
    score_confiance: float,
) -> str:
    """
    Fallback: generate a French explanation using simple rules
    when the HuggingFace API is unavailable.

    Rules applied:
    - is_new_product → new product launch template
    - trend > 0.2    → growing demand template
    - trend < -0.2   → declining (lower quantity) template
    - recency < 30   → frequent buyer template
    - default        → standard reorder template

    Args:
        designation: product name
        quantite_suggeree: recommended quantity
        recency_days: days since last order
        frequency: number of past orders
        trend: order trend ratio
        is_new_product: new product flag
        score_confiance: model confidence

    Returns:
        French explanation string
    """
    raise NotImplementedError("To be implemented")


def clear_cache() -> int:
    """
    Clear all cached explanations.
    Called by the /api/retrain endpoint after model retraining.

    Returns:
        Number of cache entries cleared
    """
    raise NotImplementedError("To be implemented")
