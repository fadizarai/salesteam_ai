"""
LAYER 3 — Services
Assembles all verifiable evidence for a given (client, product) pair
BEFORE any LLM call, so the language model can cite real numbers
instead of hallucinating them.

Usage
-----
    from src.services.deep_context import retrieve_deep_context

    ctx = retrieve_deep_context(
        client_id="CLT070730",
        code_article="25078RA3EABLACK4/128",
        recommendation_response=response.model_dump(),
    )

The returned dict is self-contained and can be serialised to JSON
or injected directly into a prompt template.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
_MAIN_TABLE_PATH = "data/processed/main_table.csv"
_TRAINING_SET_PATH = "data/processed/training_set.csv"

# Lazy-loaded singletons (reset between tests with _reset_cache())
_main_table: Optional[pd.DataFrame] = None
_training_set: Optional[pd.DataFrame] = None


def _reset_cache() -> None:
    """Invalidate the module-level DataFrames (useful in tests)."""
    global _main_table, _training_set
    _main_table = None
    _training_set = None


def _load_main_table() -> pd.DataFrame:
    global _main_table
    if _main_table is None:
        if not os.path.exists(_MAIN_TABLE_PATH):
            raise FileNotFoundError(
                f"main_table.csv not found at '{_MAIN_TABLE_PATH}'. "
                "Run the data pipeline first."
            )
        _main_table = pd.read_csv(_MAIN_TABLE_PATH, parse_dates=["date_commande"])
    return _main_table


def _load_training_set() -> pd.DataFrame:
    global _training_set
    if _training_set is None:
        if not os.path.exists(_TRAINING_SET_PATH):
            raise FileNotFoundError(
                f"training_set.csv not found at '{_TRAINING_SET_PATH}'."
            )
        _training_set = pd.read_csv(_TRAINING_SET_PATH)
    return _training_set


# ── Urgency threshold constants (must match recommendation.py) ────────────────
_URGENT_RECENCY_THRESHOLD = 1.0   # recency_relative >= 1.0
_URGENT_SCORE_THRESHOLD = 0.80    # final_score > 0.80
_RECOMMANDE_SCORE_THRESHOLD = 0.65  # final_score > 0.65

# ── Feature columns we extract from training_set ─────────────────────────────
_FEATURE_COLS = [
    "frequency",
    "avg_qty",
    "median_qty",
    "std_qty",
    "min_qty",
    "max_qty",
    "last_qty",
    "total_qty",
    "recency_days",
    "avg_delay_days",
    "recency_relative",
    "trend",
    # Seasonality — one or both may be present depending on the build
    "cat_quarterly_coef",
    "current_month_coef",
    "avg_seasonal_coef",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_suggestion(
    recommendation_response: dict,
    code_article: str,
) -> Optional[dict]:
    """Returns the matching suggestion dict from the response, or None."""
    for s in recommendation_response.get("suggestions", []):
        if s.get("code_article") == code_article:
            return s
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Real order history
# ─────────────────────────────────────────────────────────────────────────────

def _get_order_history(
    client_id: str,
    code_article: str,
    n: int = 6,
) -> dict:
    """
    Returns the last *n* individual orders from main_table.csv
    for this (client, product) pair, sorted by date ascending.

    On lookup failure, returns a dict with ``limited_history=True``.
    """
    try:
        df = _load_main_table()
    except FileNotFoundError as exc:
        logger.warning("main_table.csv unavailable: %s", exc)
        return {
            "limited_history": True,
            "reason": str(exc),
            "orders": [],
            "total_orders_found": 0,
        }

    mask = (df["code_client"] == client_id) & (df["code_article"] == code_article)
    sub = df[mask].sort_values("date_commande")

    if sub.empty:
        return {
            "limited_history": True,
            "reason": f"No orders found for client='{client_id}' / article='{code_article}'",
            "orders": [],
            "total_orders_found": 0,
        }

    total_found = len(sub)
    last_n = sub.tail(n)

    orders = [
        {
            "date": row["date_commande"].strftime("%Y-%m-%d"),
            "quantite": int(row["quantite"]),
            "code_facture": str(row.get("code_facture", "")),
        }
        for _, row in last_n.iterrows()
    ]

    return {
        "limited_history": False,
        "orders": orders,
        "total_orders_found": total_found,
        "showing_last_n": len(orders),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Exact feature values
# ─────────────────────────────────────────────────────────────────────────────

def _get_feature_values(
    client_id: str,
    code_article: str,
    recommendation_response: dict,
) -> dict:
    """
    Extracts the exact ML feature values for this (client, product) pair.

    Priority:
    1. The suggestion object already computed in recommend()
       (available as a field of recommendation_response).
    2. The training_set.csv row (latest snapshot).

    Returns a flat dict with all available feature names -> values,
    plus a computed cv (coefficient of variation).
    """
    features: dict = {}
    sugg = _find_suggestion(recommendation_response, code_article)

    if sugg:
        # Fields present in ProductSuggestion that double as feature proxies
        features["score_confiance"] = float(sugg.get("score_confiance", 0.0))
        features["score_final"] = float(sugg.get("score_final", 0.0))
        features["timing_boost"] = float(sugg.get("timing_boost", 1.0))
        features["recency_relative"] = float(sugg.get("recency_relative", 0.0))
        
        # Dynamic fields computed in recommend() based on visit_date
        for key in ["recency_days", "avg_delay_days", "trend", "frequency"]:
            if sugg.get(key) is not None:
                if isinstance(sugg[key], float):
                    features[key] = float(sugg[key])
                else:
                    features[key] = int(sugg[key])

    # Fall back / enrich from training_set
    try:
        ts = _load_training_set()
        mask = (ts["code_client"] == client_id) & (ts["code_article"] == code_article)
        rows = ts[mask]
        if not rows.empty:
            row = (
                rows.sort_values("visit_date").iloc[-1]
                if "visit_date" in rows.columns
                else rows.iloc[-1]
            )
            for col in _FEATURE_COLS:
                if col in row.index and col not in features:
                    val = row[col]
                    if pd.notna(val):
                        if isinstance(val, (float, np.floating)):
                            features[col] = float(val)
                        elif isinstance(val, (int, np.integer)):
                            features[col] = int(val)
                        else:
                            features[col] = val
    except Exception as exc:
        logger.warning(
            "Could not load training_set features for %s/%s: %s",
            client_id, code_article, exc,
        )

    # Compute CV
    avg_qty = features.get("avg_qty", 0.0)
    std_qty = features.get("std_qty", 0.0)
    if avg_qty and avg_qty > 0:
        features["cv"] = round(float(std_qty) / float(avg_qty), 4)
    else:
        features["cv"] = None

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Score decomposition
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_trend_boost(sugg: dict) -> Optional[float]:
    """
    Back-calculates trend_boost from:
        score_final = score_confiance x timing_boost x trend_boost
    Returns None if any value is missing or denominator is 0.
    """
    try:
        sf = float(sugg["score_final"])
        sc = float(sugg["score_confiance"])
        tb = float(sugg["timing_boost"])
        if sc > 0 and tb > 0:
            return round(sf / (sc * tb), 4)
    except (KeyError, TypeError, ZeroDivisionError):
        pass
    return None


def _get_score_decomposition(
    recommendation_response: dict,
    code_article: str,
) -> dict:
    """
    Extracts every scoring field from the already-computed suggestion object.
    Returns an empty dict with a not_found flag if the product is absent
    from the response (e.g. the product was not surfaced as a candidate).
    """
    sugg = _find_suggestion(recommendation_response, code_article)
    if sugg is None:
        return {"not_found": True}

    return {
        "not_found": False,
        "score_confiance": sugg.get("score_confiance"),
        "timing_boost": sugg.get("timing_boost"),
        # trend_boost is not stored directly in ProductSuggestion;
        # back-calculated from the score identity.
        "trend_boost_estimated": _estimate_trend_boost(sugg),
        "score_final": sugg.get("score_final"),
        "source_quantite": sugg.get("source_quantite"),
        "quantite_suggeree": sugg.get("quantite_suggeree"),
        "quantite_min": sugg.get("quantite_min"),
        "quantite_max": sugg.get("quantite_max"),
        "urgency_group": sugg.get("urgency_group"),
        "recency_relative": sugg.get("recency_relative"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Rank context
# ─────────────────────────────────────────────────────────────────────────────

def _get_rank_context(
    recommendation_response: dict,
    code_article: str,
) -> dict:
    """
    Computes the rank of this product within its urgency group,
    and returns the scores of its nearest neighbours in the global ranking.
    """
    suggestions = recommendation_response.get("suggestions", [])
    if not suggestions:
        return {"not_found": True}

    target_idx = next(
        (i for i, s in enumerate(suggestions) if s.get("code_article") == code_article),
        None,
    )
    if target_idx is None:
        return {"not_found": True}

    target = suggestions[target_idx]
    urgency_group = target.get("urgency_group", "unknown")

    # Rank within urgency group (already sorted by final_score DESC in recommend())
    same_group = [s for s in suggestions if s.get("urgency_group") == urgency_group]
    rank_in_group = next(
        (i + 1 for i, s in enumerate(same_group) if s.get("code_article") == code_article),
        None,
    )

    # Global neighbours
    above = suggestions[target_idx - 1] if target_idx > 0 else None
    below = suggestions[target_idx + 1] if target_idx < len(suggestions) - 1 else None

    def _summary(s: Optional[dict]) -> Optional[dict]:
        if s is None:
            return None
        return {
            "code_article": s.get("code_article"),
            "designation": s.get("designation"),
            "score_final": s.get("score_final"),
            "urgency_group": s.get("urgency_group"),
        }

    return {
        "not_found": False,
        "global_rank": target_idx + 1,
        "total_suggestions": len(suggestions),
        "rank_in_group": rank_in_group,
        "group_size": len(same_group),
        "urgency_group": urgency_group,
        "neighbour_above": _summary(above),
        "neighbour_below": _summary(below),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Urgency threshold analysis
# ─────────────────────────────────────────────────────────────────────────────

def _get_threshold_analysis(
    recommendation_response: dict,
    code_article: str,
) -> dict:
    """
    States explicitly which thresholds were crossed (or nearly missed)
    to classify this product as urgent or recommande, and by how much.
    """
    sugg = _find_suggestion(recommendation_response, code_article)
    if sugg is None:
        return {"not_found": True}

    score_final = float(sugg.get("score_final", 0.0))
    recency_rel = float(sugg.get("recency_relative", 0.0))
    urgency_group = str(sugg.get("urgency_group", "unknown"))

    result: dict = {
        "not_found": False,
        "urgency_group": urgency_group,
        "recency_relative": recency_rel,
        "score_final": score_final,
    }

    if urgency_group == "urgent":
        result["thresholds_met"] = ["recency_relative >= 1.0", "final_score > 0.80"]
        result["recency_margin"] = round(recency_rel - _URGENT_RECENCY_THRESHOLD, 4)
        result["score_margin"] = round(score_final - _URGENT_SCORE_THRESHOLD, 4)
        result["explanation"] = (
            f"Ce produit est URGENT car recency_relative={recency_rel:.2f} "
            f"(seuil 1.0, marge +{recency_rel - _URGENT_RECENCY_THRESHOLD:.2f}) "
            f"ET score_final={score_final:.3f} "
            f"(seuil 0.80, marge +{score_final - _URGENT_SCORE_THRESHOLD:.3f})."
        )
    elif urgency_group == "recommande":
        result["thresholds_met"] = ["final_score > 0.65"]
        result["score_margin"] = round(score_final - _RECOMMANDE_SCORE_THRESHOLD, 4)
        if recency_rel < _URGENT_RECENCY_THRESHOLD:
            note = f"Note : recency_relative={recency_rel:.2f} < 1.0, donc non classe URGENT."
        else:
            note = f"Note : score_final={score_final:.3f} <= 0.80, donc non classe URGENT malgre recency OK."
        result["explanation"] = (
            f"Ce produit est RECOMMANDE car score_final={score_final:.3f} "
            f"(seuil 0.65, marge +{score_final - _RECOMMANDE_SCORE_THRESHOLD:.3f}). {note}"
        )
    else:
        result["thresholds_met"] = []
        result["explanation"] = (
            f"Groupe inconnu '{urgency_group}'. Aucun seuil documente."
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_deep_context(
    client_id: str,
    code_article: str,
    recommendation_response: dict,
) -> dict:
    """
    Assembles all verifiable evidence for a single (client, product) pair
    from an already-generated recommendation response.

    Parameters
    ----------
    client_id : str
        The client identifier (e.g. "CLT070730").
    code_article : str
        The article code (e.g. "25078RA3EABLACK4/128").
    recommendation_response : dict
        The raw dict produced by recommend().model_dump() (or any
        equivalent dict matching the RecommendResponse schema).

    Returns
    -------
    dict with five top-level keys:

    order_history
        Real individual orders (date + quantity) from main_table.csv.
        Contains limited_history=True if the data is unavailable.

    feature_values
        Exact ML feature values used by the models for this pair,
        enriched with a computed cv field.

    score_decomposition
        Full score breakdown: score_confiance, timing_boost,
        trend_boost_estimated, score_final, source_quantite,
        quantity bounds, urgency_group.

    rank_context
        Rank of this product within its urgency group, global rank,
        and scores of the nearest neighbours in the overall ranking.

    threshold_analysis
        Explicit statement of which urgency thresholds were crossed,
        and by how much, with a human-readable French explanation.

    Notes
    -----
    * This function never raises on missing data — it returns partial
      results with descriptive limited_history / not_found flags.
    * It does NOT call any model or LLM — it only reads pre-computed
      artefacts (CSVs and the recommendation response itself).
    """
    logger.debug(
        "retrieve_deep_context called: client=%s article=%s",
        client_id,
        code_article,
    )

    return {
        "client_id": client_id,
        "code_article": code_article,
        "order_history": _get_order_history(client_id, code_article, n=6),
        "feature_values": _get_feature_values(client_id, code_article, recommendation_response),
        "score_decomposition": _get_score_decomposition(recommendation_response, code_article),
        "rank_context": _get_rank_context(recommendation_response, code_article),
        "threshold_analysis": _get_threshold_analysis(recommendation_response, code_article),
    }
