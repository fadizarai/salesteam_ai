"""
LAYER 3 — Services
Orchestrates the full recommendation pipeline.

Workflow:
1. Load pre-computed feature matrix from dataset (data/processed/training_set.csv)
2. Load XGBoost Classifier + XGBoost Regressor + Category LabelEncoder
3. Filter products for the requested client_id
4. Run predict_proba() -> purchase probabilities (classifier)
5. Filter by threshold / nb_suggestions
6. Run predict() -> suggested quantities (regressor, clip + round to int)
7. Generate human-readable French explanations
8. Return RecommendResponse
"""

import os
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from src.api.schemas import (
    RecommendRequest,
    RecommendResponse,
    ProductSuggestion,
    AIConfig,
)

logger = logging.getLogger(__name__)

MODEL_PATH = "src/models/classifier_lsat.joblib"
REGRESSOR_PATH = "src/models/regressor_lsat.joblib"
ENCODER_PATH = "src/models/encoder_categorie.joblib"
DATA_PATH = "data/processed/training_set.csv"

_model = None
_regressor = None
_encoder = None
_df_data = None


def _get_artifacts():
    global _model, _regressor, _encoder
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            # Fallback to models/ root
            alt_model = "models/classifier_lsat.joblib"
            alt_encoder = "models/encoder_categorie.joblib"
            if os.path.exists(alt_model):
                _model = joblib.load(alt_model)
                _encoder = joblib.load(alt_encoder)
            else:
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        else:
            _model = joblib.load(MODEL_PATH)
            _encoder = joblib.load(ENCODER_PATH)

    if _regressor is None:
        # Regressor is optional — graceful fallback to avg_qty if not found
        alt_regressor = "models/regressor_lsat.joblib"
        if os.path.exists(REGRESSOR_PATH):
            _regressor = joblib.load(REGRESSOR_PATH)
            logger.info("XGBoost Regressor loaded.")
        elif os.path.exists(alt_regressor):
            _regressor = joblib.load(alt_regressor)
            logger.info("XGBoost Regressor loaded from fallback path.")
        else:
            logger.warning(
                f"Regressor not found at {REGRESSOR_PATH}. "
                "Quantity will fall back to ceil(avg_qty). "
                "Run src/models/train_regressor.py to enable AI quantities."
            )
            _regressor = None

    return _model, _regressor, _encoder


def _get_dataset():
    global _df_data
    if _df_data is None:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
        logger.info(f"Loading dataset from {DATA_PATH}...")
        _df_data = pd.read_csv(DATA_PATH)
    return _df_data


def _clamp_prediction(pred: float, row: pd.Series) -> tuple[int, int, int]:
    """
    Clamp the regressor's raw quantity prediction to realistic historical bounds.

    Problem this solves:
        The raw XGBoost regressor output is unconstrained. Using naive ±25%
        around the prediction gives meaningless bounds like quantite_min=1 and
        quantite_max=228 for a product the client typically orders in batches of 20.

    Strategy:
        1. Derive hard bounds from the client's own purchase history:
               lower = max(1, hist_min × 0.5)   — floor: at least half the smallest order
               upper = hist_max × 2.0            — ceiling: at most double the largest order
        2. Build a confidence interval around the prediction:
               qty_min = max(lower, pred × 0.75)
               qty_max = min(upper, pred × 1.25)
        3. Clamp the suggestion itself within [lower, upper].

    Returns:
        (qty_suggested, qty_min, qty_max)
    """
    hist_avg = float(row.get("avg_qty", pred))
    hist_min = float(row.get("min_qty", 1.0))
    hist_max = float(row.get("max_qty", pred * 2))

    # Hard historical bounds
    lower = max(1, int(hist_min * 0.5))
    upper = max(lower + 1, int(hist_max * 2.0))

    # Confidence interval around prediction
    qty_min = max(lower, int(pred * 0.75))
    qty_max = min(upper, int(pred * 1.25))

    # Ensure min < max
    if qty_min >= qty_max:
        qty_max = qty_min + max(1, int(hist_avg * 0.25))

    # Clamp suggestion within historical bounds
    qty_suggested = max(lower, min(int(round(pred)), upper))

    return qty_suggested, qty_min, qty_max


def _compute_timing_boost(recency_rel: float) -> float:
    """Computes timing boost factor based on relative recency."""
    if recency_rel >= 1.5:
        return 3.0   # Very overdue
    elif recency_rel >= 1.0:
        return 2.0   # Overdue
    elif recency_rel >= 0.85:
        return 1.5   # Due soon
    else:
        return 1.0   # Not due yet


def get_available_clients(limit: int = None) -> list[dict]:
    """Returns a list of clients available in the dataset with metadata."""
    df = _get_dataset()

    agg_dict = {
        "total_products": ("code_article", "count"),
        "bought_products": ("target_bought", "sum"),
    }
    if "client_avg_basket_size" in df.columns:
        agg_dict["avg_basket"] = ("client_avg_basket_size", "first")
    if "has_gps" in df.columns:
        agg_dict["has_gps"] = ("has_gps", "first")

    client_counts = (
        df.groupby("code_client")
        .agg(**agg_dict)
        .reset_index()
    )

    if "avg_basket" not in client_counts.columns:
        client_counts["avg_basket"] = 1.0
    if "has_gps" not in client_counts.columns:
        client_counts["has_gps"] = False
    
    # Sort clients with active history first
    client_counts = client_counts.sort_values(
        by=["bought_products", "total_products"], ascending=False
    )
    if limit and limit > 0:
        client_counts = client_counts.head(limit)

    return client_counts.to_dict(orient="records")


def recommend(request: RecommendRequest) -> RecommendResponse:
    """
    Full recommendation pipeline: features -> predictions -> explanations.
    """
    model, regressor, encoder = _get_artifacts()
    df_all = _get_dataset()

    client_id = request.client_id
    df_client = df_all[df_all["code_client"] == client_id].copy()

    # Deduplicate: keep only the latest visit snapshot for each unique product
    if "visit_date" in df_client.columns:
        df_client["visit_date"] = pd.to_datetime(df_client["visit_date"])
        df_client = df_client.sort_values("visit_date").drop_duplicates(subset=["code_article"], keep="last")
    else:
        df_client = df_client.drop_duplicates(subset=["code_article"], keep="last")

    if df_client.empty:
        logger.warning(f"No history found for client_id='{client_id}'")
        return RecommendResponse(
            client_id=client_id,
            commercial_id=request.commercial_id,
            nb_suggestions=0,
            suggestions=[],
            generated_at=datetime.now().isoformat(),
        )

    # Category encoding
    df_client["categorie"] = df_client["categorie"].fillna("UNKNOWN").astype(str)
    known_classes = set(encoder.classes_)
    df_client["categorie_clean"] = df_client["categorie"].apply(
        lambda c: c if c in known_classes else "UNKNOWN"
    )
    df_client["categorie_encoded"] = encoder.transform(df_client["categorie_clean"])

    if "median_qty" not in df_client.columns:
        df_client["median_qty"] = df_client["avg_qty"]

    # Define explicit features for classifier and regressor to match training
    classifier_features = [
        "frequency",
        "total_qty",
        "avg_qty",
        "avg_delay_days",
        "recency_days",
        "recency_relative",
        "std_qty",
        "min_qty",
        "best_month",
        "avg_seasonal_coef"
    ]
    regressor_features = [
        "avg_qty",
        "median_qty",
        "std_qty",
        "min_qty",
        "max_qty",
        "last_qty",
        "frequency",
        "recency_days",
        "avg_delay_days",
        "current_month_coef",
        "avg_seasonal_coef"
    ]

    # ── Regressor: predict quantities for all rows now, reuse for candidates ──
    regressor_qty: np.ndarray | None = None
    if regressor is not None:
        try:
            X_reg = df_client[regressor_features].copy()
            for col in X_reg.select_dtypes(include=["bool"]).columns:
                X_reg[col] = X_reg[col].astype(int)
            
            raw_qty = regressor.predict(X_reg)
            regressor_qty = np.clip(np.round(raw_qty), 1, None).astype(int)
            df_client["regressor_qty"] = regressor_qty
        except Exception as exc:
            logger.warning(f"Regressor prediction failed ({exc}). Falling back to avg_qty.")
            regressor_qty = None

    # XGBoost Inference (Classifier)
    X_clf = df_client[classifier_features].copy()
    for col in X_clf.select_dtypes(include=["bool"]).columns:
        X_clf[col] = X_clf[col].astype(int)

    probabilities = model.predict_proba(X_clf)[:, 1]
    df_client["probabilite_achat"] = probabilities

    # ── Business Re-Ranking ──────────────────────────────────────────────────
    # The ML model gives a raw probability of purchase (ml_score).
    # We apply two multiplicative boosts on top:
    #
    # 1. timing_boost — based on recency_relative = recency_days / avg_delay_days
    #    If the client is at 85%+ of their normal reorder cycle, they are likely
    #    to need this product soon. If overdue (>1.0), the boost is stronger.
    #
    # 2. trend_boost — amplifies growing products, penalizes declining ones.
    #
    # final_score = ml_score × timing_boost × trend_boost
    # Candidates are then sorted by final_score (not raw ml_score).

    def _compute_final_score(row: pd.Series) -> float:
        ml_score = float(row["probabilite_achat"])
        recency_rel = float(row.get("recency_relative", 1.0))
        trend = float(row.get("trend", 0.0))

        timing_boost = _compute_timing_boost(recency_rel)

        # Trend boost
        if trend > 0.1:
            trend_boost = 1.2    # Growing demand
        elif trend < -0.2:
            trend_boost = 0.7    # Declining demand
        else:
            trend_boost = 1.0    # Stable

        return ml_score * timing_boost * trend_boost

    df_client["final_score"] = df_client.apply(_compute_final_score, axis=1)
    df_client["timing_boost"] = df_client.apply(
        lambda r: _compute_timing_boost(float(r.get("recency_relative", 1.0))),
        axis=1,
    )

    # Filter categories if config has filter_categories
    config = request.config or AIConfig()
    if config.filter_categories:
        df_client = df_client[df_client["categorie"].isin(config.filter_categories)]

    # Sort by final_score (business re-ranked), not raw ML probability
    ranked = df_client.sort_values(by="final_score", ascending=False)

    # ── Urgency-based Selection ──
    # 1. URGENT (⚡) : recency_relative >= 1.0 AND final_score > 0.80. Capped at 7.
    urgent_df = ranked[
        (ranked["recency_relative"] >= 1.0) & (ranked["final_score"] > 0.80)
    ].head(7).copy()
    urgent_df["urgency_group"] = "urgent"

    # 2. RECOMMANDÉ (✅) : final_score > 0.65. (Limit to 5)
    rec_candidates = ranked[~ranked["code_article"].isin(urgent_df["code_article"])]
    recommended_df = rec_candidates[rec_candidates["final_score"] > 0.65].head(5).copy()
    recommended_df["urgency_group"] = "recommande"

    # Combine candidates
    candidates = pd.concat([urgent_df, recommended_df])

    # Fallback to recommended head 5 if completely empty (e.g. newly initialized client)
    if candidates.empty:
        recommended_df = ranked.head(5).copy()
        recommended_df["urgency_group"] = "recommande"
        candidates = recommended_df

    suggestions = []
    for _, row in candidates.iterrows():
        prob = float(row["probabilite_achat"])
        recency = float(row.get("recency_days", 30))
        recency_rel = float(row.get("recency_relative", 1.0))
        freq = int(row.get("frequency", 1))
        avg_qty = float(row.get("avg_qty", 1.0))
        designation = str(row["designation"])
        cat = str(row["categorie"])
        urg_grp = str(row["urgency_group"])

        # ── Quantity: CV-based fallback with CV threshhead(7)old > 1.0
        std_qty = float(row.get("std_qty", 0.0))
        cv = std_qty / avg_qty if avg_qty > 0 else 999.0

        if cv > 1.0:
            raw_pred = float(np.ceil(avg_qty))
            qty_source = "historique (variance trop élevée)"
        elif "regressor_qty" in row and pd.notna(row["regressor_qty"]):
            raw_pred = float(row["regressor_qty"])
            qty_source = "IA"
        else:
            raw_pred = float(np.ceil(avg_qty))
            qty_source = "historique"

        # Clamp to realistic bounds
        sugg_qty, qty_min, qty_max = _clamp_prediction(raw_pred, row)

        is_new = bool(row.get("is_new_product", False))

        # Explanation generation
        from src.services.explanation import explain_suggestion
        explication = explain_suggestion(
            client_id=client_id,
            code_article=str(row["code_article"]),
            designation=designation,
            categorie=cat,
            quantite_suggeree=sugg_qty,
            score_confiance=prob,
            recency_days=int(recency),
            frequency=freq,
            trend=float(row.get("trend", 0)),
            is_new_product=is_new,
            qty_source=qty_source,
        )

        suggestions.append(
            ProductSuggestion(
                code_article=str(row["code_article"]),
                designation=designation,
                categorie=cat,
                quantite_suggeree=sugg_qty,
                quantite_min=qty_min,
                quantite_max=qty_max,
                source_quantite=qty_source,
                score_confiance=round(prob, 4),
                score_final=round(float(row.get("final_score", prob)), 4),
                timing_boost=round(float(row.get("timing_boost", 1.0)), 1),
                is_nouveau_produit=is_new,
                explication=explication,
                urgency_group=urg_grp,
                recency_relative=round(recency_rel, 2),
            )
        )

    return RecommendResponse(
        client_id=client_id,
        commercial_id=request.commercial_id,
        nb_suggestions=len(suggestions),
        suggestions=suggestions,
        generated_at=datetime.now().isoformat(),
    )


