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


def get_available_clients(limit: int = 50) -> list[dict]:
    """Returns a list of clients available in the dataset with metadata."""
    df = _get_dataset()
    client_counts = (
        df.groupby("code_client")
        .agg(
            total_products=("code_article", "count"),
            bought_products=("target_bought", "sum"),
            avg_basket=("client_avg_basket_size", "first"),
            has_gps=("has_gps", "first"),
        )
        .reset_index()
    )
    
    # Sort clients with active history first
    client_counts = client_counts.sort_values(
        by=["bought_products", "total_products"], ascending=False
    ).head(limit)

    return client_counts.to_dict(orient="records")


def recommend(request: RecommendRequest) -> RecommendResponse:
    """
    Full recommendation pipeline: features -> predictions -> explanations.
    """
    model, regressor, encoder = _get_artifacts()
    df_all = _get_dataset()

    client_id = request.client_id
    df_client = df_all[df_all["code_client"] == client_id].copy()

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

    # Prepare feature matrix X
    cols_to_drop = [
        "code_client",
        "code_article",
        "designation",
        "categorie",
        "categorie_clean",
        "target_qty",
        "target_bought",
    ]
    existing = [c for c in cols_to_drop if c in df_client.columns]
    X = df_client.drop(columns=existing)

    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(int)

    # ── Regressor: predict quantities for all rows now, reuse for candidates ──
    regressor_qty: np.ndarray | None = None
    if regressor is not None:
        try:
            raw_qty = regressor.predict(X)
            regressor_qty = np.clip(np.round(raw_qty), 1, None).astype(int)
            df_client["regressor_qty"] = regressor_qty
        except Exception as exc:
            logger.warning(f"Regressor prediction failed ({exc}). Falling back to avg_qty.")
            regressor_qty = None

    # XGBoost Inference
    probabilities = model.predict_proba(X)[:, 1]
    df_client["probabilite_achat"] = probabilities

    # Filter categories if config has filter_categories
    config = request.config or AIConfig()
    if config.filter_categories:
        df_client = df_client[df_client["categorie"].isin(config.filter_categories)]

    # Sort candidates by probability
    ranked = df_client.sort_values(by="probabilite_achat", ascending=False)
    
    # Filter candidates >= 0.30 probability threshold or fallback to top candidates
    candidates = ranked[ranked["probabilite_achat"] >= 0.30]
    if candidates.empty:
        candidates = ranked.head(config.nb_suggestions)
    else:
        candidates = candidates.head(config.nb_suggestions)

    suggestions = []
    for _, row in candidates.iterrows():
        prob = float(row["probabilite_achat"])
        recency = float(row.get("recency_days", 30))
        freq = int(row.get("frequency", 1))
        avg_qty = float(row.get("avg_qty", 1.0))
        designation = str(row["designation"])
        cat = str(row["categorie"])

        # ── Quantity: use regressor prediction if available, else avg_qty ──
        if "regressor_qty" in row and pd.notna(row["regressor_qty"]):
            sugg_qty = max(1, int(row["regressor_qty"]))
            qty_source = "IA"
        else:
            sugg_qty = max(1, int(np.ceil(avg_qty)))
            qty_source = "historique"

        # Explanation generation
        if prob >= 0.85:
            explication = (
                f"Produit phare ({cat}) — acheté {freq} fois par ce client. "
                f"Quantité suggérée par l'IA : {sugg_qty} unités (récence : {int(recency)}j)."
            )
        elif prob >= 0.60:
            explication = (
                f"Demande régulière observée. "
                f"Quantité recommandée ({qty_source}) : {sugg_qty} unités."
            )
        else:
            explication = (
                f"Proposition de réassort basée sur la saisonnalité et la catégorie {cat}. "
                f"Quantité estimée : {sugg_qty} unités."
            )

        is_new = bool(row.get("is_new_product", False))

        suggestions.append(
            ProductSuggestion(
                code_article=str(row["code_article"]),
                designation=designation,
                categorie=cat,
                quantite_suggeree=sugg_qty,
                quantite_min=1,
                quantite_max=max(sugg_qty * 3, 5),
                score_confiance=round(prob, 4),
                is_nouveau_produit=is_new,
                explication=explication,
            )
        )

    return RecommendResponse(
        client_id=client_id,
        commercial_id=request.commercial_id,
        nb_suggestions=len(suggestions),
        suggestions=suggestions,
        generated_at=datetime.now().isoformat(),
    )
