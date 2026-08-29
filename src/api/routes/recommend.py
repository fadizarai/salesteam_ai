"""
LAYER 4 — API
POST /api/recommend           — Fast bulk recommendation for a client visit.
POST /api/explain-detailed    — On-demand detailed explanation for ONE product card.
                                Never called automatically; only when the user
                                explicitly clicks a card in the frontend.
"""

import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    RecommendRequest,
    RecommendResponse,
    DetailedExplanationRequest,
    DetailedExplanationResponse,
)
from src.services.recommendation import recommend, get_detailed_explanation

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest):
    """
    Generate an automatic order proposal for a client visit.
    Returns list of product suggestions with quantities
    and French explanation.

    Side effect: caches the response internally so that a subsequent
    POST /api/explain-detailed call for the same client can reuse it
    without re-running the ML pipeline.
    """
    logger.info(
        f"Recommendation requested: client={request.client_id} "
        f"commercial={request.commercial_id}"
    )
    try:
        result = recommend(request, _skip_llm=True)
        return result
    except Exception as e:
        logger.error(f"Error generating recommendation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain-detailed", response_model=DetailedExplanationResponse)
async def explain_detailed(request: DetailedExplanationRequest):
    """
    Generate a detailed, 4-part French explanation for a single product card.

    **When to call this endpoint**
    - ONLY when the user explicitly clicks on a specific product card in the UI.
    - NEVER call this automatically for all suggestions returned by /api/recommend.

    **How it works**
    1. Reuses the last recommend() response cached for this client_id (no ML re-run).
    2. If the cache is cold (first visit), runs recommend() internally to warm it.
    3. Calls retrieve_deep_context() to assemble all verifiable evidence.
    4. Calls explain_suggestion_detailed() → LLM (500 tokens, temp 0.4)
       with rule-based fallback if the API is unavailable.

    **Response time**
    Typically 3-8 seconds (LLM call). The /api/recommend response is unaffected.
    """
    logger.info(
        "Detailed explanation requested: client=%s article=%s",
        request.client_id,
        request.code_article,
    )
    try:
        explication = get_detailed_explanation(
            client_id=request.client_id,
            code_article=request.code_article,
        )
        return DetailedExplanationResponse(
            client_id=request.client_id,
            code_article=request.code_article,
            explication_detaillee=explication,
            generated_at=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(
            "Error generating detailed explanation for client=%s article=%s: %s",
            request.client_id, request.code_article, e,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))

