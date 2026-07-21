"""
LAYER 4 — API
POST /api/recommend endpoint.
Receives client_id from Flutter / React Frontend, returns order proposal.
"""

import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException

from src.api.schemas import RecommendRequest, RecommendResponse
from src.services.recommendation import recommend

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest):
    """
    Generate an automatic order proposal for a client visit.
    Returns list of product suggestions with quantities
    and French explanation.
    """
    logger.info(
        f"Recommendation requested: client={request.client_id} "
        f"commercial={request.commercial_id}"
    )
    try:
        result = recommend(request)
        return result
    except Exception as e:
        logger.error(f"Error generating recommendation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
