"""
LAYER 4 — API
POST /api/recommend endpoint.
Receives client_id from Flutter, returns order proposal.
"""

import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from src.api.schemas import RecommendRequest, RecommendResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest):
    """
    Generate an automatic order proposal for a client visit.
    Flutter sends client_id + optional AI config.
    Returns list of product suggestions with quantities
    and French explanation.
    """
    logger.info(
        f"Recommendation requested: client={request.client_id} "
        f"commercial={request.commercial_id}"
    )
    try:
        # TODO: call recommendation service
        # from src.services.recommendation import recommend
        # result = recommend(request)
        # return result
        raise NotImplementedError(
            "Recommendation service not yet implemented. "
            "Will be connected in next step."
        )
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
