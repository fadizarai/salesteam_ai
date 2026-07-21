"""
LAYER 4 — API
POST /api/feedback endpoint.
Records the sales rep reaction after order validation.
"""

import logging
from fastapi import APIRouter, HTTPException

from src.api.schemas import FeedbackRequest, FeedbackResponse
from src.services.feedback import save_feedback

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/feedback", response_model=FeedbackResponse)
async def record_feedback(request: FeedbackRequest):
    """
    Record feedback from the sales rep.
    Stores which products were accepted, rejected or modified
    and the final quantities chosen.
    This data is saved to disk and will be used to retrain models weekly.
    """
    logger.info(
        f"Feedback received: client={request.client_id} "
        f"items={len(request.items)}"
    )
    try:
        response = save_feedback(request)
        return response
    except Exception as e:
        logger.error(f"Error saving feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
