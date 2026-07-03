"""
LAYER 4 — API
POST /api/feedback endpoint.
Records the sales rep reaction after order validation.
"""

import logging
from fastapi import APIRouter, HTTPException
from src.api.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/feedback", response_model=FeedbackResponse)
async def record_feedback(request: FeedbackRequest):
    """
    Record feedback from the sales rep.
    Stores which products were accepted, rejected or modified
    and the final quantities chosen.
    This data will be used to retrain models weekly.
    """
    logger.info(
        f"Feedback received: client={request.client_id} "
        f"items={len(request.items)}"
    )
    try:
        # TODO: call feedback service
        # from src.services.feedback import save_feedback
        # save_feedback(request)
        raise NotImplementedError(
            "Feedback service not yet implemented."
        )
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
