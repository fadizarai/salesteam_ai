"""
LAYER 4 — API
Admin endpoints : health check and model retraining trigger.
"""

import logging
from datetime import datetime
from fastapi import APIRouter
from src.api.schemas import HealthResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check that the API is running correctly."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=False,
        timestamp=datetime.now().isoformat(),
    )


@router.post("/retrain")
async def retrain():
    """
    Trigger model retraining with latest data and feedback.
    To be called weekly or after significant feedback accumulation.
    """
    logger.info("Retraining requested...")
    # TODO: trigger retraining pipeline
    return {
        "status": "scheduled",
        "message": "Retraining will be implemented in next step.",
    }
