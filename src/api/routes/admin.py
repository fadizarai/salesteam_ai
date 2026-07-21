"""
LAYER 4 — API
Admin endpoints : health check and model retraining trigger.
"""

import logging
from datetime import datetime
from fastapi import APIRouter
from src.api.schemas import HealthResponse
from src.models.train_classifier import run_training_pipeline

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check that the API is running correctly."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=True,
        timestamp=datetime.now().isoformat(),
    )


@router.post("/retrain")
async def retrain():
    """
    Trigger model retraining with latest data and feedback.
    To be called weekly or after significant feedback accumulation.
    """
    logger.info("Retraining requested via Admin API...")
    try:
        model, encoder = run_training_pipeline()
        return {
            "status": "success",
            "message": "Model retraining pipeline completed successfully.",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Retraining failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        }
