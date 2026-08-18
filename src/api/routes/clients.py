"""
LAYER 4 — API
GET /api/clients endpoint.
Returns list of available clients with statistics for UI dropdown / search.
"""

import logging
from fastapi import APIRouter, HTTPException
from src.services.recommendation import get_available_clients

from typing import Optional

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/clients")
async def list_clients(limit: Optional[int] = None):
    """
    Get all clients from the dataset with historical statistics.
    """
    try:
        clients = get_available_clients(limit=limit)
        return {"total": len(clients), "clients": clients}
    except Exception as e:
        logger.error(f"Error fetching clients: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
