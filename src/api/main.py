"""
LAYER 4 — API
FastAPI application entry point.
Registers all routers and middleware.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from src.api.routes import recommend, feedback, admin, clients
from src.api.schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SalesTeam AI API starting...")
    yield
    logger.info("SalesTeam AI API stopped.")


app = FastAPI(
    title="SalesTeam AI",
    description="Intelligent order recommendation agent for sales reps.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend.router, prefix="/api", tags=["Recommendation"])
app.include_router(clients.router,   prefix="/api", tags=["Clients"])
app.include_router(feedback.router,  prefix="/api", tags=["Feedback"])
app.include_router(admin.router,     prefix="/api", tags=["Admin"])


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=True,
        timestamp=datetime.now().isoformat(),
    )
