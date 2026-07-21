"""
LAYER 3 — Services
Saves sales rep feedback to CSV and prepares it for retraining.

Workflow:
1. Receive FeedbackRequest from /api/feedback route
2. Validate and enrich feedback items (add timestamp, date)
3. Append to data/feedback/feedback_YYYY-MM.csv (monthly files)
4. Compute acceptance rate and modification statistics
5. Return a summary for logging
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd

from src.api.schemas import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)

FEEDBACK_DIR = Path("data/feedback")


def save_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Save sales rep feedback from a FeedbackRequest to CSV.
    Appends to the monthly CSV file (creates it if not exists).
    """
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    
    now = datetime.now()
    monthly_filename = f"feedback_{now.strftime('%Y-%m')}.csv"
    file_path = FEEDBACK_DIR / monthly_filename

    rows = []
    for item in request.items:
        rows.append({
            "submitted_at": request.submitted_at or now.isoformat(),
            "client_id": request.client_id,
            "commercial_id": request.commercial_id,
            "code_facture": request.code_facture or "",
            "code_article": item.code_article,
            "accepte": item.accepte,
            "quantite_finale": item.quantite_finale,
            "modifie": item.modifie,
            "date_visite_mois": now.strftime("%Y-%m"),
        })

    df_new = pd.DataFrame(rows)

    if file_path.exists():
        df_new.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(file_path, mode="w", header=True, index=False)

    logger.info(f"Saved {len(rows)} feedback items to {file_path}")

    return FeedbackResponse(
        status="ok",
        message=f"Feedback recorded successfully ({len(rows)} items)",
        nb_items=len(rows),
    )


def load_feedback_for_retraining(
    feedback_dir: str = str(FEEDBACK_DIR),
    min_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load all feedback CSV files and merge them into
    a single DataFrame for model retraining.
    """
    p = Path(feedback_dir)
    if not p.exists():
        return pd.DataFrame()

    csv_files = list(p.glob("feedback_*.csv"))
    if not csv_files:
        return pd.DataFrame()

    dfs = [pd.read_csv(f) for f in csv_files]
    combined_df = pd.concat(dfs, ignore_index=True)

    if min_date and "submitted_at" in combined_df.columns:
        combined_df = combined_df[combined_df["submitted_at"] >= min_date]

    return combined_df.sort_values(by="submitted_at", ascending=True)
