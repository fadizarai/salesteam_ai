"""
LAYER 3 — Services
Saves sales rep feedback to CSV and prepares it for retraining.

Workflow:
1. Receive FeedbackRequest from /api/feedback route
2. Validate and enrich feedback items (add timestamp, date)
3. Append to data/feedback/feedback_YYYY-MM.csv (monthly files)
4. Compute acceptance rate and modification statistics
5. Return a summary for logging

The feedback CSV schema is:
    submitted_at, client_id, commercial_id, code_facture,
    code_article, accepte, quantite_finale, modifie,
    date_visite_mois

Retraining:
    load_feedback_for_retraining() reads all feedback CSVs,
    joins with feature matrix and returns a dataset suitable
    for fine-tuning the classifier and regressor.
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

FEEDBACK_DIR = "data/feedback"


def save_feedback(request: object) -> dict:
    """
    Save sales rep feedback from a FeedbackRequest to CSV.

    Appends to the monthly CSV file (creates it if not exists).
    One row per FeedbackItem in the request.

    Args:
        request: FeedbackRequest Pydantic model
                 (from src.api.schemas)

    Returns:
        Summary dict: {nb_items, nb_accepted, nb_modified,
                       acceptance_rate, file_path}
    """
    raise NotImplementedError("To be implemented")


def load_feedback_for_retraining(
    feedback_dir: str = FEEDBACK_DIR,
    min_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load all feedback CSV files and merge them into
    a single DataFrame for model retraining.

    Args:
        feedback_dir: directory containing feedback CSV files
        min_date: optional ISO date string to filter feedback
                  (only load feedback after this date)

    Returns:
        Combined DataFrame with all feedback records,
        sorted by submitted_at ascending
    """
    raise NotImplementedError("To be implemented")


def compute_feedback_stats(
    feedback_df: pd.DataFrame,
) -> dict:
    """
    Compute aggregate statistics over a feedback DataFrame.

    Stats computed:
    - total_feedbacks: total number of items
    - acceptance_rate: % of items accepted without modification
    - modification_rate: % of items accepted with modification
    - rejection_rate: % of items rejected
    - top_accepted_products: top 10 most accepted products
    - top_rejected_products: top 10 most rejected products

    Args:
        feedback_df: DataFrame from load_feedback_for_retraining()

    Returns:
        dict of stat name → value
    """
    raise NotImplementedError("To be implemented")


def prepare_retraining_dataset(
    feedback_df: pd.DataFrame,
    feature_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Join feedback with feature matrix to produce a
    training dataset for model fine-tuning.

    Label logic:
    - classifier label: accepte (1=accepted, 0=rejected)
    - regressor label:  quantite_finale (only for accepted items)

    Args:
        feedback_df: loaded feedback DataFrame
        feature_matrix: current feature matrix

    Returns:
        Tuple of (X, y_classifier, y_regressor)
    """
    raise NotImplementedError("To be implemented")
