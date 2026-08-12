"""
LAYER 2 — AI / Machine Learning
Builds the target variable for training using a visit-level approach.

WHY VISIT-LEVEL ?
-----------------
The naive approach (positive = any pair that ever appeared in history,
negative = any pair that never appeared) leaks the label directly into
the features: frequency=0 for all negatives, frequency>0 for all positives.
The model never learns anything meaningful — it just checks frequency > 0.

The visit-level approach fixes this:
  For each client invoice (visit), we ask:
  "Given what this client ordered BEFORE this visit,
   which of their historically-known products did they choose
   to reorder this time?"

  - target = 1 : product was ordered on THIS invoice
  - target = 0 : product was in history but skipped this visit

  Features are computed from data strictly BEFORE the visit date,
  so there is zero temporal leakage.

This produces realistic positive rates (~20-40%) and makes features
like recency_relative, trend, avg_delay_days actually discriminative.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum number of prior orders for a client before we include their visit
MIN_HISTORY_ORDERS = 3

# Minimum number of times a product must appear in a client's history
# before it is considered a "known" candidate product
MIN_PRODUCT_APPEARANCES = 1


def _compute_product_features(
    history: pd.DataFrame,
    client_id: str,
    product_id: str,
    visit_date: pd.Timestamp,
) -> dict:
    """
    Compute all features for a (client, product) pair from history
    strictly before visit_date.
    """
    prod_hist = history[history["code_article"] == product_id].copy()

    if prod_hist.empty:
        return None

    qtys = prod_hist["quantite"].values
    dates = prod_hist["date_commande"].values

    avg_qty = float(np.mean(qtys))
    std_qty = float(np.std(qtys)) if len(qtys) > 1 else 0.0
    min_qty = float(np.min(qtys))
    max_qty = float(np.max(qtys))
    total_qty = float(np.sum(qtys))
    frequency = int(len(qtys))
    last_qty = float(qtys[-1])

    last_date = pd.Timestamp(dates[-1])
    recency_days = float((visit_date - last_date).days)

    # Inter-order delay
    if len(dates) >= 2:
        diffs = np.diff([pd.Timestamp(d) for d in dates])
        delay_days = [d.days for d in diffs]
        avg_delay_days = float(np.mean(delay_days))
    else:
        avg_delay_days = 30.0

    avg_delay_days = max(avg_delay_days, 1.0)
    recency_relative = recency_days / avg_delay_days

    # Trend: compare first half vs second half of order history
    n = len(qtys)
    if n >= 2:
        mid = n // 2
        old_avg = np.mean(qtys[:mid]) if mid > 0 else 0
        new_avg = np.mean(qtys[mid:]) if (n - mid) > 0 else 0
        trend = float((new_avg - old_avg) / (old_avg + 1e-6))
    else:
        trend = 0.0

    # Seasonal coefficient (monthly)
    SEASONAL_COEF = {
        1: 0.85, 2: 0.90, 3: 1.10, 4: 1.20, 5: 1.00, 6: 1.05,
        7: 1.25, 8: 1.20, 9: 1.15, 10: 1.00, 11: 1.10, 12: 1.30,
    }
    monthly_qtys = prod_hist.groupby(prod_hist["date_commande"].dt.month)["quantite"].mean()
    avg_seasonal_coef = float(
        monthly_qtys.index.map(SEASONAL_COEF).values @ monthly_qtys.values / max(monthly_qtys.sum(), 1)
    ) if len(monthly_qtys) > 0 else 1.0

    best_month = int(monthly_qtys.idxmax()) if len(monthly_qtys) > 0 else -1
    current_month_coef = SEASONAL_COEF.get(visit_date.month, 1.0)

    return {
        "code_client": client_id,
        "code_article": product_id,
        "categorie": prod_hist["categorie"].iloc[0] if "categorie" in prod_hist.columns and not prod_hist["categorie"].empty else "UNKNOWN",
        "designation": prod_hist["designation"].iloc[0] if "designation" in prod_hist.columns and not prod_hist["designation"].empty else "UNKNOWN",
        "avg_qty": avg_qty,
        "std_qty": std_qty,
        "min_qty": min_qty,
        "max_qty": max_qty,
        "total_qty": total_qty,
        "frequency": frequency,
        "last_qty": last_qty,
        "recency_days": recency_days,
        "avg_delay_days": avg_delay_days,
        "recency_relative": recency_relative,
        "trend": trend,
        "avg_seasonal_coef": avg_seasonal_coef,
        "best_month": best_month,
        "current_month_coef": current_month_coef,
        "visit_date": visit_date,
    }


def build_visit_level_dataset(
    df: pd.DataFrame,
    min_history_orders: int = MIN_HISTORY_ORDERS,
    min_product_appearances: int = MIN_PRODUCT_APPEARANCES,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Build a visit-level training dataset where each row represents
    one (client, product) candidate at the time of a specific visit.

    Args:
        df: main_table with columns including date_commande,
            code_client, code_article, code_facture, quantite
        min_history_orders: minimum prior rows for a client to be included
        min_product_appearances: min times a product must appear in client
            history before being treated as a candidate
        save_path: optional CSV path to save the result

    Returns:
        DataFrame with one row per (client, product, visit) with features
        computed from prior history and target_bought column.
    """
    df = df.copy()
    df["date_commande"] = pd.to_datetime(df["date_commande"])
    df = df.sort_values(["code_client", "date_commande"]).reset_index(drop=True)

    logger.info(f"Building visit-level dataset from {len(df)} rows...")

    # Get one row per invoice (unique visit)
    invoices = (
        df.drop_duplicates(subset=["code_facture"])
        [["code_facture", "code_client", "date_commande"]]
        .sort_values("date_commande")
        .reset_index(drop=True)
    )
    logger.info(f"Total invoices to process: {len(invoices)}")

    rows = []
    skipped_no_history = 0
    skipped_no_candidates = 0

    for i, invoice in invoices.iterrows():
        if i % 500 == 0:
            logger.info(f"  Processing invoice {i}/{len(invoices)} ({len(rows)} rows built so far)...")

        client_id = invoice["code_client"]
        visit_date = invoice["date_commande"]
        facture_id = invoice["code_facture"]

        # History = all orders by this client strictly before this visit
        history = df[
            (df["code_client"] == client_id) &
            (df["date_commande"] < visit_date)
        ]

        if len(history) < min_history_orders:
            skipped_no_history += 1
            continue

        # Products ordered on THIS visit
        ordered_this_visit = set(
            df[df["code_facture"] == facture_id]["code_article"].tolist()
        )

        # Known products = products this client has ordered before,
        # appearing at least min_product_appearances times
        product_counts = history.groupby("code_article").size()
        known_products = product_counts[
            product_counts >= min_product_appearances
        ].index.tolist()

        if not known_products:
            skipped_no_candidates += 1
            continue

        # Build one row per known product
        for product_id in known_products:
            features = _compute_product_features(
                history, client_id, product_id, visit_date
            )
            if features is None:
                continue

            features["target_bought"] = 1 if product_id in ordered_this_visit else 0
            features["target_qty"] = 0.0

            # For positives, get the actual quantity ordered this visit
            if features["target_bought"] == 1:
                visit_rows = df[
                    (df["code_facture"] == facture_id) &
                    (df["code_article"] == product_id)
                ]
                if not visit_rows.empty:
                    features["target_qty"] = float(visit_rows["quantite"].sum())

            rows.append(features)

    logger.info(
        f"Done. Built {len(rows)} rows. "
        f"Skipped {skipped_no_history} visits (not enough history), "
        f"{skipped_no_candidates} visits (no known candidates)."
    )

    result = pd.DataFrame(rows)

    pos_rate = result["target_bought"].mean() * 100
    logger.info(
        f"Dataset shape: {result.shape} | "
        f"Positive rate: {pos_rate:.1f}% "
        f"({result['target_bought'].sum()} positives / {len(result)} total)"
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(save_path, index=False)
        logger.info(f"Saved visit-level training set to {save_path}")

    return result


if __name__ == "__main__":
    import os
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    main_table_path = "data/processed/main_table.csv"
    output_path = "data/processed/training_set.csv"

    logger.info(f"Loading main_table from {main_table_path}...")
    df = pd.read_csv(main_table_path, parse_dates=["date_commande"])
    logger.info(f"Loaded {len(df)} rows. Date range: {df['date_commande'].min()} → {df['date_commande'].max()}")

    training_set = build_visit_level_dataset(
        df,
        min_history_orders=MIN_HISTORY_ORDERS,
        min_product_appearances=MIN_PRODUCT_APPEARANCES,
        save_path=output_path,
    )

    print("\n" + "=" * 60)
    print("   VISIT-LEVEL TRAINING SET SUMMARY")
    print("=" * 60)
    print(f"  Total rows         : {len(training_set)}")
    print(f"  Unique clients     : {training_set['code_client'].nunique()}")
    print(f"  Unique products    : {training_set['code_article'].nunique()}")
    print(f"  Positive rate      : {training_set['target_bought'].mean()*100:.1f}%")
    print(f"  Date range         : {training_set['visit_date'].min()} -> {training_set['visit_date'].max()}")
    print(f"  Columns            : {list(training_set.columns)}")
    print("=" * 60)