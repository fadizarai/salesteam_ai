"""
LAYER 2 — AI / Machine Learning
Transforms raw cleaned data into ML features.
These features are the 6 parameters of the AI agent.
NO model training here — only feature computation.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Seasonal coefficients by month (Tunisia context)
SEASONAL_COEF = {
    1: 0.85,   # January   - slow post holidays
    2: 0.90,   # February  - normal
    3: 1.10,   # March     - Ramadan starts
    4: 1.20,   # April     - Ramadan / Eid peak
    5: 1.00,   # May       - normal
    6: 1.05,   # June      - summer starts
    7: 1.25,   # July      - summer peak
    8: 1.20,   # August    - summer peak
    9: 1.15,   # September - back to school
    10: 1.00,  # October   - normal
    11: 1.10,  # November  - pre holidays
    12: 1.30,  # December  - year end peak
}


def build_feature_matrix(
    df: pd.DataFrame,
    reference_date: pd.Timestamp = None,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Build complete feature matrix for ML models.
    One row per (code_client, code_article) pair.
    Input  : cleaned main table (187,022 rows)
    Output : feature matrix (N pairs × 27 features)
    """
    logger.info("Starting build_feature_matrix...")

    # Ensure date_commande is datetime
    df = df.copy()
    df["date_commande"] = pd.to_datetime(df["date_commande"])

    # Define reference date
    if reference_date is None:
        reference_date = pd.to_datetime("2026-06-22")
    else:
        reference_date = pd.to_datetime(reference_date)

    # IMPORTANT: sort by ['code_client','code_article', 'date_commande'] before all computations.
    df = df.sort_values(by=["code_client", "code_article", "date_commande"]).copy()

    # Create helper columns
    # Use .shift(1) inside groupby for lag features to avoid data leakage
    df["quantite_lag1"] = df.groupby(["code_client", "code_article"])["quantite"].shift(1)
    df["date_diff"] = df.groupby(["code_client", "code_article"])["date_commande"].diff().dt.days

    # GROUP 1: Order history (per client, article pair)
    logger.info("Computing Group 1 features (Order history)...")
    grp = df.groupby(["code_client", "code_article"])

    g1 = pd.DataFrame()
    g1["avg_qty"] = grp["quantite"].mean()
    g1["std_qty"] = grp["quantite"].std().fillna(0.0)
    g1["min_qty"] = grp["quantite"].min()
    g1["max_qty"] = grp["quantite"].max()
    g1["total_qty"] = grp["quantite"].sum()
    g1["frequency"] = grp["quantite"].count()

    # last_qty: quantity of the most recent order (using shift(1) to avoid leakage)
    g1["last_qty"] = grp["quantite_lag1"].last().fillna(0.0)

    # recency_days
    last_date = grp["date_commande"].max()
    g1["recency_days"] = (reference_date - last_date).dt.days

    # avg_delay_days
    g1["avg_delay_days"] = grp["date_diff"].mean().fillna(30.0)

    # trend
    def compute_trend(series):
        n = len(series)
        if n < 4:
            return 0.0
        mid = n // 2
        old_avg = series.iloc[:mid].mean()
        new_avg = series.iloc[mid:].mean()
        return (new_avg - old_avg) / (old_avg + 1e-6)

    g1["trend"] = grp["quantite"].apply(compute_trend)
    g1 = g1.reset_index()

    # GROUP 2: Seasonality (per client, article pair)
    logger.info("Computing Group 2 features (Seasonality)...")
    df["month_coef"] = df["mois"].map(SEASONAL_COEF)
    
    g2 = pd.DataFrame()
    g2_grp = df.groupby(["code_client", "code_article"])
    g2["avg_seasonal_coef"] = g2_grp["month_coef"].mean()

    # best_month: month with highest average quantity
    monthly_means = df.groupby(["code_client", "code_article", "mois"])["quantite"].mean().reset_index()
    best_months = monthly_means.sort_values(by="quantite", ascending=False).drop_duplicates(
        subset=["code_client", "code_article"]
    ).rename(columns={"mois": "best_month"})[["code_client", "code_article", "best_month"]]
    
    g2 = g2.reset_index().merge(best_months, on=["code_client", "code_article"], how="left")
    g2["current_month_coef"] = SEASONAL_COEF[reference_date.month]

    # GROUP 3: Geography (per client)
    logger.info("Computing Group 3 features (Geography)...")
    g3 = df.groupby("code_client").agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first")
    ).reset_index()
    g3["has_gps"] = g3["latitude"].notna()

    # GROUP 4: Product info (per article)
    logger.info("Computing Group 4 features (Product info)...")
    prod_cat = df.groupby("code_article")["categorie"].agg(
        lambda x: x.mode().iloc[0] if len(x) > 0 and not x.mode().empty else "UNKNOWN"
    ).reset_index()

    g4 = df.groupby("code_article").agg(
        designation=("designation", "first"),
        is_bulk_product=("is_bulk_order", "any"),
        nb_clients=("code_client", "nunique"),
        first_order_date=("date_commande", "min")
    ).reset_index()
    
    g4 = g4.merge(prod_cat, on="code_article", how="left")
    g4["days_since_first_order"] = (reference_date - g4["first_order_date"]).dt.days
    g4["is_new_product"] = g4["days_since_first_order"] <= 90
    g4 = g4.drop(columns=["first_order_date"])

    # GROUP 5: Client profile (per client)
    logger.info("Computing Group 5 features (Client profile)...")
    g5 = df.groupby("code_client").agg(
        client_total_products=("code_article", "nunique"),
        client_total_invoices=("code_facture", "nunique")
    ).reset_index()

    basket_sizes = df.groupby(["code_client", "code_facture"])["code_article"].nunique().reset_index()
    client_basket = basket_sizes.groupby("code_client")["code_article"].mean().reset_index().rename(
        columns={"code_article": "client_avg_basket_size"}
    )
    g5 = g5.merge(client_basket, on="code_client", how="left")

    # MERGE ALL FEATURE GROUPS
    logger.info("Merging all feature groups...")
    features = g1.merge(g2, on=["code_client", "code_article"], how="left")
    features = features.merge(g3, on="code_client", how="left")
    features = features.merge(g4, on="code_article", how="left")
    features = features.merge(g5, on="code_client", how="left")

    # Keep and order final columns
    cols_to_keep = [
        "code_client", "code_article",
        "avg_qty", "std_qty", "min_qty", "max_qty", "total_qty",
        "frequency", "last_qty", "recency_days",
        "avg_delay_days", "trend",
        "current_month_coef", "avg_seasonal_coef", "best_month",
        "has_gps", "latitude", "longitude",
        "categorie", "designation", "is_bulk_product",
        "nb_clients", "days_since_first_order", "is_new_product",
        "client_total_products", "client_total_invoices",
        "client_avg_basket_size"
    ]
    
    features = features[cols_to_keep].copy()

    # Save if save_path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(save_path, index=False)
        logger.info(f"Saved feature matrix to {save_path}")

    return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    input_path = "data/processed/lsat_clean.csv"
    output_path = "data/processed/feature_matrix.csv"
    
    # Load clean data
    df = pd.read_csv(input_path, parse_dates=["date_commande"])
    
    # Build feature matrix
    features = build_feature_matrix(df, save_path=output_path)
    
    # Print statistics
    print(f"\nFeature matrix shape : {features.shape[0]} rows x {features.shape[1]} columns")
    print(f"Unique clients       : {features['code_client'].nunique()}")
    print(f"Unique products      : {features['code_article'].nunique()}")
    print(f"Pairs (clientxproduct): {len(features)}")
    print(f"Columns : {list(features.columns)}")
    print("\nNumeric Columns Summary:")
    print(features.describe().to_string())
    print("\nFirst 3 rows:")
    print(features.head(3).to_string())
