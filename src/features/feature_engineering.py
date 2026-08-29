"""
LAYER 2 — AI / Machine Learning
Transforms raw cleaned data into ML features.
NO model training here — only feature computation.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def build_categorical_quarterly_index(df: pd.DataFrame) -> dict:
    """
    Computes quarterly seasonal index per category:
    coef[categorie][quarter] = mean(quantite | categorie, quarter) / mean(quantite | categorie, all quarters)
    Excludes the outlier week of 2024-10-13 to 2024-10-20 (client CLT105703 massive one-off order).
    """
    df_clean = df.copy()
    df_clean["date_commande"] = pd.to_datetime(df_clean["date_commande"])
    
    # Exclude 2024-10-13 to 2024-10-20 anomaly
    mask_outlier = (df_clean["date_commande"] >= "2024-10-13") & (df_clean["date_commande"] <= "2024-10-20")
    df_clean = df_clean[~mask_outlier]
    
    df_clean["trimestre"] = df_clean["date_commande"].dt.quarter
    df_clean["categorie"] = df_clean["categorie"].fillna("UNKNOWN").astype(str)
    
    cat_q_means = df_clean.groupby(["categorie", "trimestre"])["quantite"].mean()
    cat_all_means = df_clean.groupby("categorie")["quantite"].mean()
    
    ratio = (cat_q_means / (cat_all_means + 1e-6)).unstack().fillna(1.0)
    # Clip to avoid extreme ratios
    ratio = ratio.clip(0.5, 2.0)
    return ratio.to_dict(orient="index")


def build_feature_matrix(
    df: pd.DataFrame,
    reference_date: pd.Timestamp = None,
    save_path: str = None,
) -> pd.DataFrame:
    logger.info("Starting build_feature_matrix...")

    df = df.copy()
    df["date_commande"] = pd.to_datetime(df["date_commande"])

    if reference_date is None:
        reference_date = pd.to_datetime("2026-06-22")
    else:
        reference_date = pd.to_datetime(reference_date)

    df = df.sort_values(by=["code_client", "code_article", "date_commande"]).copy()

    df["quantite_lag1"] = df.groupby(["code_client", "code_article"])["quantite"].shift(1)
    df["date_diff"] = df.groupby(["code_client", "code_article"])["date_commande"].diff().dt.days

    logger.info("Computing Group 1 features (Order history)...")
    grp = df.groupby(["code_client", "code_article"])

    g1 = pd.DataFrame()
    g1["avg_qty"] = grp["quantite"].mean()
    g1["median_qty"] = grp["quantite"].median()
    g1["std_qty"] = grp["quantite"].std().fillna(0.0)
    g1["min_qty"] = grp["quantite"].min()
    g1["max_qty"] = grp["quantite"].max()
    g1["total_qty"] = grp["quantite"].sum()
    g1["frequency"] = grp["quantite"].count()
    g1["last_qty"] = grp["quantite"].last()

    last_date = grp["date_commande"].max()
    g1["recency_days"] = (reference_date - last_date).dt.days

    g1["avg_delay_days"] = grp["date_diff"].mean().fillna(30.0)
    g1["recency_relative"] = g1["recency_days"] / g1["avg_delay_days"].replace(0, 30.0)

    g1["societe"] = grp["societe"].first().values
    company_map = {"LSAT": 0, "NEWTECH": 1, "ONETEL": 2}
    g1["company_encoded"] = g1["societe"].map(company_map)

    def compute_trend(series):
        n = len(series)
        if n < 2:
            return 0.0
        mid = n // 2
        old_avg = series.iloc[:mid].mean()
        new_avg = series.iloc[mid:].mean()
        return (new_avg - old_avg) / (old_avg + 1e-6)

    # Capped trend between -0.9 and 3.0 to prevent outlier explosion
    g1["trend"] = grp["quantite"].apply(compute_trend).clip(-0.9, 3.0)
    g1 = g1.reset_index()

    logger.info("Computing Group 2 features (Categorical Quarterly Seasonality)...")
    cat_quarterly_index = build_categorical_quarterly_index(df)
    ref_quarter = int(reference_date.quarter)

    logger.info("Computing Group 3 features (Geography)...")
    g3 = df.groupby("code_client").agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first")
    ).reset_index()
    g3["has_gps"] = g3["latitude"].notna()

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

    # Map cat_quarterly_coef on product category
    def map_quarterly_coef(cat):
        cat_str = str(cat) if pd.notna(cat) else "UNKNOWN"
        return float(cat_quarterly_index.get(cat_str, {}).get(ref_quarter, 1.0))

    g4["cat_quarterly_coef"] = g4["categorie"].apply(map_quarterly_coef)

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

    logger.info("Merging all feature groups...")
    features = g1.merge(g3, on="code_client", how="left")
    features = features.merge(g4, on="code_article", how="left")
    features = features.merge(g5, on="code_client", how="left")

    cols_to_keep = [
        "code_client", "code_article",
        "avg_qty", "median_qty", "std_qty", "min_qty", "max_qty", "total_qty",
        "frequency", "last_qty", "recency_days",
        "avg_delay_days", "recency_relative", "trend",
        "company_encoded",
        "cat_quarterly_coef",
        "has_gps", "latitude", "longitude",
        "categorie", "designation", "is_bulk_product",
        "nb_clients", "days_since_first_order", "is_new_product",
        "client_total_products", "client_total_invoices",
        "client_avg_basket_size"
    ]

    features = features[cols_to_keep].copy()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(save_path, index=False)
        logger.info(f"Saved feature matrix to {save_path}")

    return features


def build_features_for_negative_pairs(
    negative_pairs: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    reference_date = pd.to_datetime(reference_date)

    client_features = (
        feature_matrix
        .drop_duplicates(subset=["code_client"])
        [["code_client", "company_encoded", "has_gps", "latitude", "longitude",
          "client_total_products", "client_total_invoices",
          "client_avg_basket_size"]]
    )

    product_features = (
        feature_matrix
        .drop_duplicates(subset=["code_article"])
        [["code_article", "categorie", "designation", "is_bulk_product",
          "nb_clients", "days_since_first_order", "is_new_product", "cat_quarterly_coef"]]
    )

    neg = negative_pairs[["code_client", "code_article"]].copy()
    neg = neg.merge(client_features, on="code_client", how="left")
    neg = neg.merge(product_features, on="code_article", how="left")

    neg["avg_qty"] = 0.0
    neg["std_qty"] = 0.0
    neg["min_qty"] = 0.0
    neg["max_qty"] = 0.0
    neg["total_qty"] = 0.0
    neg["frequency"] = 0
    neg["last_qty"] = 0.0
    neg["recency_days"] = 9999.0
    neg["avg_delay_days"] = 999.0
    neg["recency_relative"] = 9999.0 / 999.0
    neg["trend"] = 0.0
    neg["cat_quarterly_coef"] = 1.0

    cols_order = [
        "code_client", "code_article",
        "avg_qty", "std_qty", "min_qty", "max_qty", "total_qty",
        "frequency", "last_qty", "recency_days",
        "avg_delay_days", "recency_relative", "trend",
        "company_encoded",
        "cat_quarterly_coef",
        "has_gps", "latitude", "longitude",
        "categorie", "designation", "is_bulk_product",
        "nb_clients", "days_since_first_order", "is_new_product",
        "client_total_products", "client_total_invoices",
        "client_avg_basket_size"
    ]

    return neg[cols_order].copy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    commandes_path = "data/processed/commandes_clean.csv"
    lignes_path = "data/processed/lignes_clean.csv"
    gps_path = "data/processed/gps_clean.csv"
    output_path = "data/processed/feature_matrix.csv"

    logger.info("Loading cleaned datasets...")
    df_commandes = pd.read_csv(commandes_path, parse_dates=["date_commande"])
    df_lignes = pd.read_csv(lignes_path)
    df_gps = pd.read_csv(gps_path)

    from src.data.loader import build_main_table
    logger.info("Merging clean tables via build_main_table...")
    df = build_main_table(df_lignes, df_commandes, df_gps)

    main_table_path = "data/processed/main_table.csv"
    df.to_csv(main_table_path, index=False)
    logger.info(f"Saved merged main table to {main_table_path}")

    logger.info(f"Merged table shape: {df.shape}")

    features = build_feature_matrix(df, save_path=output_path)

    print(f"\nFeature matrix shape : {features.shape[0]} rows x {features.shape[1]} columns")
    print(f"Unique clients       : {features['code_client'].nunique()}")
    print(f"Unique products      : {features['code_article'].nunique()}")
    print(f"Columns : {list(features.columns)}")