"""
LAYER 1 — Data Access
Cleans the main table after loading.
Handles nulls, duplicates, type normalization.
"""

import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate order lines.
    A duplicate is same invoice + same product.
    """
    before = len(df)
    df = df.drop_duplicates(subset=["code_facture", "code_article"], keep="first").copy()
    removed = before - len(df)
    logger.info(f"remove_duplicates: {removed} rows removed.")
    return df


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values.
    - Drop rows where code_client is null
    - Drop rows where code_article is null
    - Fill categorie nulls with 'UNKNOWN'
    - Fill designation nulls with code_article value
    - DO NOT touch latitude/longitude nulls (expected)
    """
    before = len(df)
    
    # Drop rows where code_client or code_article is null
    df = df.dropna(subset=["code_client", "code_article"]).copy()
    dropped = before - len(df)
    
    # Fill categorie nulls with 'UNKNOWN'
    null_cat = df["categorie"].isnull().sum()
    df["categorie"] = df["categorie"].fillna("UNKNOWN")
    
    # Fill designation nulls with code_article value
    null_desig = df["designation"].isnull().sum()
    df["designation"] = df["designation"].fillna(df["code_article"])
    
    logger.info(
        f"handle_nulls: Dropped {dropped} rows with null client/article. "
        f"Filled {null_cat} null categories with 'UNKNOWN'. "
        f"Filled {null_desig} null designations with code_article."
    )
    return df


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace and uppercase category codes.
    """
    for col in ["categorie", "code_article", "code_client", "designation"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    df["categorie"] = df["categorie"].str.upper()
    return df


def flag_bulk_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag bulk orders with quantite > 1000.
    """
    df["is_bulk_order"] = df["quantite"] > 1000
    bulk_count = df["is_bulk_order"].sum()
    logger.info(f"flag_bulk_orders: {bulk_count} bulk orders flagged.")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time features from date_commande.
    """
    df["date_commande"] = pd.to_datetime(df["date_commande"])
    df["mois"] = df["date_commande"].dt.month
    df["annee"] = df["date_commande"].dt.year
    df["annee_mois"] = df["date_commande"].dt.to_period("M")
    df["jour_semaine"] = df["date_commande"].dt.dayofweek
    df["trimestre"] = df["date_commande"].dt.quarter
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline.
    Calls all cleaning steps in order.

    Args:
        df: raw main table from loader.build_main_table()

    Returns:
        Cleaned DataFrame ready for feature engineering
    """
    start_rows = len(df)
    logger.info(f"Cleaning started : {start_rows} rows")
    
    df = remove_duplicates(df)
    df = handle_nulls(df)
    df = normalize_text(df)
    df = flag_bulk_orders(df)
    df = add_time_features(df)
    
    end_rows = len(df)
    removed = start_rows - end_rows
    logger.info(f"Cleaning done : {end_rows} rows remaining, {removed} removed")
    
    return df


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    
    input_path = "data/processed/lsat_main_table.csv"
    output_path = "data/processed/lsat_clean.csv"
    
    if os.path.exists(input_path):
        df_raw = pd.read_csv(input_path)
        print(f"Shape before cleaning: {df_raw.shape}")
        
        df_clean = clean(df_raw)
        
        print(f"Shape after cleaning: {df_clean.shape}")
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        print(f"Saved cleaned data to {output_path}")
    else:
        print(f"Error: {input_path} does not exist.")
