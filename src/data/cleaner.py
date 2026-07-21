"""
LAYER 1 — Data Access
Cleans the three raw files individually.
Handles nulls, duplicates, text normalization, and outputs separate files.
"""

import pandas as pd
import logging
from pathlib import Path
from src.data.loader import load_factures, load_lignes, load_clients

logger = logging.getLogger(__name__)


def clean_commandes(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["code_facture", "code_client"]).copy()
    dropped = before - len(df)

    for col in ["code_facture", "code_client", "ref_commercial", "societe"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["date_commande"] = pd.to_datetime(df["date_commande"])
    df["mois"] = df["date_commande"].dt.month
    df["annee"] = df["date_commande"].dt.year
    df["annee_mois"] = df["date_commande"].dt.to_period("M")
    df["jour_semaine"] = df["date_commande"].dt.dayofweek
    df["trimestre"] = df["date_commande"].dt.quarter

    logger.info(f"clean_commandes: Dropped {dropped} rows. Cleaned shape: {df.shape}")
    return df


def clean_lignes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the order lines (lignes) table.
    """
    before = len(df)
    # Drop rows where critical fields are null
    df = df.dropna(subset=["code_facture", "code_article"]).copy()
    dropped = before - len(df)
    
    # Remove duplicate order lines (same invoice + same product)
    df = df.drop_duplicates(subset=["code_facture", "code_article"], keep="first").copy()
    dups_removed = before - dropped - len(df)
    
    # Fill category and designation nulls
    null_cat = df["categorie"].isnull().sum()
    # Mapping automatique : remplir les catégories manquantes par le code_article
    mapping_categorie = (
        df.dropna(subset=["categorie"])
        .groupby("code_article")["categorie"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    )

    mask = df["categorie"].isna()
    df.loc[mask, "categorie"] = df.loc[mask, "code_article"].map(mapping_categorie)

    # Affecter UNKNOWN au reste
    df["categorie"] = df["categorie"].fillna("UNKNOWN")
    null_desig = df["designation"].isnull().sum()
    df["designation"] = df["designation"].fillna(df["code_article"])
    
    # Strip whitespace and uppercase category
    for col in ["code_facture", "code_article", "designation", "categorie", "societe_ligne"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df["categorie"] = df["categorie"].str.upper()
    
    # Flag bulk orders
    df["is_bulk_order"] = df["quantite"] > 1000
    
    # Filter quantity outliers (quantite > 0)
    df = df[df["quantite"] > 0].copy()
    
    logger.info(
        f"clean_lignes: Dropped {dropped} nulls, removed {dups_removed} duplicates. "
        f"Filled {null_cat} null categories, {null_desig} null designations. Cleaned shape: {df.shape}"
    )
    return df


def clean_gps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the GPS coordinates (clients) table.
    """
    before = len(df)
    df = df.dropna(subset=["code_client"]).copy()
    dropped = before - len(df)
    
    for col in ["code_client", "societe"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    logger.info(f"clean_gps: Dropped {dropped} rows. Cleaned shape: {df.shape}")
    return df


def clean_all(
    factures_path: str,
    lignes_path: str,
    clients_path: str,
    out_dir: str = "data/processed",
) -> None:
    """
    Full cleaning pipeline that saves 3 clean CSV files.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting clean_all process...")
    
    # 1. Process invoices (commandes)
    df_fact = load_factures(factures_path)
    df_fact_clean = clean_commandes(df_fact)
    df_fact_clean.to_csv(out_path / "commandes_clean.csv", index=False)
    logger.info(f"Saved cleaned commandes to {out_path / 'commandes_clean.csv'}")
    
    # 2. Process order lines (lignes)
    df_lignes = load_lignes(lignes_path)
    df_lignes_clean = clean_lignes(df_lignes)
    df_lignes_clean.to_csv(out_path / "lignes_clean.csv", index=False)
    logger.info(f"Saved cleaned lignes to {out_path / 'lignes_clean.csv'}")
    
    # 3. Process GPS (clients)
    df_clients = load_clients(clients_path)
    df_clients_clean = clean_gps(df_clients)
    df_clients_clean.to_csv(out_path / "gps_clean.csv", index=False)
    logger.info(f"Saved cleaned GPS data to {out_path / 'gps_clean.csv'}")
    
    logger.info("All 3 files processed successfully!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    clean_all(
        factures_path="data/raw/commande_phonesTech_lsat.xlsx",
        lignes_path="data/raw/commande_lines_lsat.xlsx",
        clients_path="data/raw/client_lat_lng_lsat.xlsx",
        out_dir="data/processed",
    )
