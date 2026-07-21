"""
LAYER 1 — Data Access
Loads and joins the 3 Excel source files.
this file is only  for reading and joining data.
"""

import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_factures(filepath: str) -> pd.DataFrame:
    """
    Load the invoices ( les factures ) Excel file.

    Args:
        filepath: path to commande_phonesTech Excel file

    Returns:
        DataFrame ( table in pandas = dataframe )  with columns:
        code_facture, code_client, date_commande,
        ref_commercial, societe
    """
    logger.info(f"Loading invoices from {filepath}")
    df = pd.read_excel(filepath)

    df = df.rename(columns={
        "No_":              "code_facture",
        "client_code":      "code_client",
        "Posting Date":     "date_commande",
        "Salesperson Code": "ref_commercial",
        "company":          "societe",
    })

    df["date_commande"] = pd.to_datetime(df["date_commande"])

    logger.info(
        f"Invoices loaded: {len(df)} rows, "
        f"{df['code_client'].nunique()} unique clients, "
        f"period {df['date_commande'].min().date()} → "
        f"{df['date_commande'].max().date()}"
    )
    return df


def load_lignes(filepath: str) -> pd.DataFrame:
    """
    Load the order lines Excel file.

    Args:
        filepath: path to commande_lines_PhonesTech Excel file

    Returns:
        DataFrame with columns:
        code_facture, quantite, code_article,
        designation, categorie, societe_ligne
    """
    logger.info(f"Loading order lines from {filepath}")
    df = pd.read_excel(filepath)

    df = df.rename(columns={
        "Document No_":          "code_facture",
        "Quantity":              "quantite",
        "code_article":          "code_article",
        "designation_article":   "designation",
        "Item Category Code":    "categorie",
        "company":               "societe_ligne",
    })

    logger.info(
        f"Order lines loaded: {len(df)} rows, "
        f"{df['code_article'].nunique()} unique products"
    )
    return df


def load_clients(filepath: str) -> pd.DataFrame:
    """
    Load the GPS coordinates file.

    Args:
        filepath: path to client-lat-lng Excel file

    Returns:
        DataFrame with columns:
        code_client, societe, latitude, longitude
    """
    logger.info(f"Loading client GPS from {filepath}")
    df = pd.read_excel(filepath)

    df = df.rename(columns={
        "clientCode": "code_client",
        "company":    "societe",
        "lat":        "latitude",
        "lng":        "longitude",
    })

    logger.info(
        f"GPS loaded: {len(df)} clients with coordinates"
    )
    return df


def build_main_table(
    lignes: pd.DataFrame,
    factures: pd.DataFrame,
    clients: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join the three clean tables into one main table.
    Jointure sur (code_facture + societe) et (code_client + societe).
    """
    logger.info("Building main table by merging lines, invoices, and clients...")

    # Jointure des lignes et des factures sur le numéro de facture et la société (societe_ligne vs societe)
    df = lignes.merge(
        factures,
        left_on=["code_facture", "societe_ligne"],
        right_on=["code_facture", "societe"],
        how="inner" # Seulement les lignes qui matchent les deux pour éviter les incohérences
    )

    # Jointure avec les coordonnées GPS sur le code client et la société
    df = df.merge(
        clients,
        on=["code_client", "societe"],
        how="left"
    )
    return df



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    f = load_factures("data/raw/commande phonesTech 01-01-2024--23-06-2026.xlsx")
    l = load_lignes("data/raw/commande lines PhonesTech.xlsx")
    c = load_clients("data/raw/client-lat-lng.xlsx")
    print(f"Factures raw shape: {f.shape}")
    print(f"Lignes raw shape: {l.shape}")
    print(f"Clients GPS raw shape: {c.shape}")
