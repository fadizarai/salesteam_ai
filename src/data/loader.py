"""
LAYER 1 — Data Access
Loads and joins the 3 Excel source files.
No business logic here — only reading and joining data.
"""

import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_factures(filepath: str, company_filter: str = "LSAT") -> pd.DataFrame:
    """
    Load the invoices Excel file and filter by company.

    Args:
        filepath: path to commande_phonesTech Excel file
        company_filter: company name to filter (default LSAT)

    Returns:
        DataFrame with columns:
        code_facture, code_client, date_commande,
        ref_commercial, societe
    """
    logger.info(f"Loading invoices from {filepath}")
    df = pd.read_excel(filepath)

    if company_filter:
        df = df[df["company"] == company_filter].copy()
        logger.info(f"Filtered to {company_filter}: {len(df)} invoices")

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


def load_lignes(filepath: str, facture_codes: list = None) -> pd.DataFrame:
    """
    Load the order lines Excel file.
    Filter by invoice codes to get only LSAT lines
    (company field in this file does NOT contain LSAT).

    Args:
        filepath: path to commande_lines_PhonesTech Excel file
        facture_codes: list of invoice No_ to filter on

    Returns:
        DataFrame with columns:
        code_facture, quantite, code_article,
        designation, categorie, societe_ligne
    """
    logger.info(f"Loading order lines from {filepath}")
    df = pd.read_excel(filepath)

    if facture_codes is not None:
        before = len(df)
        df = df[df["Document No_"].isin(facture_codes)].copy()
        logger.info(
            f"Filtered lines by invoice codes: "
            f"{before} → {len(df)} rows"
        )

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


def load_clients(filepath: str, company_filter: str = "LSAT") -> pd.DataFrame:
    """
    Load the GPS coordinates file.

    Args:
        filepath: path to client-lat-lng Excel file
        company_filter: filter by company (default LSAT)

    Returns:
        DataFrame with columns:
        code_client, societe, latitude, longitude
    """
    logger.info(f"Loading client GPS from {filepath}")
    df = pd.read_excel(filepath)

    if company_filter:
        df = df[df["company"] == company_filter].copy()

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
    factures_path: str,
    lignes_path: str,
    clients_path: str,
    company: str = "LSAT",
    save_path: str = None,
) -> pd.DataFrame:
    """
    Build the main analysis table by joining all 3 sources.

    Join logic:
    1. Load invoices filtered by company
    2. Load order lines filtered by those invoice codes
    3. Left join with GPS coordinates
    4. Add time features (month, year, week)
    5. Filter quantity outliers (> 1000)

    Args:
        factures_path: path to invoices Excel
        lignes_path: path to order lines Excel
        clients_path: path to GPS Excel
        company: company to filter (default LSAT)
        save_path: if provided, save CSV to this path

    Returns:
        Complete joined DataFrame ready for feature engineering
    """
    # Step 1 — load invoices
    factures = load_factures(factures_path, company_filter=company)

    # Step 2 — load lines filtered by LSAT invoice codes
    facture_codes = factures["code_facture"].tolist()
    lignes = load_lignes(lignes_path, facture_codes=facture_codes)

    # Step 3 — join invoices + lines
    df = lignes.merge(
        factures[["code_facture", "code_client",
                  "date_commande", "ref_commercial", "societe"]],
        on="code_facture",
        how="left",
    )
    logger.info(f"After join invoices + lines: {len(df)} rows")

    # Step 4 — left join with GPS (not all clients have coordinates)
    clients = load_clients(clients_path, company_filter=company)
    df = df.merge(
        clients[["code_client", "latitude", "longitude"]],
        on="code_client",
        how="left",
    )
    clients_with_gps = df["latitude"].notna().sum()
    logger.info(
        f"Clients with GPS: "
        f"{df[df['latitude'].notna()]['code_client'].nunique()} / "
        f"{df['code_client'].nunique()}"
    )

    # Step 5 — add time features
    df["mois"]        = df["date_commande"].dt.month
    df["annee"]       = df["date_commande"].dt.year
    df["annee_mois"]  = df["date_commande"].dt.to_period("M")
    df["jour_semaine"] = df["date_commande"].dt.dayofweek

    # Step 6 — filter quantity outliers
    before = len(df)
    df = df[df["quantite"] > 0].copy()
    logger.info(
        f"Quantity filter (q > 0): "
        f"{before} → {len(df)} rows "
        f"({before - len(df)} outliers removed)"
    )

    logger.info(
        f"Main table ready: {len(df)} rows × {len(df.columns)} columns"
    )

    # Step 7 — save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Saved to {save_path}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df = build_main_table(
        factures_path="data/raw/commande phonesTech 01-01-2024--23-06-2026.xlsx",
        lignes_path="data/raw/commande lines PhonesTech.xlsx",
        clients_path="data/raw/client-lat-lng.xlsx",
        company="LSAT",
        save_path="data/processed/lsat_main_table.csv",
    )

    print("\n" + "="*50)
    print("MAIN TABLE SUMMARY")
    print("="*50)
    print(f"Shape         : {df.shape}")
    print(f"Clients       : {df['code_client'].nunique()}")
    print(f"Products      : {df['code_article'].nunique()}")
    print(f"Invoices      : {df['code_facture'].nunique()}")
    print(f"Commercials   : {df['ref_commercial'].nunique()}")
    print(f"Period        : {df['date_commande'].min().date()} -> {df['date_commande'].max().date()}")
    print(f"With GPS      : {df['latitude'].notna().sum()} rows")
    print(f"\nColumns       : {list(df.columns)}")
    print(f"\nSample:")
    print(df.head(3).to_string())
