"""
LAYER 2 — AI / Machine Learning
Builds the target variable for training.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

NEGATIVE_RATIO = 3


def build_monthly_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date_commande"] = pd.to_datetime(df["date_commande"])
    df["annee_mois"] = df["date_commande"].dt.to_period("M")

    monthly = (
        df.groupby(["code_client", "code_article", "annee_mois"])["quantite"]
        .sum()
        .reset_index()
        .rename(columns={"quantite": "qty_mois"})
    )

    logger.info(f"Monthly orders built: {len(monthly)} rows")
    return monthly


def build_positive_pairs(monthly: pd.DataFrame) -> pd.DataFrame:
    monthly = monthly.sort_values(["code_client", "code_article", "annee_mois"])

    monthly["next_qty"] = monthly.groupby(["code_client", "code_article"])["qty_mois"].shift(-1)

    last_seen = (
        monthly.groupby(["code_client", "code_article"])
        .last()
        .reset_index()
    )

    last_seen["target_qty"] = last_seen["next_qty"].fillna(0.0)
    last_seen["target_bought"] = (last_seen["target_qty"] > 0).astype(int)

    positives = last_seen[["code_client", "code_article", "target_qty", "target_bought"]].copy()

    logger.info(
        f"Positive pairs built: {len(positives)} | "
        f"reordered next month: {positives['target_bought'].sum()} "
        f"({positives['target_bought'].mean()*100:.1f}%)"
    )
    return positives


def build_negative_pairs(
    monthly: pd.DataFrame,
    positives: pd.DataFrame,
    all_products: pd.Series,
    ratio: int = NEGATIVE_RATIO,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    all_products_arr = all_products.unique()

    client_products = (
        monthly.groupby("code_client")["code_article"]
        .apply(set)
        .to_dict()
    )

    negatives_rows = []
    n_positives_per_client = positives.groupby("code_client").size()

    for client, n_pos in n_positives_per_client.items():
        n_neg_needed = n_pos * ratio
        never_ordered = np.setdiff1d(
            all_products_arr,
            list(client_products.get(client, set())),
            assume_unique=False,
        )

        if len(never_ordered) == 0:
            continue

        n_sample = min(n_neg_needed, len(never_ordered))
        sampled = rng.choice(never_ordered, size=n_sample, replace=False)

        for article in sampled:
            negatives_rows.append((client, article))

    negatives = pd.DataFrame(negatives_rows, columns=["code_client", "code_article"])
    negatives["target_qty"] = 0.0
    negatives["target_bought"] = 0

    logger.info(f"Negative pairs sampled: {len(negatives)} (ratio {ratio}:1)")
    return negatives


def build_training_set(
    main_table: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    reference_date: pd.Timestamp,
    save_path: str = None,
) -> pd.DataFrame:
    from src.features.feature_engineering import build_features_for_negative_pairs

    logger.info("Building training set...")

    monthly = build_monthly_orders(main_table)
    positives = build_positive_pairs(monthly)
    negatives = build_negative_pairs(
        monthly, positives, all_products=main_table["code_article"]
    )

    positive_training = feature_matrix.merge(
        positives, on=["code_client", "code_article"], how="inner"
    )

    negative_features = build_features_for_negative_pairs(
        negatives, feature_matrix, reference_date
    )
    negative_training = negative_features.merge(
        negatives[["code_client", "code_article", "target_qty", "target_bought"]],
        on=["code_client", "code_article"], how="left"
    )

    training_set = pd.concat([positive_training, negative_training], ignore_index=True)

    logger.info(
        f"Final training set: {len(training_set)} rows | "
        f"positive rate: {training_set['target_bought'].mean()*100:.1f}%"
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        training_set.to_csv(save_path, index=False)
        logger.info(f"Saved training set to {save_path}")

    return training_set


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from src.data.loader import load_factures, load_lignes, load_clients, build_main_table
    from src.data.cleaner import clean_commandes, clean_lignes, clean_gps
    from src.features.feature_engineering import build_feature_matrix

    factures_path = "data/raw/commande_phonesTech_lsat.xlsx"
    lignes_path = "data/raw/commande_lines_lsat.xlsx"
    gps_path = "data/raw/client_lat_lng_lsat.xlsx"
    output_path = "data/processed/training_set.csv"

    reference_date = pd.to_datetime("2026-06-22")

    logger.info("Loading and cleaning LSAT data...")
    df_fac = clean_commandes(load_factures(factures_path))
    df_lignes = clean_lignes(load_lignes(lignes_path))
    df_gps = clean_gps(load_clients(gps_path))

    main_table = build_main_table(df_lignes, df_fac, df_gps)
    logger.info(f"Main table: {main_table.shape}")

    feature_matrix = build_feature_matrix(main_table, reference_date=reference_date)
    logger.info(f"Feature matrix: {feature_matrix.shape}")

    training_set = build_training_set(
        main_table, feature_matrix, reference_date, save_path=output_path
    )

    print(f"\nTraining set shape: {training_set.shape}")
    print(f"Columns: {list(training_set.columns)}")
    print(f"\nTarget distribution:")
    print(training_set["target_bought"].value_counts())
    print(f"\nFirst 5 rows:")
    print(training_set.head(5).to_string())