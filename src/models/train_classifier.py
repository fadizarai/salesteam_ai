"""
Script d'entraînement du classifieur XGBoost (Layer 2 — AI / ML)

Etape 1 -- Charger le training_set (visit-level)
Etape 2 -- Encoder la colonne categorielle 'categorie'
Etape 3 -- Preparer X (features) et y (target_bought)
Etape 4 -- Split TEMPOREL (train <= 2025 / val early 2026 / test late 2026)
Etape 5 -- Calculer scale_pos_weight sur le train uniquement
Etape 6 -- Entrainer XGBoost avec early stopping sur une validation dediee
Etape 7 -- Evaluer (classification report, ROC-AUC, PR-AUC, importance features)
Etape 8 -- Sauvegarder modele + encodeur + metadonnees de colonnes
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
# pyrefly: ignore [missing-import]
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)



def run_training_pipeline(
    data_path: str = "data/processed/training_set.csv",
    model_output_path: str = "src/models/classifier_lsat.joblib",
    encoder_output_path: str = "src/models/encoder_categorie.joblib",
    metadata_output_path: str = "src/models/classifier_lsat_metadata.json",
    train_cutoff: str = "2025-12-31",   # train on visits up to this date
    val_cutoff: str = "2026-03-31",     # validate on visits in early 2026
    random_state: int = 42,
):
    # Étape 1 — Charger le training_set
    logger.info(f"Étape 1: Chargement des données depuis {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} introuvable. Exécuter target_builder.py d'abord.")

    df = pd.read_csv(data_path)
    logger.info(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes.")

    # Étape 2 — Encoder 'categorie'
    logger.info("Étape 2: Encodage de 'categorie'...")
    le_categorie = LabelEncoder()
    df["categorie"] = df["categorie"].fillna("UNKNOWN").astype(str)
    df["categorie_encoded"] = le_categorie.fit_transform(df["categorie"])
    logger.info(f"Catégories uniques encodées : {len(le_categorie.classes_)}")

    # Étape 3 — X / y
    logger.info("Étape 3: Séparation features (X) / cible (y)...")
    feature_cols = [
        "frequency",
        "total_qty",
        "avg_qty",
        "avg_delay_days",
        "recency_days",
        "recency_relative",
        "std_qty",
        "min_qty",
        "best_month",
        "avg_seasonal_coef"
    ]

    bool_cols = df[feature_cols].select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    logger.info(f"Nombre de features retenues : {len(feature_cols)} → {feature_cols}")

    # Etape 4 -- Split TEMPOREL (evite la fuite temporelle)
    # Un split aleatoire par client peut encore laisser des visites futures
    # informer des visites passees via les features agreges.
    # Le split temporel est la seule vraie garantie : le modele ne voit
    # jamais une visite qui se produit APRES la date de coupure.
    #
    # train  : visites <= train_cutoff (2025-12-31)
    # val    : visites entre train_cutoff et val_cutoff (Q1 2026)
    # test   : visites apres val_cutoff (Q2 2026+)
    logger.info("Etape 4: Split TEMPOREL train/val/test...")

    if "visit_date" not in df.columns:
        raise ValueError(
            "La colonne 'visit_date' est absente du training_set. "
            "Regenerez training_set.csv avec le nouveau target_builder.py."
        )

    df["visit_date"] = pd.to_datetime(df["visit_date"])
    train_cutoff_dt = pd.Timestamp(train_cutoff)
    val_cutoff_dt   = pd.Timestamp(val_cutoff)

    train_df = df[df["visit_date"] <= train_cutoff_dt].copy()
    val_df   = df[(df["visit_date"] > train_cutoff_dt) & (df["visit_date"] <= val_cutoff_dt)].copy()
    test_df  = df[df["visit_date"] > val_cutoff_dt].copy()

    logger.info(
        f"Train : {len(train_df)} rows ({train_df['code_client'].nunique()} clients, "
        f"up to {train_cutoff})"
    )
    logger.info(
        f"Val   : {len(val_df)} rows ({val_df['code_client'].nunique()} clients, "
        f"{train_cutoff} -> {val_cutoff})"
    )
    logger.info(
        f"Test  : {len(test_df)} rows ({test_df['code_client'].nunique()} clients, "
        f"after {val_cutoff})"
    )

    if len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Val ou test set vide avec les dates de coupure actuelles. "
            f"Verifiez que les donnees couvrent au-dela de {val_cutoff}."
        )

    X_train, y_train = train_df[feature_cols], train_df["target_bought"]
    X_val,   y_val   = val_df[feature_cols],   val_df["target_bought"]
    X_test,  y_test  = test_df[feature_cols],  test_df["target_bought"]

    # Verify no temporal leakage
    train_max = train_df["visit_date"].max()
    val_min   = val_df["visit_date"].min()
    test_min  = test_df["visit_date"].min()
    logger.info(f"Anti-leakage check: train_max={train_max.date()} | val_min={val_min.date()} | test_min={test_min.date()}")

    # Étape 5 — scale_pos_weight sur le train uniquement
    logger.info("Étape 5: Calcul de scale_pos_weight (train uniquement)...")
    num_negatives = (y_train == 0).sum()
    num_positives = (y_train == 1).sum()
    scale_pos_weight = num_negatives / max(1, num_positives)
    logger.info(
        f"Classe 0: {num_negatives} | Classe 1: {num_positives} | "
        f"scale_pos_weight: {scale_pos_weight:.2f}"
    )

    # Étape 6 — Entraînement avec early stopping sur la validation
    logger.info("Étape 6: Entraînement XGBoost avec early stopping...")
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=random_state,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        early_stopping_rounds=20,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info(f"Entraînement terminé — arrêt à l'arbre {model.best_iteration}.")

    # Étape 7 — Évaluation sur le test
    logger.info("Étape 7: Évaluation sur le jeu de test...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc_score = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    report_text = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)

    print("\n" + "=" * 60)
    print("      RÉSULTATS DE L'ÉVALUATION DU CLASSIFIEUR XGBOOST")
    print("=" * 60)
    print(report_text)
    print(f"ROC-AUC Score : {auc_score:.4f}")
    print(f"PR-AUC Score  : {pr_auc:.4f}  (référence utile pour classes déséquilibrées)")
    print("\nMatrice de Confusion :")
    print(cm)
    print("\nTop 15 features les plus importantes :")
    print(importances.head(15).to_string())
    print("=" * 60 + "\n")

    # Étape 8 — Sauvegarde modèle + encodeur + métadonnées
    logger.info("Étape 8: Sauvegarde du modèle, de l'encodeur et des métadonnées...")

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(encoder_output_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_output_path)
    joblib.dump(le_categorie, encoder_output_path)

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "best_iteration": int(model.best_iteration),
        "scale_pos_weight": float(scale_pos_weight),
        "metrics": {
            "roc_auc": float(auc_score),
            "pr_auc": float(pr_auc),
        },
        "train_clients": int(train_df["code_client"].nunique()),
        "test_clients": int(test_df["code_client"].nunique()),
    }
    with open(metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    root_model_path = "models/classifier_lsat.joblib"
    root_encoder_path = "models/encoder_categorie.joblib"
    Path(root_model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, root_model_path)
    joblib.dump(le_categorie, root_encoder_path)

    logger.info(f"Modèle sauvegardé : {model_output_path} (+ {root_model_path})")
    logger.info(f"Encodeur sauvegardé : {encoder_output_path} (+ {root_encoder_path})")
    logger.info(f"Métadonnées sauvegardées : {metadata_output_path}")

    return model, le_categorie, metadata


if __name__ == "__main__":
    run_training_pipeline()