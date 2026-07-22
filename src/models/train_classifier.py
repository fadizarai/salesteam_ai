"""
Script d'entraînement du classifieur XGBoost (Layer 2 — AI / ML)

Étape 1 — Charger le training_set
Étape 2 — Encoder la colonne catégorielle 'categorie'
Étape 3 — Préparer X (features) et y (target_bought)
Étape 4 — Split train/test PAR CLIENT (évite la fuite de données)
Étape 5 — Calculer scale_pos_weight sur le train uniquement
Étape 6 — Entraîner XGBoost avec early stopping sur une validation dédiée
Étape 7 — Évaluer (classification report, ROC-AUC, PR-AUC, importance features)
Étape 8 — Sauvegarder modèle + encodeur + métadonnées de colonnes
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
from sklearn.model_selection import GroupShuffleSplit
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


def split_by_client(
    df: pd.DataFrame,
    group_col: str = "code_client",
    test_size: float = 0.2,
    random_state: int = 42,
):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def run_training_pipeline(
    data_path: str = "data/processed/training_set.csv",
    model_output_path: str = "src/models/classifier_lsat.joblib",
    encoder_output_path: str = "src/models/encoder_categorie.joblib",
    metadata_output_path: str = "src/models/classifier_lsat_metadata.json",
    test_size: float = 0.2,
    val_size: float = 0.1,
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
    cols_to_drop = [
        "code_client", "code_article", "designation",
        "categorie", "target_qty", "target_bought",
    ]
    existing_drop = [c for c in cols_to_drop if c in df.columns]
    feature_cols = [c for c in df.columns if c not in existing_drop]

    bool_cols = df[feature_cols].select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    logger.info(f"Nombre de features retenues : {len(feature_cols)} → {feature_cols}")

    # Étape 4 — Split PAR CLIENT (train/val/test)
    # Un split aléatoire classique (stratify sur les lignes) laisse le même
    # client apparaître à la fois dans train et dans test avec des produits
    # différents. Comme plusieurs features (recency, avg_qty, client_total_*)
    # sont calculées au niveau du client, le modèle peut alors mémoriser des
    # signatures de client plutôt que d'apprendre un pattern généralisable.
    # On force donc TOUTES les lignes d'un même client à rester du même côté
    # (train, val, ou test) via GroupShuffleSplit sur code_client.
    logger.info("Étape 4: Split train/val/test par client (anti-fuite)...")

    train_val_df, test_df = split_by_client(df, "code_client", test_size, random_state)
    train_df, val_df = split_by_client(
        train_val_df, "code_client", val_size / (1 - test_size), random_state
    )

    X_train, y_train = train_df[feature_cols], train_df["target_bought"]
    X_val, y_val = val_df[feature_cols], val_df["target_bought"]
    X_test, y_test = test_df[feature_cols], test_df["target_bought"]

    logger.info(
        f"Train: {X_train.shape} ({train_df['code_client'].nunique()} clients) | "
        f"Val: {X_val.shape} ({val_df['code_client'].nunique()} clients) | "
        f"Test: {X_test.shape} ({test_df['code_client'].nunique()} clients)"
    )

    overlap = (
        set(train_df["code_client"]) & set(test_df["code_client"])
    )
    if overlap:
        logger.error(f"FUITE DÉTECTÉE : {len(overlap)} clients partagés train/test !")
    else:
        logger.info("Vérification anti-fuite OK : aucun client partagé entre train et test.")

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
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=30,
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