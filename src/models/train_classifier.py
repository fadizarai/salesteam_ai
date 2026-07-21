"""
Script d'entraînement du classifieur XGBoost (Layer 2 — AI / ML)

Étapes exactes suivies :
Étape 1 — Charger le training_set (data/processed/training_set.csv)
Étape 2 — Encoder la variable catégorielle 'categorie' avec LabelEncoder et sauvegarder l'encodeur avec joblib
Étape 3 — Préparer les matrices X (features) et y (target_bought) en supprimant les identifiants et cibles
Étape 4 — Diviser le dataset en ensembles d'entraînement et de test (Stratified Train/Test Split)
Étape 5 — Calculer le ratio de déséquilibre des classes (scale_pos_weight)
Étape 6 — Entraîner le modèle XGBoost Classifier
Étape 7 — Évaluer les performances (Classification Report, ROC-AUC Score)
Étape 8 — Sauvegarder le modèle et l'encodeur pour la production
"""

import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
# pyrefly: ignore [missing-import]
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_training_pipeline(
    data_path: str = "data/processed/training_set.csv",
    model_output_path: str = "src/models/classifier_lsat.joblib",
    encoder_output_path: str = "src/models/encoder_categorie.joblib",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Exécute le pipeline complet d'entraînement du classifieur XGBoost.
    """
    # =========================================================================
    # Étape 1 — Charger le training_set
    # =========================================================================
    logger.info(f"Étape 1: Chargement des données depuis {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Le fichier {data_path} n'existe pas. Veuillez exécuter target_builder.py d'abord.")
    
    df = pd.read_csv(data_path)
    logger.info(f"Données chargées avec succès : {df.shape[0]} lignes, {df.shape[1]} colonnes.")

    # =========================================================================
    # Étape 2 — Encoder la colonne 'categorie'
    # =========================================================================
    logger.info("Étape 2: Encodage de la colonne 'categorie' avec LabelEncoder...")
    le_categorie = LabelEncoder()
    
    # Vérification si 'categorie' est présente dans le dataframe
    if "categorie" in df.columns:
        df["categorie"] = df["categorie"].fillna("UNKNOWN").astype(str)
        df["categorie_encoded"] = le_categorie.fit_transform(df["categorie"])
        logger.info(f"Nombre de catégories uniques encodées : {len(le_categorie.classes_)}")
    else:
        logger.warning("La colonne 'categorie' n'a pas été trouvée dans le dataframe.")

    # =========================================================================
    # Étape 3 — Dropper les colonnes non pertinentes et définir X et y
    # =========================================================================
    logger.info("Étape 3: Séparation des features (X) et de la cible (y)...")
    cols_to_drop = [
        "code_client", "code_article", "designation",
        "categorie",              # On garde categorie_encoded à la place
        "target_qty", "target_bought"
    ]
    
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    X = df.drop(columns=existing_cols_to_drop)
    y = df["target_bought"]

    # Convertir d'éventuelles colonnes booléennes en entiers (0/1)
    bool_cols = X.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        X[col] = X[col].astype(int)

    logger.info(f"Dimensions de la matrice de features X : {X.shape}")
    logger.info(f"Distribution de la cible y : {y.value_counts().to_dict()}")

    # =========================================================================
    # Étape 4 — Split Train / Test
    # =========================================================================
    logger.info(f"Étape 4: Séparation Train/Test ({int((1-test_size)*100)}% train / {int(test_size*100)}% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(f"Taille de X_train : {X_train.shape}, X_test : {X_test.shape}")

    # =========================================================================
    # Étape 5 — Calculer le déséquilibre des classes pour scale_pos_weight
    # =========================================================================
    logger.info("Étape 5: Calcul du ratio d'imbalance (scale_pos_weight)...")
    num_negatives = (y_train == 0).sum()
    num_positives = (y_train == 1).sum()
    scale_pos_weight = num_negatives / max(1, num_positives)
    logger.info(f"Nombre de classe 0 (Non-achat) : {num_negatives}")
    logger.info(f"Nombre de classe 1 (Achat)     : {num_positives}")
    logger.info(f"Ratio scale_pos_weight calculé : {scale_pos_weight:.2f}")

    # =========================================================================
    # Étape 6 — Entraîner le modèle XGBoost Classifier
    # =========================================================================
    logger.info("Étape 6: Entraînement du modèle XGBoost Classifier...")
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train, y_train)
    logger.info("Entraînement terminé avec succès !")

    # =========================================================================
    # Étape 7 — Évaluer le modèle sur l'ensemble de test
    # =========================================================================
    logger.info("Étape 7: Évaluation des performances du modèle...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc_score = roc_auc_score(y_test, y_proba)
    report_text = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("      RÉSULTATS DE L'ÉVALUATION DU CLASSIFIEUR XGBOOST")
    print("=" * 60)
    print(report_text)
    print(f"ROC-AUC Score: {auc_score:.4f}")
    print("\nMatrice de Confusion :")
    print(cm)
    print("=" * 60 + "\n")

    # =========================================================================
    # Étape 8 — Sauvegarder le modèle et l'encodeur
    # =========================================================================
    logger.info("Étape 8: Sauvegarde du modèle et de l'encodeur...")
    
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(encoder_output_path).parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, model_output_path)
    joblib.dump(le_categorie, encoder_output_path)
    
    root_model_path = "models/classifier_lsat.joblib"
    root_encoder_path = "models/encoder_categorie.joblib"
    Path(root_model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, root_model_path)
    joblib.dump(le_categorie, root_encoder_path)

    logger.info(f"Modèle sauvegardé dans : {model_output_path} (et {root_model_path})")
    logger.info(f"Encodeur sauvegardé dans : {encoder_output_path} (et {root_encoder_path})")

    return model, le_categorie


if __name__ == "__main__":
    run_training_pipeline()
