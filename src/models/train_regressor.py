"""
Script d'entraînement du régresseur XGBoost (Layer 2 — AI / ML)

Ce script prédit LA QUANTITÉ à recommander, mais uniquement pour les
paires (client, produit) que le classifieur a déjà identifiées comme
"va acheter". Il est entraîné UNIQUEMENT sur les lignes où un achat
réel a eu lieu (target_qty > 0), jamais sur les négatifs.

Étape 1 — Charger training_set.csv et filtrer les positifs réels
Étape 2 — Charger l'encodeur de catégorie déjà entraîné (réutiliser le même)
Étape 3 — Préparer X (features) et y (target_qty)
Étape 4 — Split train/val/test par client (même logique anti-fuite que le classifieur)
Étape 5 — Entraîner XGBoost Regressor avec early stopping
Étape 6 — Évaluer (MAE, RMSE, comparaison contre baseline avg_qty)
Étape 7 — Sauvegarder le modèle et les métadonnées
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def split_by_client(
    df: pd.DataFrame,
    group_col: str = "code_client",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Coupe le dataset en train/test en forçant TOUS les achats d'un
    même client à rester du même côté. Même logique que le classifieur —
    empêche le modèle de "mémoriser" un client via d'autres produits.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def evaluate_regressor(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_baseline: pd.Series,
    label: str = "Test",
) -> dict:
    """
    Calcule MAE, RMSE, et compare contre la baseline avg_qty.

    La baseline est la prédiction la plus simple possible : suggérer
    la quantité moyenne historique du client pour ce produit. Si
    XGBoost ne fait pas mieux que cette baseline, le modèle n'apporte
    aucune valeur ajoutée et quelque chose est à revoir dans les features.

    MAE  (Mean Absolute Error) : en moyenne, de combien d'unités on
         se trompe. Une MAE de 2.3 signifie qu'on se trompe en moyenne
         de 2.3 unités sur la quantité suggérée.

    RMSE (Root Mean Squared Error) : même idée mais pénalise plus
         fortement les grosses erreurs. Un RMSE bien supérieur au MAE
         signifie qu'il y a quelques commandes très mal prédites.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mae_baseline = mean_absolute_error(y_true, y_baseline)
    rmse_baseline = np.sqrt(mean_squared_error(y_true, y_baseline))

    improvement_mae = (mae_baseline - mae) / mae_baseline * 100
    improvement_rmse = (rmse_baseline - rmse) / rmse_baseline * 100

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "mae_baseline": float(mae_baseline),
        "rmse_baseline": float(rmse_baseline),
        "improvement_mae_pct": float(improvement_mae),
        "improvement_rmse_pct": float(improvement_rmse),
    }

    print(f"\n{'-'*55}")
    print(f"  {label}")
    print(f"{'-'*55}")
    print(f"  MAE   XGBoost  : {mae:.3f} unités")
    print(f"  MAE   Baseline : {mae_baseline:.3f} unités  (juste avg_qty)")
    print(f"  Gain MAE       : {improvement_mae:+.1f}%")
    print(f"  RMSE  XGBoost  : {rmse:.3f}")
    print(f"  RMSE  Baseline : {rmse_baseline:.3f}")
    print(f"  Gain RMSE      : {improvement_rmse:+.1f}%")
    print(f"{'-'*55}")

    if improvement_mae < 0:
        logger.warning(
            "XGBoost est MOINS bon que la baseline avg_qty. "
            "Vérifier la qualité des features ou réduire le surapprentissage."
        )
    else:
        logger.info(
            f"XGBoost améliore la baseline de {improvement_mae:.1f}% sur MAE."
        )

    return metrics


def run_regressor_pipeline(
    data_path: str = "data/processed/training_set.csv",
    encoder_path: str = "src/models/encoder_categorie.joblib",
    model_output_path: str = "src/models/regressor_lsat.joblib",
    metadata_output_path: str = "src/models/regressor_lsat_metadata.json",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    # =========================================================================
    # Étape 1 — Charger training_set et filtrer les positifs réels uniquement
    # =========================================================================
    # Le régresseur n'a RIEN à apprendre des négatifs (target_qty = 0).
    # Les inclure forcerait le modèle à passer son temps à prédire des zéros,
    # ce qui est déjà fait par le classifieur. On travaille uniquement sur les
    # 15 536 lignes où un achat réel a eu lieu — c'est le sous-ensemble qui
    # contient l'information sur "combien".
    logger.info(f"Étape 1: Chargement et filtrage des positifs réels...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} introuvable. Exécuter target_builder.py d'abord.")

    df_full = pd.read_csv(data_path)
    logger.info(f"Dataset complet : {len(df_full)} lignes")

    df = df_full[df_full["target_qty"] > 0].copy()
    logger.info(
        f"Après filtrage positifs réels : {len(df)} lignes "
        f"({len(df)/len(df_full)*100:.1f}% du dataset total)"
    )

    if len(df) < 100:
        raise ValueError(
            f"Seulement {len(df)} positifs disponibles — pas assez pour entraîner."
        )

    # =========================================================================
    # Étape 2 — Réutiliser l'encodeur de catégorie déjà entraîné
    # =========================================================================
    # On réutilise EXACTEMENT le même LabelEncoder que le classifieur —
    # pas question d'en créer un nouveau. Si on en entraîne un nouveau,
    # une catégorie "GSM NOKIA" pourrait être encodée 3 dans le classifieur
    # et 7 dans le régresseur, ce qui casserait la cohérence du pipeline
    # de production (où les deux modèles partagent les mêmes données d'entrée).
    logger.info(f"Étape 2: Chargement de l'encodeur depuis {encoder_path}...")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"Encodeur introuvable : {encoder_path}. "
            "Exécuter train_classifier.py d'abord."
        )

    le_categorie = joblib.load(encoder_path)

    df["categorie"] = df["categorie"].fillna("UNKNOWN").astype(str)

    # Certaines catégories présentes dans les positifs pourraient ne pas avoir
    # été vues par le LabelEncoder si elles n'existaient que dans les négatifs.
    # On les remplace par "UNKNOWN" avant la transformation.
    known_categories = set(le_categorie.classes_)
    unknown_mask = ~df["categorie"].isin(known_categories)
    if unknown_mask.sum() > 0:
        logger.warning(
            f"{unknown_mask.sum()} lignes avec catégorie inconnue → remplacées par UNKNOWN"
        )
        df.loc[unknown_mask, "categorie"] = "UNKNOWN"

    df["categorie_encoded"] = le_categorie.transform(df["categorie"])

    # =========================================================================
    # Étape 3 — Préparer X et y
    # =========================================================================
    # y = target_qty : la VRAIE quantité commandée lors du prochain achat.
    # avg_qty est conservée séparément pour servir de baseline de comparaison.
    logger.info("Étape 3: Préparation de X (features) et y (target_qty)...")

    feature_cols = [
        "avg_qty",
        "std_qty",
        "min_qty",
        "max_qty",
        "last_qty",
        "frequency",
        "recency_days",
        "avg_delay_days",
        "current_month_coef",
        "avg_seasonal_coef"
    ]

    bool_cols = df[feature_cols].select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    X = df[feature_cols].copy()
    y = df["target_qty"].copy()
    baseline = df["avg_qty"].copy()

    logger.info(
        f"X shape : {X.shape} | "
        f"y : moyenne={y.mean():.1f}, médiane={y.median():.1f}, "
        f"max={y.max():.0f}"
    )

    # =========================================================================
    # Étape 4 — Split train/val/test par client
    # =========================================================================
    # On reconstruit les groupes client sur ce sous-ensemble filtré.
    # Même raison que le classifieur : les features client_* et recency_*
    # sont identiques pour toutes les lignes d'un client — un split par ligne
    # permettrait au modèle de mémoriser les signatures client.
    logger.info("Étape 4: Split train/val/test par client...")

    df["_y"] = y.values
    df["_baseline"] = baseline.values

    train_val_df, test_df = split_by_client(df, "code_client", test_size, random_state)
    train_df, val_df = split_by_client(
        train_val_df, "code_client", val_size / (1 - test_size), random_state
    )

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df["_y"]
    y_val = val_df["_y"]
    y_test = test_df["_y"]
    baseline_test = test_df["_baseline"]

    overlap = set(train_df["code_client"]) & set(test_df["code_client"])
    if overlap:
        logger.error(f"FUITE DÉTECTÉE : {len(overlap)} clients partagés train/test !")
    else:
        logger.info("Anti-fuite OK : aucun client partagé entre train et test.")

    logger.info(
        f"Train: {X_train.shape} ({train_df['code_client'].nunique()} clients) | "
        f"Val: {X_val.shape} ({val_df['code_client'].nunique()} clients) | "
        f"Test: {X_test.shape} ({test_df['code_client'].nunique()} clients)"
    )

    # =========================================================================
    # Étape 5 — Entraîner XGBoost Regressor
    # =========================================================================
    # objective="reg:squarederror" : minimise l'erreur quadratique moyenne —
    # c'est l'objectif standard pour prédire une quantité continue.
    #
    # eval_metric="mae" : pendant l'entraînement, on surveille le MAE sur la
    # validation (pas le RMSE) parce que le MAE est plus interprétable dans
    # un contexte commercial — "je me trompe de X unités en moyenne" — et
    # moins sensible aux quelques très grandes commandes outliers.
    #
    # early_stopping_rounds=30 : si le MAE sur la validation ne s'améliore
    # pas pendant 30 arbres consécutifs, on s'arrête — évite le surapprentissage
    # sans avoir à deviner le bon nombre d'arbres à la main.
    logger.info("Étape 5: Entraînement XGBoost Regressor...")
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
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

    # =========================================================================
    # Étape 6 — Évaluation
    # =========================================================================
    logger.info("Étape 6: Évaluation du régresseur...")
    y_pred_raw = model.predict(X_test)

    # Les quantités ne peuvent pas être négatives — on clip à 0.
    # XGBoost peut produire de légères valeurs négatives sur des produits rares.
    y_pred = np.clip(y_pred_raw, a_min=0, a_max=None)

    # On arrondit aussi à l'entier le plus proche parce qu'on ne commande
    # pas 4.7 unités — dans le contexte business, la quantité est toujours entière.
    y_pred_rounded = np.round(y_pred).astype(int)

    print("\n" + "=" * 60)
    print("      RÉSULTATS DU RÉGRESSEUR XGBOOST")
    print("=" * 60)

    metrics_continuous = evaluate_regressor(
        y_test, y_pred, baseline_test,
        label="Prédiction continue (avant arrondi)"
    )
    metrics_rounded = evaluate_regressor(
        y_test, y_pred_rounded, baseline_test,
        label="Prédiction arrondie à l'entier (résultat final)"
    )

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    print("\nTop 15 features les plus importantes (régresseur) :")
    print(importances.head(15).to_string())

    print("\nDistribution des quantités prédites (arrondies) :")
    pred_series = pd.Series(y_pred_rounded)
    print(pred_series.describe().to_string())
    print("=" * 60 + "\n")

    # =========================================================================
    # Étape 7 — Sauvegarde
    # =========================================================================
    logger.info("Étape 7: Sauvegarde du modèle et des métadonnées...")

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)

    root_model_path = "models/regressor_lsat.joblib"
    Path(root_model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, root_model_path)

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "best_iteration": int(model.best_iteration),
        "n_training_positives": int(len(df)),
        "metrics_continuous": metrics_continuous,
        "metrics_rounded": metrics_rounded,
        "train_clients": int(train_df["code_client"].nunique()),
        "test_clients": int(test_df["code_client"].nunique()),
        "note": (
            "Ce modèle est entraîné UNIQUEMENT sur les achats réels (target_qty > 0). "
            "Il doit être appelé APRÈS le classifieur, uniquement pour les paires "
            "que le classifieur a prédit comme achat=1."
        ),
    }

    with open(metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"Modèle sauvegardé : {model_output_path} (+ {root_model_path})")
    logger.info(f"Métadonnées sauvegardées : {metadata_output_path}")

    return model, metadata


if __name__ == "__main__":
    run_regressor_pipeline()
