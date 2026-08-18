"""
Script d'évaluation comparative empirique des seuils de fallback CV :
  - Scénario A : Seuil de référence actuel CV > 1.0
  - Scénario B : Seuil candidat CV > 0.8

Objectif :
Mesurer l'impact net sur le MAE et le RMSE du jeu de test (split par client),
analyser la tranche frontière (0.8 < CV <= 1.0), et vérifier les 4 cas clients documentés.
"""

import os
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_cv_comparison(
    data_path: str = "data/processed/training_set.csv",
    model_path: str = "src/models/regressor_lsat.joblib",
    metadata_path: str = "src/models/regressor_lsat_metadata.json",
    test_size: float = 0.2,
    random_state: int = 42,
):
    print("=" * 80)
    print(" EVALUATION COMPARATIVE : SEUIL CV > 1.0 vs SEUIL CV > 0.8")
    print("=" * 80)

    # 1. Chargement des données
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fichier de données {data_path} introuvable.")

    df_full = pd.read_csv(data_path)
    df_pos = df_full[df_full["target_qty"] > 0].copy()
    logger.info(f"Dataset total : {len(df_full):,} lignes | Positifs réels : {len(df_pos):,} lignes")

    # 2. Chargement du modèle régresseur
    if not os.path.exists(model_path):
        alt_model_path = "models/regressor_lsat.joblib"
        if os.path.exists(alt_model_path):
            model_path = alt_model_path
        else:
            raise FileNotFoundError(f"Modèle régresseur {model_path} introuvable.")

    regressor = joblib.load(model_path)
    logger.info(f"Modèle chargé depuis : {model_path}")

    # Feature columns
    feature_cols = [
        "avg_qty",
        "median_qty",
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

    if "median_qty" not in df_pos.columns:
        df_pos["median_qty"] = df_pos["avg_qty"]

    bool_cols = df_pos[feature_cols].select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        df_pos[col] = df_pos[col].astype(int)

    # 3. Split par client identique à train_regressor.py
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df_pos, groups=df_pos["code_client"]))
    
    train_df = df_pos.iloc[train_idx].copy()
    test_df = df_pos.iloc[test_idx].copy()

    logger.info(f"Train set : {len(train_df):,} lignes ({train_df['code_client'].nunique()} clients)")
    logger.info(f"Test set  : {len(test_df):,} lignes ({test_df['code_client'].nunique()} clients)")

    # 4. Calcul de CV et prédictions
    test_df["cv"] = np.where(test_df["avg_qty"] > 0, test_df["std_qty"] / test_df["avg_qty"], 999.0)
    
    X_test = test_df[feature_cols].copy()
    y_true = test_df["target_qty"].values
    y_baseline = np.ceil(test_df["avg_qty"].values)

    raw_preds = regressor.predict(X_test)
    raw_preds_clipped = np.clip(np.round(raw_preds), 1, None)

    # Scénario A : CV > 1.0 (Actuel)
    mask_cv10 = test_df["cv"] > 1.0
    y_pred_cv10 = np.where(mask_cv10, y_baseline, raw_preds_clipped)

    # Scénario B : CV > 0.8 (Candidat)
    mask_cv08 = test_df["cv"] > 0.8
    y_pred_cv08 = np.where(mask_cv08, y_baseline, raw_preds_clipped)

    # 5. Métriques globales sur l'ensemble du jeu de test
    mae_base = mean_absolute_error(y_true, y_baseline)
    rmse_base = np.sqrt(mean_squared_error(y_true, y_baseline))

    mae_raw = mean_absolute_error(y_true, raw_preds_clipped)
    rmse_raw = np.sqrt(mean_squared_error(y_true, raw_preds_clipped))

    mae_cv10 = mean_absolute_error(y_true, y_pred_cv10)
    rmse_cv10 = np.sqrt(mean_squared_error(y_true, y_pred_cv10))

    mae_cv08 = mean_absolute_error(y_true, y_pred_cv08)
    rmse_cv08 = np.sqrt(mean_squared_error(y_true, y_pred_cv08))

    gain_raw = (mae_base - mae_raw) / mae_base * 100
    gain_cv10 = (mae_base - mae_cv10) / mae_base * 100
    gain_cv08 = (mae_base - mae_cv08) / mae_base * 100

    print("\n" + "=" * 80)
    print(" 1. PERFORMANCE GLOBALE SUR L'ENSEMBLE DU JEU DE TEST (8,843 lignes)")
    print("=" * 80)
    print(f"{'Approche':<38} | {'MAE (unites)':<13} | {'Gain vs Base':<13} | {'RMSE':<12}")
    print("-" * 80)
    print(f"{'Baseline simple (avg_qty)':<38} | {mae_base:<13.3f} | {'---':<13} | {rmse_base:<12.3f}")
    print(f"{'XGBoost pur (sans fallback CV)':<38} | {mae_raw:<13.3f} | {gain_raw:+6.2f}%      | {rmse_raw:<12.3f}")
    print(f"{'Scenario A : Hybride CV > 1.0 (Actuel)':<38} | {mae_cv10:<13.3f} | {gain_cv10:+6.2f}%      | {rmse_cv10:<12.3f}")
    print(f"{'Scenario B : Hybride CV > 0.8 (Candidat)':<38} | {mae_cv08:<13.3f} | {gain_cv08:+6.2f}%      | {rmse_cv08:<12.3f}")
    print("-" * 80)

    # 6. Analyse specifique de la tranche frontiere : 0.8 < CV <= 1.0
    frontier_mask = (test_df["cv"] > 0.8) & (test_df["cv"] <= 1.0)
    frontier_df = test_df[frontier_mask].copy()
    nb_frontier = len(frontier_df)
    pct_frontier = nb_frontier / len(test_df) * 100

    print("\n" + "=" * 80)
    print(f" 2. ANALYSE DE LA TRANCHE FRONTIERE : 0.8 < CV <= 1.0 ({nb_frontier} lignes / {pct_frontier:.2f}% du test set)")
    print("=" * 80)
    print("Ce sont les seules lignes dont la decision bascule entre CV > 1.0 (IA) et CV > 0.8 (Historique).")

    if nb_frontier > 0:
        y_true_f = frontier_df["target_qty"].values
        y_base_f = np.ceil(frontier_df["avg_qty"].values)
        y_xgb_f = raw_preds_clipped[frontier_mask]

        mae_xgb_f = mean_absolute_error(y_true_f, y_xgb_f)
        mae_base_f = mean_absolute_error(y_true_f, y_base_f)
        rmse_xgb_f = np.sqrt(mean_squared_error(y_true_f, y_xgb_f))
        rmse_base_f = np.sqrt(mean_squared_error(y_true_f, y_base_f))

        print(f"\n  * Sur ces {nb_frontier} lignes frontieres :")
        print(f"      MAE avec XGBoost (Scenario CV > 1.0) : {mae_xgb_f:.3f} unites")
        print(f"      MAE avec Moyenne (Scenario CV > 0.8) : {mae_base_f:.3f} unites")
        print(f"      RMSE avec XGBoost                    : {rmse_xgb_f:.3f}")
        print(f"      RMSE avec Moyenne                    : {rmse_base_f:.3f}")

        diff_mae = mae_xgb_f - mae_base_f
        if diff_mae > 0:
            print(f"\n  ==> Le fallback Moyenne (CV > 0.8) est PLUS PRECIS de {diff_mae:.3f} unites (MAE) sur cette tranche.")
        else:
            print(f"\n  ==> XGBoost (CV > 1.0) est PLUS PRECIS de {-diff_mae:.3f} unites (MAE) sur cette tranche.")
    else:
        print("  Aucune ligne dans la tranche frontière.")

    # 7. Vérification des 4 cas clients documentés
    print("\n" + "=" * 80)
    print(" 3. IMPACT SUR LES 4 CAS CLIENTS DOCUMENTÉS")
    print("=" * 80)

    target_clients = ["CLT070730", "CLT091206", "CLT106366", "CLT109532"]

    for cid in target_clients:
        clt_data = df_pos[df_pos["code_client"] == cid].copy()
        clt_data["cv"] = np.where(clt_data["avg_qty"] > 0, clt_data["std_qty"] / clt_data["avg_qty"], 999.0)
        
        # Inférence
        X_clt = clt_data[feature_cols].copy()
        clt_raw = np.clip(np.round(regressor.predict(X_clt)), 1, None)
        clt_base = np.ceil(clt_data["avg_qty"].values)
        
        clt_data["source_cv10"] = np.where(clt_data["cv"] > 1.0, "historique", "IA")
        clt_data["source_cv08"] = np.where(clt_data["cv"] > 0.8, "historique", "IA")
        clt_data["changed"] = clt_data["source_cv10"] != clt_data["source_cv08"]

        changed_rows = clt_data[clt_data["changed"]]
        
        print(f"\n>> Client {cid} ({len(clt_data)} achats historiques) :")
        print(f"   - Articles avec CV > 1.0 (Historique sous les deux seuils) : {(clt_data['cv'] > 1.0).sum()}")
        print(f"   - Articles avec CV <= 0.8 (IA sous les deux seuils)         : {(clt_data['cv'] <= 0.8).sum()}")
        print(f"   - Articles qui BASCULENT de IA a Historique (0.8 < CV <= 1.0): {len(changed_rows)}")

        if len(changed_rows) > 0:
            print("   Detail des articles qui basculent :")
            for idx_val, r in changed_rows.head(5).iterrows():
                des = r.get("designation", r["code_article"])
                loc_idx = clt_data.index.get_loc(idx_val)
                pred_ia = clt_raw[loc_idx]
                pred_hist = clt_base[loc_idx]
                real = r["target_qty"]
                print(f"     * {r['code_article']} ({des[:25]}) : CV={r['cv']:.2f} | XGBoost={pred_ia:.0f} vs avg_qty={pred_hist:.0f} | Reel={real:.0f}")
        else:
            print("   -> Aucun article ne change de source pour ce client.")

    print("\n" + "=" * 80)
    print(" FIN DE L'ÉVALUATION")
    print("=" * 80)


if __name__ == "__main__":
    run_cv_comparison()
