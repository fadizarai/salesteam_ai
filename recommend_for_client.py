"""
Script de génération de Projet de Commande par Client (SalesTeam AI)
Permet de tester le modèle en entrant un code client (ex: CLT009160).
"""

import sys
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "src/models/classifier_lsat.joblib"
ENCODER_PATH = "src/models/encoder_categorie.joblib"
DATA_PATH = "data/processed/training_set.csv"


def generate_order_proposal(client_id: str = "CLT009160", threshold: float = 0.50):
    """
    Génère un projet de commande automatique pour un client donné.
    """
    print("=" * 80)
    print(f"       SALESTEAM AI — PROJET DE COMMANDE AUTOMATIQUE")
    print(f"       CLIENT SÉLECTIONNÉ : {client_id}")
    print("=" * 80)

    # 1. Chargement des modèles
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    # 2. Filtrer le dataset pour ce client
    # Remarque : En production, les données viennent de la base de données.
    df_all = pd.read_csv(DATA_PATH)
    df_client = df_all[df_all["code_client"] == client_id].copy()

    if df_client.empty:
        print(f"[ERREUR] Aucun historique trouvé pour le client '{client_id}'.")
        print(f"Clients d'exemples disponibles: CLT009160, CLT010283, CLT011029, CLT011090")
        return

    print(f"-> {len(df_client)} produits analysés dans l'historique du client {client_id}.\n")

    # 3. Préparation des données pour le modèle XGBoost
    df_client["categorie"] = df_client["categorie"].fillna("UNKNOWN").astype(str)
    
    # Encodage sécurisé (gestion des catégories inconnues)
    known_classes = set(encoder.classes_)
    df_client["categorie_clean"] = df_client["categorie"].apply(lambda c: c if c in known_classes else "UNKNOWN")
    df_client["categorie_encoded"] = encoder.transform(df_client["categorie_clean"])

    cols_to_drop = [
        "code_client", "code_article", "designation",
        "categorie", "categorie_clean", "target_qty", "target_bought"
    ]
    existing = [c for c in cols_to_drop if c in df_client.columns]
    X = df_client.drop(columns=existing)

    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(int)

    # 4. Inférence avec le modèle XGBoost
    probabilities = model.predict_proba(X)[:, 1]
    df_client["probabilite_achat"] = probabilities
    df_client["score_pct"] = (probabilities * 100).round(1)

    # 5. Sélection et tri des produits recommandés (probabilité >= threshold)
    propositions = df_client[df_client["probabilite_achat"] >= threshold].sort_values(
        by="probabilite_achat", ascending=False
    )

    # 6. Affichage du Projet de Commande
    print("--------------------------------------------------------------------------------")
    print("                     PROJET DE COMMANDE PROPOSÉ AU COMMERCIAL                   ")
    print("--------------------------------------------------------------------------------")
    
    if propositions.empty:
        print("Aucun produit recommandé au-dessus du seuil de confiance (50%).")
    else:
        print(f"{'CODE ARTICLE':<16} | {'DÉSIGNATION ARTICLE':<30} | {'PROBA':<7} | {'STATUT DE RECOMMANDATION'}")
        print("-" * 80)
        for _, row in propositions.iterrows():
            code_art = str(row["code_article"])[:15]
            designation = str(row["designation"])[:28]
            proba = f"{row['probabilite_achat'] * 100:.1f}%"
                
            if row['probabilite_achat'] >= 0.80:
                statut = "PRIORITE HAUTE (Achat tres probable)"
            else:
                statut = "PROPOSITION SECONDAIRE"
                
            print(f"{code_art:<16} | {designation:<30} | {proba:<7} | {statut}")

    print("=" * 80)
    print(f"Total articles proposes dans ce projet de commande : {len(propositions)} / {len(df_client)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    target_client = sys.argv[1] if len(sys.argv) > 1 else "CLT009160"
    generate_order_proposal(target_client)
