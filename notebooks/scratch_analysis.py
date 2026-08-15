import pandas as pd
import numpy as np
import joblib
import os, sys

os.chdir(r"c:\Users\Fedy Zarai\.gemini\antigravity-ide\scratch\salesteam_ai")

# Charger donnees
df = pd.read_csv("data/processed/training_set.csv")
reg = joblib.load("src/models/regressor_lsat.joblib")

# Features
regressor_features = ["avg_qty", "std_qty", "min_qty", "max_qty", "last_qty",
                      "frequency", "recency_days", "avg_delay_days",
                      "current_month_coef", "avg_seasonal_coef"]

# Feature importances
importances = pd.Series(reg.feature_importances_, index=regressor_features).sort_values(ascending=False)
print("=== FEATURE IMPORTANCES ===")
for feat, imp in importances.items():
    print(f"  {feat:25s} {imp:.4f}")

# Stats du produit specifique pour le client
client = df[df["code_client"] == "CLT070730"].copy()
if "visit_date" in client.columns:
    client["visit_date"] = pd.to_datetime(client["visit_date"])
    client = client.sort_values("visit_date").drop_duplicates(subset=["code_article"], keep="last")

prod = client[client["code_article"] == "25078RA3EABLACK4/128"]

print()
print("=== FEATURES ENTREE DU REGRESSEUR (pour ce produit) ===")
row = prod.iloc[0]
for f in regressor_features:
    print(f"  {f:25s} = {row[f]}")

# Stats globales sur les positifs du dataset
pos = df[df["target_qty"] > 0]
print()
print("=== STATS GLOBALES target_qty (positifs) ===")
print(pos["target_qty"].describe())
tq = pos["target_qty"]
print()
print(f"Mediane target_qty: {tq.median():.1f}")
print(f"P90 target_qty: {tq.quantile(0.9):.1f}")
print(f"P95 target_qty: {tq.quantile(0.95):.1f}")
print(f"P99 target_qty: {tq.quantile(0.99):.1f}")

# Historique des commandes brutes pour ce client/produit
print()
print("=== LIGNES BRUTES (commandes_clean) pour CLT070730 + produit ===")
try:
    lignes = pd.read_csv("data/processed/lignes_clean.csv")
    cmd = pd.read_csv("data/processed/commandes_clean.csv")
    merged = lignes.merge(cmd, on="num_commande", how="left")
    sub = merged[(merged["code_client"] == "CLT070730") & (merged["code_article"].str.contains("25078RA3EABLACK4/128", na=False))]
    print(sub[["num_commande", "code_article", "quantite", "date_commande"]].to_string())
except Exception as e:
    print(f"Erreur: {e}")
