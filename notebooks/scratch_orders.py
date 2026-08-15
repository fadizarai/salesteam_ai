import pandas as pd
import numpy as np
import os

os.chdir(r"c:\Users\Fedy Zarai\.gemini\antigravity-ide\scratch\salesteam_ai")

lignes = pd.read_csv("data/processed/lignes_clean.csv")
cmd = pd.read_csv("data/processed/commandes_clean.csv")
merged = lignes.merge(cmd, on="code_facture", how="left")
sub = merged[(merged["code_client"] == "CLT070730") & (merged["code_article"].str.contains("25078RA3EABLACK4/128", na=False))]
print("=== COMMANDES BRUTES CLT070730 / 25078RA3EABLACK4/128 ===")
print(sub[["code_facture", "code_article", "quantite", "date_commande"]].sort_values("date_commande").to_string())
print()
print(f"Nb commandes: {len(sub)}")
print(f"Quantites: {sub['quantite'].tolist()}")
print(f"Min: {sub['quantite'].min()}, Max: {sub['quantite'].max()}, Moy: {sub['quantite'].mean():.1f}")
print(f"Std: {sub['quantite'].std():.1f}")
