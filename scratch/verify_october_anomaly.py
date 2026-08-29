import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/main_table.csv")
df["date_commande"] = pd.to_datetime(df["date_commande"])

print("================================================================")
print("             AUDIT DE L'ANOMALIE DU 13-20 OCTOBRE 2024          ")
print("================================================================")

daily = df.groupby(df["date_commande"].dt.date)["quantite"].agg(["sum", "count", "mean", "max"]).reset_index()
daily.columns = ["date", "total_qty", "nb_lignes", "avg_qty_per_line", "max_line_qty"]

mean_daily = daily["total_qty"].mean()
median_daily = daily["total_qty"].median()
std_daily = daily["total_qty"].std()
p95_daily = daily["total_qty"].quantile(0.95)
p99_daily = daily["total_qty"].quantile(0.99)

print(f"Statistiques globales journalières sur tout l'historique :")
print(f"  • Moyenne journalière       : {mean_daily:.1f} unités")
print(f"  • Médiane journalière       : {median_daily:.1f} unités")
print(f"  • Écart-type journalier     : {std_daily:.1f} unités")
print(f"  • 95e percentile            : {p95_daily:.1f} unités")
print(f"  • 99e percentile            : {p99_daily:.1f} unités")

mask_oct = (df["date_commande"] >= "2024-10-01") & (df["date_commande"] <= "2024-10-31")
oct_df = df[mask_oct]

print(f"\nTotal volume commandé sur tout le mois d'octobre 2024 : {oct_df['quantite'].sum():,.0f} unités")

mask_week = (df["date_commande"] >= "2024-10-13") & (df["date_commande"] <= "2024-10-20")
week_df = df[mask_week]

print(f"\nSemaine ciblée (13 au 20 octobre 2024) :")
print(f"  • Lignes de commande        : {len(week_df):,}")
print(f"  • Quantité totale           : {week_df['quantite'].sum():,.0f} unités")
print(f"  • Part du mois d'octobre    : {week_df['quantite'].sum() / oct_df['quantite'].sum() * 100:.1f}%")
print(f"  • Nombre de factures        : {week_df['code_facture'].nunique()}")
print(f"  • Nombre de clients distincts: {week_df['code_client'].nunique()}")

print("\n--- Les 5 commandes unitaires les plus massives de cette semaine ---")
top_orders = week_df.sort_values("quantite", ascending=False)[
    ["date_commande", "code_client", "code_facture", "code_article", "designation", "categorie", "quantite"]
].head(10)
print(top_orders.to_string())

print("\n--- Analyse des clients à méga-commandes (ex: CLT099780, CLT105703) ---")
for client_id in ["CLT099780", "CLT105703", "CLT106347"]:
    client_history = df[df["code_client"] == client_id]
    nb_cmd = client_history["code_facture"].nunique()
    total_qty = client_history["quantite"].sum()
    dates = client_history["date_commande"].dt.date.unique()
    qty_in_week = client_history[client_history["date_commande"].isin(week_df["date_commande"])]["quantite"].sum()
    print(f"Client {client_id} :")
    print(f"  • Nombre total de factures dans l'histoire : {nb_cmd}")
    print(f"  • Quantité totale achetée à vie            : {total_qty:,.0f} unités")
    print(f"  • Quantité achetée pendant CETTE semaine    : {qty_in_week:,.0f} unités ({qty_in_week / total_qty * 100:.1f}% de sa vie !)")
    print(f"  • Dates d'activité                         : {dates}")

print("\n--- Impact sur le coefficient de saisonnalité de la catégorie GSM NOKIA ---")
# Calcul avec et sans cette semaine
df_with = df.copy()
df_with["trimestre"] = df_with["date_commande"].dt.quarter
means_with = df_with.groupby(["categorie", "trimestre"])["quantite"].mean()
all_means_with = df_with.groupby("categorie")["quantite"].mean()
ratio_with = (means_with / all_means_with).unstack().loc["GSM NOKIA"]

df_without = df[~mask_week].copy()
df_without["trimestre"] = df_without["date_commande"].dt.quarter
means_without = df_without.groupby(["categorie", "trimestre"])["quantite"].mean()
all_means_without = df_without.groupby("categorie")["quantite"].mean()
ratio_without = (means_without / all_means_without).unstack().loc["GSM NOKIA"]

print("Coefficients GSM NOKIA AVEC l'anomalie :")
print(ratio_with.to_dict())
print("\nCoefficients GSM NOKIA SANS l'anomalie :")
print(ratio_without.to_dict())
