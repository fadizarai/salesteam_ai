import pandas as pd
import numpy as np

# Load orders data
lignes = pd.read_csv("data/processed/lignes_clean.csv")
cmd = pd.read_csv("data/processed/commandes_clean.csv")
merged = lignes.merge(cmd, on="code_facture", how="left")
merged["date_commande"] = pd.to_datetime(merged["date_commande"])
merged = merged.sort_values(by=["code_client", "code_article", "date_commande"])

def calc_midsplit(qtys):
    n = len(qtys)
    if n < 2: return 0.0
    mid = n // 2
    old_avg = np.mean(qtys[:mid]) if mid > 0 else 0.0
    new_avg = np.mean(qtys[mid:]) if (n - mid) > 0 else 0.0
    return float(np.clip((new_avg - old_avg) / (old_avg + 1e-6), -0.9, 3.0))

def calc_linear_slope(qtys):
    n = len(qtys)
    if n < 2: return 0.0
    x = np.arange(n)
    y = np.array(qtys, dtype=float)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    if y_mean == 0: return 0.0
    slope = np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean)**2) + 1e-6)
    return float(np.clip(slope / y_mean, -0.9, 3.0))

def calc_ema(qtys, span=3):
    n = len(qtys)
    if n < 2: return 0.0
    s = pd.Series(qtys, dtype=float)
    ema = s.ewm(span=min(span, n), adjust=False).mean().iloc[-1]
    base_mean = s.mean()
    if base_mean == 0: return 0.0
    return float(np.clip((ema - base_mean) / base_mean, -0.9, 3.0))

# Sample 5000 pairs with frequency >= 3 to analyze status agreement
grouped = merged.groupby(["code_client", "code_article"])["quantite"].apply(list)
frequent = grouped[grouped.apply(len) >= 3]

results = []
for (client_id, article_id), qtys in frequent.items():
    m = calc_midsplit(qtys)
    s = calc_linear_slope(qtys)
    e = calc_ema(qtys)
    
    # Classify status (-1: Declin, 0: Neutre, +1: Croissance)
    stat_m = 1 if m > 0.10 else (-1 if m < -0.15 else 0)
    stat_s = 1 if s > 0.10 else (-1 if s < -0.15 else 0)
    stat_e = 1 if e > 0.10 else (-1 if e < -0.15 else 0)
    
    results.append({
        "client": client_id,
        "article": article_id,
        "n": len(qtys),
        "midsplit": m,
        "slope": s,
        "ema": e,
        "stat_m": stat_m,
        "stat_s": stat_s,
        "stat_e": stat_e,
        "agree_all": (stat_m == stat_s == stat_e),
        "disagree_m_s": (stat_m != stat_s),
        "disagree_m_e": (stat_m != stat_e)
    })

df_res = pd.DataFrame(results)

print("==========================================================================")
print(f"       ANALYSE STATISTIQUE GLOBALE SUR {len(df_res):,} PAIRES ACTIVES (FREQ >= 3)")
print("==========================================================================")
print(f"1. Taux d'accord total sur le statut (Croissance / Neutre / Declin) :")
print(f"   - Accord unanime (3 formules d'accord)   : {df_res['agree_all'].mean()*100:.1f}%")
print(f"   - Desaccord MidSplit vs Linear_Slope     : {df_res['disagree_m_s'].mean()*100:.1f}%")
print(f"   - Desaccord MidSplit vs EMA_Ratio        : {df_res['disagree_m_e'].mean()*100:.1f}%")

print("\n2. Pourquoi elles ne sont pas interchangeables ? (Exemple type de desaccord)")
# Find an interesting disagreement case
dis = df_res[df_res["stat_m"] != df_res["stat_s"]].head(5)
for _, r in dis.iterrows():
    qtys = merged[(merged["code_client"] == r["client"]) & (merged["code_article"] == r["article"])]["quantite"].values
    status_map = {1: "CROISSANCE", 0: "NEUTRE", -1: "DECLIN"}
    print(f"\n- Client {r['client']} / {r['article']} ({r['n']} cmd) : {list(qtys)}")
    print(f"  * MidSplit     : {r['midsplit']:+.2f} ({status_map[r['stat_m']]})")
    print(f"  * Linear Slope : {r['slope']:+.2f} ({status_map[r['stat_s']]})")
    print(f"  * EMA Ratio    : {r['ema']:+.2f} ({status_map[r['stat_e']]})")
