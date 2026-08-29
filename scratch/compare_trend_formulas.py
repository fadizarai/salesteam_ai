import pandas as pd
import numpy as np

# Load orders data
lignes = pd.read_csv("data/processed/lignes_clean.csv")
cmd = pd.read_csv("data/processed/commandes_clean.csv")
merged = lignes.merge(cmd, on="code_facture", how="left")
merged["date_commande"] = pd.to_datetime(merged["date_commande"])

# Define the 3 trend calculation formulas
def calc_midsplit_trend(qtys):
    n = len(qtys)
    if n < 2:
        return 0.0
    mid = n // 2
    old_avg = np.mean(qtys[:mid]) if mid > 0 else 0.0
    new_avg = np.mean(qtys[mid:]) if (n - mid) > 0 else 0.0
    trend = (new_avg - old_avg) / (old_avg + 1e-6)
    return float(np.clip(trend, -0.9, 3.0))

def calc_linear_slope_trend(qtys):
    n = len(qtys)
    if n < 2:
        return 0.0
    x = np.arange(n)
    y = np.array(qtys, dtype=float)
    # Linear regression slope: cov(x, y) / var(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    if y_mean == 0:
        return 0.0
    slope = np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean)**2) + 1e-6)
    # Normalized slope relative to mean
    norm_slope = slope / y_mean
    return float(np.clip(norm_slope, -0.9, 3.0))

def calc_ema_trend(qtys, span=3):
    n = len(qtys)
    if n < 2:
        return 0.0
    s = pd.Series(qtys, dtype=float)
    ema = s.ewm(span=min(span, n), adjust=False).mean().iloc[-1]
    base_mean = s.mean()
    if base_mean == 0:
        return 0.0
    ema_trend = (ema - base_mean) / base_mean
    return float(np.clip(ema_trend, -0.9, 3.0))

print("==========================================================================")
print("     COMPARAISON DES 3 FORMULES DE TREND SUR CAS DOCUMENTÉS               ")
print("==========================================================================")

# Test cases
test_cases = [
    ("CLT070730", "25078RA3EABLACK4/128", "REDMI 15C 4/128GB (Cas 11 commandes)"),
    ("CLT091206", "23129RN51XBLACK8/256", "REDMI NOTE 13 PRO 8/256"),
    ("CLT011712", "SP01Z07Z1946Y", "NOKIA 105 BLUE"),
    ("CLT100521", "24048RN6CGBLACK8/256", "POCO F6 PRO"),
]

for client_id, article_id, label in test_cases:
    sub = merged[(merged["code_client"] == client_id) & (merged["code_article"] == article_id)].sort_values("date_commande")
    if sub.empty:
        print(f"\n[!] Cas {client_id} / {article_id} non trouvé.")
        continue
    
    qtys = sub["quantite"].values
    t_mid = calc_midsplit_trend(qtys)
    t_slope = calc_linear_slope_trend(qtys)
    t_ema = calc_ema_trend(qtys)
    
    print(f"\n[*] Client {client_id} - Produit : {label}")
    print(f"   Commandes reelles ({len(qtys)} commandes) : {list(qtys)}")
    print(f"   * MidSplit (actuel) : {t_mid:+.3f}  ->  Status : {'[+] CROISSANCE' if t_mid > 0.1 else ('[-] DECLIN' if t_mid < -0.1 else '[=] STABLE')}")
    print(f"   * Linear Slope      : {t_slope:+.3f}  ->  Status : {'[+] CROISSANCE' if t_slope > 0.1 else ('[-] DECLIN' if t_slope < -0.1 else '[=] STABLE')}")
    print(f"   * EMA Ratio         : {t_ema:+.3f}  ->  Status : {'[+] CROISSANCE' if t_ema > 0.1 else ('[-] DECLIN' if t_ema < -0.1 else '[=] STABLE')}")

# Global dataset check
df_features = pd.read_csv("data/processed/feature_matrix.csv")
print(f"\n==========================================================================")
print(f"     IMPACT DU TREND_BOOST SUR RECOMMENDATION.PY                          ")
print(f"==========================================================================")
print("Dans recommendation.py :")
print("  if trend > 0.10:  trend_boost = 1.20 (+20% de score)")
print("  elif trend < -0.15: trend_boost = 0.70 (-30% de score)")
print("  else: trend_boost = 1.00 (neutre)")

# Compute how many pairs flip status in feature_matrix
mids = df_features["trend"].values
print(f"\nRepartition actuelle sur les {len(df_features):,} paires (client, produit) :")
print(f"  - En Croissance (trend > +0.10) : {(mids > 0.10).sum():,} ({(mids > 0.10).mean()*100:.1f}%)")
print(f"  - Neutres (-0.15 <= trend <= 0.10) : {((mids >= -0.15) & (mids <= 0.10)).sum():,} ({((mids >= -0.15) & (mids <= 0.10)).mean()*100:.1f}%)")
print(f"  - En Declin (trend < -0.15)     : {(mids < -0.15).sum():,} ({(mids < -0.15).mean()*100:.1f}%)")
