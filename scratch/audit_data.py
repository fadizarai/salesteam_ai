import sys, os, json
sys.path.insert(0, '.')
import pandas as pd

print("=== DATA FILES AUDIT ===\n")

proc = 'data/processed'
files_to_check = [
    'commandes_clean.csv',
    'gps_clean.csv',
    'lignes_clean.csv',
    'main_table.csv',
    'feature_matrix.csv',
    'training_set.csv',
]

results = {}
for fname in files_to_check:
    path = os.path.join(proc, fname)
    if not os.path.exists(path):
        print(f"MISSING: {fname}")
        continue
    df = pd.read_csv(path)
    n_clients = df['code_client'].nunique() if 'code_client' in df.columns else 'N/A'
    n_articles = df['code_article'].nunique() if 'code_article' in df.columns else 'N/A'
    date_range = ''
    for dc in ['date_commande', 'visit_date']:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors='coerce')
            date_range = f" | dates: {df[dc].min().date()} -> {df[dc].max().date()}"
            break
    print(f"{fname}: {len(df)} rows | clients={n_clients} | articles={n_articles}{date_range}")
    results[fname] = {'rows': len(df), 'clients': n_clients, 'articles': n_articles}

print("\n=== METADATA FILES ===\n")
for mf in ['src/models/classifier_lsat_metadata.json', 'src/models/regressor_lsat_metadata.json']:
    if os.path.exists(mf):
        with open(mf) as f:
            data = json.load(f)
        print(f"\n--- {mf} ---")
        print(json.dumps(data, indent=2))

print("\n=== CLIENT COUNT DISCREPANCY ===\n")
# gps_clean
gps = pd.read_csv('data/processed/gps_clean.csv')
print(f"gps_clean.csv unique clients: {gps['code_client'].nunique()}")
# commandes_clean
cmd = pd.read_csv('data/processed/commandes_clean.csv')
print(f"commandes_clean.csv unique clients: {cmd['code_client'].nunique()}")
# feature_matrix
fm = pd.read_csv('data/processed/feature_matrix.csv')
print(f"feature_matrix.csv unique clients: {fm['code_client'].nunique()}")
# training_set
ts = pd.read_csv('data/processed/training_set.csv')
print(f"training_set.csv unique clients: {ts['code_client'].nunique()}")

# Clients in commandes but not in feature_matrix
cmd_clients = set(cmd['code_client'].unique())
fm_clients = set(fm['code_client'].unique())
ts_clients = set(ts['code_client'].unique())
print(f"\nClients in commandes NOT in feature_matrix: {len(cmd_clients - fm_clients)}")
print(f"Clients in feature_matrix NOT in training_set: {len(fm_clients - ts_clients)}")
