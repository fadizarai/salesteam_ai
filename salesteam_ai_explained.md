# 🚀 SalesTeam AI — Guide Complet & Architecture de la Solution

## 📌 Présentation Globale du Projet
**SalesTeam AI** est un agent intelligent de recommandation d'ordres d'achat conçu pour l'application mobile Flutter des commerciaux de la filiale **LSAT (ITech)**.

Lorsqu'un commercial visite un point de vente client, l'agent IA génère automatiquement un **Projet de Commande sur-mesure**, prédisant les produits les plus susceptibles d'être achetés ainsi que les quantités optimales à proposer.

---

### 🏗️ Architecture du Système (4 Couches)
1. **Layer 1 — Data Ingestion & Cleaning** (`src/data/`) : Chargement des 18 444 factures et 78 530 lignes de commande LSAT, nettoyage et géolocalisation GPS.
2. **Layer 2 — Feature Engineering & Model Training** (`src/models/`) : Calcul des 26 features comportementales et entraînement du classifieur **XGBoost Classifier**.
3. **Layer 3 — Services Logiciel** (`src/services/`) : Inférence du modèle, filtrage par probabilité d'achat et génération d'explications en français.
4. **Layer 4 — API & Frontend Interface** (`src/api/` & `frontend/`) : API REST FastAPI (Endpoints `/recommend`, `/clients`, `/feedback`, `/health`) et Tableau de bord de test interactif en **React (Vite)**.

---

## 📊 1. Traitement des Données & Feature Engineering (Layer 1 & 2)

### A. Ingestion et Nettoyage (`src/data/cleaner.py`)
- **Volume de données** : ~78,530 lignes de commande LSAT nettoyées (période Jan 2024 -> Juin 2026).
- **Normalisation** : Suppression des doublons de factures, gestion des valeurs nulles (catégories inconnues -> `UNKNOWN`), et jointure avec les coordonnées GPS (325 points de vente géolocalisés).

### B. Matrice de Features & Variable Cible (`src/models/target_builder.py`)
Le dataset d'entraînement `data/processed/training_set.csv` rassemble **153 320 lignes** et **26 features** :
- **Historique d'Achat** : `frequency`, `recency_days`, `avg_qty`, `std_qty`, `min_qty`, `max_qty`, `total_qty`.
- **Comportement Client** : `client_total_products`, `client_total_invoices`, `client_avg_basket_size`, `days_since_first_order`.
- **Saisonnalité & Contexte** : `current_month_coef`, `avg_seasonal_coef`, `best_month`, `trend`.
- **Produit & Géographie** : `categorie_encoded`, `is_bulk_product`, `is_new_product`, `has_gps`, `latitude`, `longitude`.
- **Variable Cible (`target_bought`)** : `1` si le client a acheté l'article lors de la visite, `0` sinon.

---

## 🤖 2. Entraînement & Performances du Modèle XGBoost (`src/models/train_classifier.py`)

Le script [train_classifier.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/models/train_classifier.py) exécute le pipeline d'entraînement du modèle **XGBoost Classifier**.

### 📈 Résultats de l'Évaluation du Modèle :
```text
============================================================
      RÉSULTATS DE L'ÉVALUATION DU CLASSIFIEUR XGBOOST
============================================================
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     27557
           1       0.98      1.00      0.99      3107

    accuracy                           1.00     30664
   macro avg       0.99      1.00      0.99     30664
weighted avg       1.00      1.00      1.00     30664

ROC-AUC Score: 1.0000

Matrice de Confusion :
[[27502    55]
 [    3  3104]]
============================================================
```

---

## 🌐 3. Architecture API Backend FastAPI (`src/api/` & `src/services/`)

L'API FastAPI sert de passerelle temps réel pour communiquer avec l'application Flutter et l'interface React.

### Endpoints Implémentés :
- **`POST /api/recommend`** ([recommend.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/api/routes/recommend.py)) :
  - Reçoit un `client_id` + configuration IA.
  - Appelle [recommendation.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/services/recommendation.py) pour exécuter l'inférence XGBoost.
  - Calcule la probabilité d'achat (0% à 100%), filtre les suggestions prioritaires et génère l'explication explicite en français.
- **`GET /api/clients`** ([clients.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/api/routes/clients.py)) :
  - Retourne la liste des clients disponibles avec leurs statistiques d'achat pour alimenter les composants de sélection.
- **`POST /api/feedback`** ([feedback.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/api/routes/feedback.py)) :
  - Enregistre les réactions du commercial (articles acceptés, refusés ou quantités modifiées) dans `data/feedback/feedback_YYYY-MM.csv`.
- **`GET /health` & `POST /api/retrain`** ([admin.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/api/routes/admin.py)) :
  - Vérification de santé du serveur et déclenchement automatique du réentraînement du modèle.

---

## ⚛️ 4. Interface Utilisateur de Test en React (`frontend/`)

Nous avons construit un tableau de bord web interactif en **React (Vite)** avec un design **Glassmorphism Sombre** moderne :

- **Sélecteur de Clients** : Puces d'accès rapide (`CLT009160`, `CLT010283`, `CLT011029`, etc.) et recherche personnalisée.
- **Panneau de Contrôle IA** : Ajustement du nombre maximal de suggestions souhaitées (de 3 à 15).
- **Cartes KPI en Temps Réel** : Nombre d'articles retenus, nombre d'articles à Priorité Haute (>80%), et confiance moyenne IA.
- **Tableau "Projet de Commande"** :
  - Barres de probabilité couleur avec pourcentage de confiance d'achat.
  - Badges de priorité (**Priorité Haute** / **Recommandé**).
  - Quantités suggérées modifiables.
  - Explications automatiques en français.
  - Boutons interactifs (Accepter / Rejeter) pour simuler la validation du commercial sur le terrain.

---

## 🛠️ 5. Guide d'Utilisation & Commandes Utiles

### A. Lancer le Backend FastAPI :
```powershell
# Démarrer Uvicorn sur le port 8000
python -m uvicorn src.api.main:app --port 8000 --reload
```

### B. Lancer l'Interface Web React :
```powershell
cd frontend
npm run dev
# Accès navigateur : http://localhost:5173/
```

### C. Test Rapide d'Inférence en Ligne de Commande :
```powershell
# Générer un projet de commande dans le terminal pour un client spécifique
python recommend_for_client.py CLT009160
```

### D. Réentraîner le Modèle :
```powershell
python src/models/train_classifier.py
```
