# 📊 SalesTeam AI — État d'Avancement & Document de Suivi (`PROGRESS.md`)

Ce document centralise l'état réel du projet **SalesTeam AI**, l'architecture technique, l'audit complet du codebase, les choix technologiques et la feuille de route des tâches à venir.

---

## 📌 1. Vue d'Ensemble & Architecture

**SalesTeam AI** est un système de recommandation d'ordres d'achat automatisé conçu pour assister les commerciaux de la filiale **LSAT (ITech)** lors de leurs visites en points de vente client.

### Architecture à 4 Couches :
```
[ Données Brutes (Excel) ]
           │
           ▼
[ Layer 1: Ingestion & Nettoyage ] (src/data/cleaner.py)
           │
           ▼
[ Layer 2: Feature Engineering & Modèle ] (src/features/ & src/models/)
  ├── 26 features comportementales, temporelles & géographiques
  └── XGBoost Classifier (prédiction target_bought)
           │
           ▼
[ Layer 3: Services Métier ] (src/services/)
  ├── Inférence temps réel & filtrage par seuil
  ├── Explications produit (Règles métiers & Stub LLM)
  └── Stockage du Feedback commercial
           │
           ▼
[ Layer 4: API REST & Interface ] (src/api/ & frontend/)
  ├── Backend FastAPI (Endpoints /recommend, /clients, /feedback, /retrain)
  └── Frontend Web React Vite (Tableau de bord interactif Glassmorphism)
```

---

## 🔍 2. Audit Complet des Fichiers du Projet

### 📂 Racine du Projet (`/`)
* [README.md](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/README.md) — Documentation générale du projet et guides de lancement. `[Fonctionnel]`
* [PROGRESS.md](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/PROGRESS.md) — Suivi de projet et audit de l'avancement. `[Fonctionnel]`
* [salesteam_ai_explained.md](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/salesteam_ai_explained.md) — Architecture et explications détaillées de la solution. `[Fonctionnel]`
* [recommend_for_client.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/recommend_for_client.py) — Script d'inférence CLI pour tester les recommandations par code client. `[Fonctionnel]`
* [requirements.txt](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/requirements.txt) — Dépendances Python du projet (FastAPI, XGBoost, pandas, etc.). `[Fonctionnel]`

### 📂 Layer 1 — Données (`src/data/` & `data/`)
* [src/data/loader.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/data/loader.py) — Ingestion et lecture des 3 fichiers Excel bruts. `[Fonctionnel]`
* [src/data/cleaner.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/data/cleaner.py) — Nettoyage, normalisation des types, gestion des nulls, calcul des agrégats et sauvegarde CSV. `[Fonctionnel]`
* `data/raw/` — Reçoit les fichiers Excel bruts (18 444 factures, 78 530 lignes, 325 coordonnées GPS). `[Fonctionnel]`
* `data/processed/` — Stocke les CSV nettoyés (`commandes_clean.csv`, `lignes_clean.csv`, `gps_clean.csv`, `training_set.csv`). `[Fonctionnel]`
* `data/feedback/` — Dossier de réception des retours commerciaux mensuels (`feedback_YYYY-MM.csv`). `[Fonctionnel]`

### 📂 Layer 2 — Features & Modèle ML (`src/features/` & `src/models/` & `models/`)
* [src/features/feature_engineering.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/features/feature_engineering.py) — Génération de 26 features (récence, fréquence, tendance, saisonnalité, panier moyen, géolocalisation). `[Fonctionnel]`
* [src/models/target_builder.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/models/target_builder.py) — Construction de la cible `target_bought` et échantillonnage négatif (ratio 3:1). `[Fonctionnel]`
* [src/models/train_classifier.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/models/train_classifier.py) — Entraînement de XGBoost avec split par client (`GroupShuffleSplit`), `scale_pos_weight` et sauvegarde des artefacts. `[Fonctionnel]`
* [src/models/classifier_lsat.joblib](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/models/classifier_lsat.joblib) — Modèle XGBoost entraîné et sérialisé. `[Fonctionnel]`
* [src/models/encoder_categorie.joblib](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/models/encoder_categorie.joblib) — LabelEncoder pour les catégories d'articles. `[Fonctionnel]`

### 📂 Layer 3 — Services Métier (`src/services/`)
* [src/services/recommendation.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/services/recommendation.py) — Service d'inférence principal (chargement artefacts, prédiction probabilité, filtrage, génération explications). `[Fonctionnel]`
* [src/services/feedback.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/services/feedback.py) — Enregistrement des validations/refus commerciaux dans les CSV de feedback et chargement pour réentraînement. `[Fonctionnel]`
* [src/services/explanation.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/services/explanation.py) — Service d'explication par modèle LLM HuggingFace / règles avancées avec cache. `[Incomplet - Raise NotImplementedError]`

### 📂 Layer 4 — API REST & Web Frontend (`src/api/` & `frontend/`)
* [src/api/main.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/api/main.py) — Point d'entrée FastAPI avec CORS middleware, logging et enregistrement des routes. `[Fonctionnel]`
* [src/api/schemas.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/api/schemas.py) — Modèles Pydantic de validation des requêtes et réponses HTTP. `[Fonctionnel]`
* [src/api/routes/recommend.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/api/routes/recommend.py) — Endpoint `POST /api/recommend` pour générer le projet de commande. `[Fonctionnel]`
* [src/api/routes/clients.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/api/routes/clients.py) — Endpoint `GET /api/clients` pour lister les clients du dataset. `[Fonctionnel]`
* [src/api/routes/feedback.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/api/routes/feedback.py) — Endpoint `POST /api/feedback` pour enregistrer les retours. `[Fonctionnel]`
* [src/api/routes/admin.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/api/routes/admin.py) — Endpoints `GET /health` et `POST /api/retrain`. `[Fonctionnel]`
* [frontend/src/App.jsx](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/frontend/src/App.jsx) — Application React (Vite) avec tableau de bord interactif, cartes KPI, sélecteur client, et simulation de validation de commande. `[Fonctionnel]`

---

## 📊 3. Bilan de l'Avancement

### ✅ 1. Ce qui est Terminé et Fonctionnel
* **Pipeline Data complet** : Nettoyage, imputations des catégories et agrégations sur les ~78 530 lignes LSAT.
* **Feature Store** : Matrice de 153 320 échantillons x 26 caractéristiques (fréquence, récence, tendance, saisonnalité).
* **Modèle ML** : Classification XGBoost entraînée avec séparation stricte par client (`GroupShuffleSplit`) pour garantir l'absence de fuite d'information.
* **API FastAPI** : Endpoints REST `/api/recommend`, `/api/clients`, `/api/feedback`, `/health`, `/api/retrain`.
* **Frontend React** : Interface moderne (Glassmorphism Sombre) connectée à l'API backend.
* **CLI de Test** : Script `recommend_for_client.py` opérationnel pour des vérifications immédiates en ligne de commande.

### 🟡 2. Ce qui est En Cours / Partiellement Fait
* **Module d'Explication LLM** ([src/services/explanation.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/services/explanation.py)) : Les explications actuelles sont générées par des règles métier basées sur les seuils de probabilité (`recommendation.py`). Le service d'explication avancé via l'API HuggingFace / Mistral-7B est préparé sous forme de stub et doit être implémenté.

### 🔴 3. Ce qui Reste à Faire (Roadmap / Backlog)
* [ ] **Implémentation d'Explications via LLM** ([src/services/explanation.py](file:///c:/Users/Fedy%20Zarai/.gemini/antigravity-ide/scratch/salesteam_ai/src/services/explanation.py)) : Finaliser les appels HuggingFace Inference API + fallback robuste.
* [ ] **Suite de Tests Automatisés (`tests/`)** : Ajouter des unit tests avec `pytest` sur `cleaner.py`, `recommendation.py` et les endpoints API (`httpx`).
* [ ] **Intégration du Feedback dans le Réentraînement (`src/models/train_classifier.py`)** : Fusionner automatiquement `data/feedback/feedback_YYYY-MM.csv` dans le dataset lors de l'appel à `POST /api/retrain`.
* [ ] **Modèle Régresseur de Quantité (Prophet / Random Forest)** : Actuellement, la quantité suggérée dérive de la moyenne historique (`avg_qty`). Ajouter un second étage de régression pour affiner les quantités prédites.

---

## 🎯 4. Choix Techniques & Justifications

1. **XGBoost Classifier** :
   * *Pourquoi ?* Excellente gestion des données tabulaires mixtes (numériques, encodées), rapidité d'inférence (<10ms) et robustesse naturelle face aux valeurs manquantes et déséquilibres de classes.
2. **FastAPI** :
   * *Pourquoi ?* Performances asynchrones élevées, validation de type automatique via Pydantic et génération automatique de la documentation Swagger OpenAPI.
3. **GroupShuffleSplit sur `code_client`** :
   * *Pourquoi ?* Séparer les données de train/test aléatoirement sur les lignes entraînait une fuite de mémoire du profil client. Le split par groupe d'identifiant client force le modèle à être évalué sur des clients jamais vus en entraînement.
4. **React (Vite) + Vanilla CSS Glassmorphism** :
   * *Pourquoi ?* Temps de démarrage quasi instantané (Vite HMR) et contrôle visuel total sur le design sans dépendance lourde externe.

---

## ⚠️ 5. Compromis & Limitations Connues

1. **Quantité Suggérée Basique** :
   * *Limitation* : La quantité recommandée est actuellement égale à la moyenne historique arrondie (`ceil(avg_qty)`). Elle ne prend pas encore en compte un modèle prédictif continu pour la quantité.
2. **Cold-Start Client (Nouveaux clients sans historique)** :
   * *Limitation* : Pour un client n'ayant aucune commande passée, le modèle requiert des valeurs par défaut basées sur la géolocalisation ou les meilleures ventes de sa région.
3. **Implantation du Module d'Explication LLM** :
   * *Compromis* : L'API repose actuellement sur un générateur d'explications basé sur des règles déterministes, en attendant le branchement complet du stub `explanation.py`.
