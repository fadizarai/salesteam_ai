# 🚀 SalesTeam AI — Guide de Lancement du Projet

Ce guide contient l'ensemble des commandes pour faire tourner toutes les parties du projet (Nettoyage de données, Entraînement des modèles IA, API Backend, et Frontend Web).

---

## 📋 Prérequis : Activer l'environnement virtuel (venv)

Ouvrez un terminal PowerShell à la racine du projet (`C:\Users\Fedy Zarai\.gemini\antigravity-ide\scratch\salesteam_ai`) et lancez :

```powershell
# Activation de l'environnement virtuel Python
.\venv\Scripts\Activate.ps1
```

---

## 1. ⚙️ Étape 1 : Préparation des données & Features

Ces commandes permettent de nettoyer les fichiers bruts de la filiale LSAT, de construire la table principale et de générer les variables d'entrée (features) pour l'IA.

*Ces scripts doivent être exécutés depuis la racine du projet.*

```bash
# 1. Nettoyage des fichiers Excel raw et création des CSV propres
python -m src.data.cleaner

# 2. Construction de la table fusionnée et de la matrice de features
python -m src.features.feature_engineering
```

---

## 2. 🎯 Étape 2 : Génération de la Cible (Target) & Entraînement des Modèles IA

Ces commandes construisent les données d'apprentissage (avec Negative Sampling pour les exemples de non-achat) puis entraînent le classifieur binaire et le régresseur XGBoost.

```bash
# 1. Génération du dataset d'entraînement (training_set.csv)
python -m src.models.target_builder

# 2. Entraînement du classifieur XGBoost (Recommander : Oui / Non)
python -m src.models.train_classifier

# 3. Entraînement du régresseur XGBoost (Quantité à suggérer)
python -m src.models.train_regressor
```

---

## 3. 🔌 Étape 3 : Lancer l'API Backend (FastAPI)

Pour démarrer le serveur de l'API (qui expose les recommandations aux applications clientes sur le port 8000) :

```bash
# Commande exacte pour lancer l'API FastAPI avec rechargement automatique
python -m uvicorn src.api.main:app --port 8000 --reload
```

*Note : La documentation Swagger de l'API sera accessible sur http://127.0.0.1:8000/docs dès que le serveur sera lancé.*

---

## 4. 💻 Étape 4 : Lancer le Frontend (Application Web)

Ouvrez un **deuxième terminal**, placez-vous dans le dossier `frontend` et lancez le serveur de développement :

```bash
# 1. Aller dans le dossier frontend
cd frontend

# 2. Lancer le serveur de dev (sur http://localhost:5173 par défaut)
npm run dev
```
