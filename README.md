---
title: Projet5 Staging
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Projet de Data Science & API de prédiction d’attrition

Ce projet combine :

- une **analyse exploratoire et modélisation machine learning**
- une **API FastAPI de prédiction**
- un **stockage des prédictions dans PostgreSQL (AWS RDS)**
- un **déploiement automatisé (CI/CD → Hugging Face)**

---

# Partie 1 — Analyse exploratoire & modélisation

## Objectif du projet

Construire un modèle de classification permettant de prédire si un employé va quitter l’entreprise (**attrition**), à partir de plusieurs sources de données :

- données d’évaluation
- données SIRH
- données de sondage

---

## Analyse exploratoire

Le notebook `analyse_exploratoire.ipynb` couvre :

- exploration des datasets
- traitement des valeurs manquantes
- jointure des sources de données
- nettoyage des données
- visualisations
- détection des outliers (IQR)
- analyse des relations avec la variable cible

---

## Modélisation

Le notebook `modelisation.ipynb` inclut :

- preprocessing (pipeline scikit-learn)
- encodage des variables
- gestion du déséquilibre de classes
- optimisation des hyperparamètres
- sélection de variables
- interprétabilité (SHAP)

### Modèles testés

- Dummy Classifier
- Régression Logistique (modèle retenu)
- Random Forest
- XGBoost
- SVC

---

## Évaluation

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- Matrice de confusion

---

# Partie 2 — API de prédiction (FastAPI)

## Objectif

Exposer le modèle ML via une API REST permettant :
- de faire des prédictions en temps réel
- de stocker les prédictions en base
- d’industrialiser le modèle

Les données stockées en base permettent une exploitation analytique :
- suivi du taux d’attrition prédit
- analyse des profils à risque
- évolution des prédictions dans le temps
- monitoring des performances du modèle

Ces données peuvent être utilisées pour :
- construire des dashboards (Power BI, Streamlit, etc.)
- détecter des dérives du modèle (data drift)
- aider à la prise de décision RH

## Installation

### Prérequis

Avant de lancer le projet, vérifier que les éléments suivants sont installés sur la machine :

- Python 3.12
- `uv` pour la gestion de l’environnement et des dépendances
- Docker (optionnel, pour exécuter l’application en conteneur)
- un accès à une base PostgreSQL si l’on souhaite activer le stockage des prédictions

### Cloner le dépôt

```bash
git clone <URL_DU_DEPOT>
cd Classifiez_automatiquement_des_informations
```
### Installer les dépendances

Avec uv : uv sync
Alternative avec pip : pip install -r requirements.txt

### Configurer les variables d'environnement

Créer un fichier .env à la racine du projet pour définir les paramètres de connexion à la base de données. 

Exemple : 
DB_HOST=localhost
DB_PORT=XXXX
DB_NAME=attrition_db
DB_USER=postgres
DB_PASSWORD=mot_de_passe

### Initialiser la base de données

Créer les tables nécessaires : python -m db.create_tables

Optionnel : importer des données existantes
- python -m db.import_csv

### Processus de traitement et stockage des données

Le pipeline de données du projet est structuré comme suit :

1. Collecte des données :
   - Import de datasets CSV (SIRH, évaluations, sondages)

2. Prétraitement :
   - Nettoyage des données
   - Gestion des valeurs manquantes
   - Feature engineering

3. Entraînement du modèle :
   - Pipeline scikit-learn
   - Sauvegarde du modèle (joblib)

4. Inférence via API :
   - Réception des données utilisateur
   - Validation avec Pydantic
   - Transformation en DataFrame
   - Prédiction via le modèle

5. Stockage :
   - Enregistrement des requêtes et résultats dans PostgreSQL
   - Format JSONB (input_data / output_data)

6. Exploitation :
   - Analyse des prédictions stockées
   - Suivi des performances du modèle

### Vérifier l'installation

Lancer l'API en local : uv run uvicorn api:app --reload

La documentation interactive sera ensuite disponible à l’adresse suivante : http://127.0.0.1:8000/docs

## Lancer l’API

uv run uvicorn api:app --reload

Swagger UI : http://127.0.0.1:8000/docs

Endpoint /predict

Exemple de requête : 
{
  "nombre_participation_pee": 1,
  "nb_formations_suivies": 3,
  "distance_domicile_travail": 8,
  "niveau_education": 1,
  "domaine_etude": "Infra & Cloud",
  "frequence_deplacement": "Frequent",
  "annees_depuis_la_derniere_promotion": 1,
  "age": 49,
  "genre": 0,
  "revenu_mensuel": 5130,
  "statut_marital": "Marié(e)",
  "departement": "Consulting",
  "poste": "Assistant de Direction",
  "nombre_experiences_precedentes": 1,
  "annees_dans_l_entreprise": 10,
  "niveau_hierarchique_poste": 2,
  "heure_supplementaires": 0,
  "augmentation_salaire_precedente": 23,
  "satisfaction_globale": 3.0,
  "note_evaluation": 3.0
}

Exemple de réponse : 
{
  "prediction": 0,
  "probability": 0.03833651019592016
}

## Fonctionnement interne : 

L'API : 
1. valide les données (Pydantic)
2. transforme en DataFrame
3. appelle le modèle ML
4. retourne la prédiction
5. stocke la requête et la réponse en base

## Base de données PostgreSQL (AWS)

Les prédictions sont stockées dans la table "predictions"

Structure de la table : 
- prediction_date
- model_name 
- model_version
- input_data (JSONB)
- output_data (JSONB)

Scripts utiles : 
- create_tables : créer les tables
- import_csv : importer les données
- check_predictions : consulter les prédictions

### Génération du schéma de base de données

La documentation du schéma de base de données est générée automatiquement à partir de PostgreSQL.

Commandes :

```bash
python -m db.generate_schema_doc
python -m db.generate_schema_mermaid
```

## CI/CD

Pipeline GitHub Actions : 
- Pull Request -> tests uniquement
- develop -> staging
- main -> production 

## Déploiement

### Déploiement via Docker

L’application est conteneurisée avec Docker afin de garantir un environnement reproductible.

#### Build de l’image

```bash
docker build -t attrition-api
```
#### Lancer le conteneur 

docker run -p 8000:8000 --env-file .env attrition-api, l'API sera accessible à l'adresse : http://localhost:8000/docs

### Déploiement sur Hugging Face Spaces

L’API est déployée automatiquement sur Hugging Face Spaces via Docker.

Le processus repose sur un pipeline CI/CD avec GitHub Actions :
- Pull Request → exécution des tests
- Branche develop → déploiement en environnement de staging
- Branche main → déploiement en production

À chaque mise à jour du code :
1. Les tests sont exécutés automatiquement
2. L’image Docker est reconstruite
3. L’application est redéployée sur Hugging Face

#### Variables d'environnement

Le déploiement nécessite la configuration de variables d’environnement, notamment :
- paramètres de connexion à la base PostgreSQL (AWS RDS)
- éventuels secrets applicatifs

Ces variables ne sont jamais stockées dans le code et doivent être définies :
- localement via un fichier .env
- dans les paramètres du service de déploiement (Hugging Face Spaces)

#### Vérification du déploiement

Une fois l’application déployée :
- accéder à l’interface Swagger (/docs)
- tester l’endpoint /predict
- vérifier l’insertion des prédictions en base de données

#### Monitoring

Les prédictions stockées en base permettent :
- le suivi du comportement du modèle en production
- l’analyse des performances dans le temps
- la détection de dérives potentielles (data drift / concept drift)

## Authentification & gestion des accès

### Authentification de l’API

L’accès à l’API est protégé par un mécanisme d’authentification par **API Key**.

Le client doit transmettre une clé valide dans le header HTTP suivant :

```http
X-API-Key: ma_cle_secrete
```

Côté serveur, les clés autorisées sont chargées depuis les variables d’environnement avec deux formats possibles :
- API_KEY=cle_unique
- API_KEYS=cle1,cle2,cle3

Cette approche permet :
- de sécuriser l’accès aux endpoints de l’API
- de gérer une ou plusieurs clés selon l’environnement
- de ne jamais exposer les clés directement dans le code source

En cas d’appel non autorisé :
- si aucune clé n’est fournie, l’API retourne une erreur 401 Unauthorized
- si la clé fournie est invalide, l’API retourne une erreur 403 Forbidden
- si aucune clé n’est configurée côté serveur, l’API retourne une erreur 500 Internal Server Error

La comparaison des clés est effectuée avec secrets.compare_digest, afin de limiter les risques liés aux attaques par timing.

### Gestion des accès à la base de données

L’accès à la base PostgreSQL (hébergée sur AWS RDS) est sécurisé via :

- des **identifiants (user / password)** stockés dans des variables d’environnement
- une connexion distante restreinte (configuration AWS)
- l’absence de credentials en clair dans le code source

Les informations sensibles sont injectées via un fichier `.env` en local ou via les variables du service de déploiement.

### Gestion des données

Les données manipulées par l’API sont :
- des données d’entrée utilisateur (features)
- des prédictions générées par le modèle

Ces données sont stockées dans la base PostgreSQL avec :
- un format **JSONB** pour plus de flexibilité
- une séparation claire entre :
  - `input_data` : données envoyées par le client
  - `output_data` : prédiction et probabilité retournées par le modèle

Aucune donnée personnelle sensible (PII) n’est utilisée dans ce projet.

### Bonnes pratiques mises en place

Le projet applique plusieurs bonnes pratiques de sécurité :
- authentification de l’API par clé d’accès
- chargement des secrets via variables d’environnement
- absence de clés API ou mots de passe dans le code source
- validation des entrées avec Pydantic
- comparaison sécurisée des clés avec compare_digest
- séparation des environnements de développement, staging et production

## Structure du projet

```
Classifiez_automatiquement_des_informations
├── .github
│   └── workflows
│       ├── ci-cd.yml
├── data
│   ├── df1.csv
│   ├── extrait_eval.csv
│   ├── extrait_sirh.csv
│   └── extrait_sondage.csv
├── db
│   ├── __init__.py
│   ├── check_predictions.py
│   ├── config.py
│   ├── connection.py
│   ├── create_tables.py
│   ├── generate_schema_doc.py
│   ├── generate_schema_mermaid.py
│   ├── import_csv.py
│   ├── save_predictions.py
│   └── test_connection.py
├── docs
│   ├── schema_bdd_mermaid.md
│   └── schema_bdd.md
├── models
│   └── logreg_pipeline.joblib
├── tests
│   ├── conftest.py
│   ├── test_api_integration.py
│   └── test_api_unit.py
├── .gitignore
├── .python-version
├── __init__.py
├── analyse_exploratoire.ipynb
├── api.py
├── Dockerfile
├── make_payload.py
├── modelisation.ipynb
├── pyproject.toml
├── README.md
├── requirements.txt
├── STAGING.md
└── uv.lock
```

## Couverture de tests

Le projet inclut des tests unitaires et des tests d’intégration couvrant le cœur applicatif (API, validation, logique métier).

La couverture globale est partiellement impactée par des scripts utilitaires (génération de schéma, import de données) qui ne sont pas destinés à être testés.

La couverture du code critique (API et logique métier) dépasse 90%.