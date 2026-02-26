# Projet de Data Science : Analyse exploratoire et modélisation

## Objectif du projet

Ce projet a pour objectif de construire un **modèle de classification** à partir de plusieurs sources de données (évaluation, SIRH et sondage) afin d’analyser les facteurs explicatifs d’un phénomène métier (variable cible binaire : Quitte l'entreprise oui/non) et de prédire ce phénomène à l’aide de différents algorithmes de machine learning.

Le projet est structuré en deux grandes étapes :

1. **Analyse exploratoire et préparation des données**
2. **Modélisation et évaluation des performances**

---

## Structure du projet

```text
├── data/                            # Données 
├── analyse_exploratoire.ipynb      # Notebook de l'analyse exploratoire
├── modelisation.ipynb              # Notebook de la Modélisation
├── README.md                       # Documentation du projet
├── requirements.txt                # Liste des dépendances Python
├── uv.lock                         # Verrouillage des versions pour la reproductibilité
├── pyproject.toml                  # Déclaration des dépendances et configuration du projet
├── main.py                         # Point d’entrée du projet
├── .gitignore                      # Fichiers et dossiers ignorés par Git
```

---

## Analyse exploratoire des données

Le notebook `analyse_exploratoire.ipynb` couvre les étapes suivantes :

* Import des librairies et chargement des différents jeux de données
* Observation générale des données (dimensions, types de variables, valeurs manquantes)
* Description des datasets et compréhension des variables
* Préparation des données en vue de la jointure
* Création d’un **dataframe central** issu de la jointure entre :
  * les données d’évaluation (`eval_df`)
  * les données SIRH (`sirh_df`)
  * les données de sondage (`sondage_df`)
* Nettoyage du dataset (traitement des valeurs manquantes, incohérences, doublons)
* Analyse statistique et **visualisations**
* Détection des valeurs aberrantes (méthode IQR)
* Étude de la relation entre variables explicatives et variable cible

Cette étape permet de préparer un jeu de données propre et exploitable pour la modélisation.

---

## Modélisation

Le notebook `modelisation.ipynb` est dédié à la construction et à l’évaluation des modèles de machine learning.

* Séparation des variables explicatives (`X`) et de la variable cible (`y`)
* Encodage des variables qualitatives
* Mise en place d’un **pipeline de preprocessing** (scikit-learn)
* Optimisation des hyperparamètres (GridSearch / RandomSearch)
* Gestion du déséquilibre de classes
* Sélection de variables plus avancée
* Analyse de l’interprétabilité des modèles (feature importance, SHAP)

### Modèles testés

Plusieurs modèles de classification sont entraînés et comparés :

* Dummy Classifier (baseline)
* Régression Logistique
* Random Forest
* XGBoost
* Support Vector Classifier (SVC linéaire et non linéaire)

### Évaluation

Les performances des modèles sont évaluées à l’aide de métriques classiques de classification :

* Accuracy
* Precision
* Recall
* F1-score
* Classification report
* Matrice de confusion
* Courbes ROC

L’objectif est d’identifier le modèle offrant le meilleur compromis entre performance et interprétabilité.

---

## Librairies utilisées

* Python 
* pandas, numpy
* matplotlib, seaborn
* scikit-learn
* xgboost

---

## Lancer le projet

1. Cloner le dépôt
2. Installer les dépendances nécessaires
3. Exécuter les notebooks dans l’ordre suivant :

   * `analyse_exploratoire.ipynb`
   * `modelisation.ipynb`

## Auteur

Projet réalisé dans un cadre pédagogique de data science.


A ajouter : 
Les catégories inconnues sont ignorées (encodées à 0) par le pipeline.