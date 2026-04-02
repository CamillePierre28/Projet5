# Ce code met en place des tests unitaires pour une API FastAPI en utilisant un modèle simulé (mock) afin de vérifier le comportement des endpoints 
# (/health, /model-info, /predict) ainsi que la validation des données entrantes, sans dépendre d'un vrai modèle machine learning. 
# tests/test_api_unit.py : tests rapides et isolés qui ne dépendent pas du vrai modèle (on teste l'API et la validation sans se soucier du modèle réel)
# permet de détecter les bugs de code/validation et authentification par clé API

import pytest
from fastapi.testclient import TestClient

import api

# Clé API utilisée uniquement pour les tests
TEST_API_KEY = "cle-de-test-valide"

# Classe Dummy : simule un modèle ML avec des prédictions connues
class DummyModel:
    def predict(self, X):
        # Retourne toujours la prédiction 1
        return [1]

    def predict_proba(self, X):
        # Retourne une probabilité fixe (classe 0 = 0.25, classe 1 = 0.75)
        return [[0.25, 0.75]]

# Fixture pytest pour créer un client de test avec un modèle mocké
@pytest.fixture()
def unit_client(monkeypatch):
    # Remplace la fonction get_model par une version qui retourne DummyModel
    monkeypatch.setattr(api, "get_model", lambda: DummyModel())

    # Réinitialise le cache global du modèle (au cas où)
    monkeypatch.setattr(api, "_model", None)

    # Injecte une clé API de test dans l'environnement
    monkeypatch.setenv("API_KEY", TEST_API_KEY)

    # Remplace la sauvegarde en base par une fonction vide pour éviter tout accès réel à la base pendant les tests unitaires
    monkeypatch.setattr(api, "save_prediction", lambda **kwargs: None)

    # Initialise le client de test FastAPI
    with TestClient(api.app) as c:
        yield c

# Test unitaire : vérifie que l'endpoint /health fonctionne
def test_health_unit(unit_client):
    r = unit_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
# /health renvoie 200 + status : ok

# Test unitaire : vérifie que /model-info refuse l'accès sans clé API
def test_model_info_unit_without_api_key(unit_client):
    r = unit_client.get("/model-info")
    assert r.status_code == 401

# Test unitaire : vérifie que /model-info refuse une mauvaise clé API
def test_model_info_unit_with_invalid_api_key(unit_client):
    r = unit_client.get("/model-info", headers={"X-API-Key": "mauvaise-cle"})
    assert r.status_code == 403

# Test unitaire : vérifie que /model-info retourne les bonnes informations avec une clé valide
def test_model_info_unit(unit_client):
    r = unit_client.get("/model-info", headers={"X-API-Key": TEST_API_KEY})
    assert r.status_code == 200
    data = r.json()

    # Vérifie que la liste des colonnes attendues est présente
    assert "expected_columns" in data
    assert "age" in data["expected_columns"]
# /model-info renvoie bien la liste expected_columns

# Test unitaire : vérifie que /predict refuse l'accès sans clé API
def test_predict_unit_without_api_key(unit_client):
    payload = {
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

    # Envoie une requête POST à /predict
    r = unit_client.post("/predict", json=payload)

    # Vérifie la réponse
    assert r.status_code == 401

# Test unitaire : vérifie que /predict refuse une mauvaise clé API
def test_predict_unit_with_invalid_api_key(unit_client):
    payload = {
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

    # Envoie une requête POST à /predict
    r = unit_client.post("/predict", json=payload, headers={"X-API-Key": "mauvaise-cle"})

    # Vérifie la réponse
    assert r.status_code == 403

# Test unitaire : vérifie que /predict fonctionne avec un payload valide et une clé API valide
def test_predict_unit_ok(unit_client):
    payload = {
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

    # Envoie une requête POST à /predict
    r = unit_client.post("/predict", json=payload, headers={"X-API-Key": TEST_API_KEY})

    # Vérifie la réponse
    assert r.status_code == 200
    data = r.json()

    # Vérifie les valeurs retournées par le modèle mocké
    assert data["prediction"] == 1
    assert data["probability"] == 0.75
# /predict renvoie le bon JSON (prédiction + probabilité) avec le modèle mocké

# Test unitaire : vérifie qu'un champ inconnu est rejeté
def test_predict_unit_reject_unknown_field(unit_client):
    r = unit_client.post("/predict", json={"age": 30, "champ_inconnu": 123}, headers={"X-API-Key": TEST_API_KEY})
    # Code 422 attendu (validation Pydantic avec extra="forbid")
    assert r.status_code == 422  # Pydantic extra="forbid"
# si on envoie un champ non prévu (422)

# Test unitaire : vérifie qu'un âge invalide est rejeté
def test_predict_unit_invalid_age(unit_client):
    r = unit_client.post("/predict", json={"age": 10}, headers={"X-API-Key": TEST_API_KEY})
    # Code 422 attendu (âge trop petit)
    assert r.status_code == 422
# âge trop petit (422)

# Test unitaire : vérifie qu'une valeur d'énumération invalide est rejetée
def test_predict_unit_invalid_enum(unit_client):
    r = unit_client.post("/predict", json={"frequence_deplacement": "Souvent"}, headers={"X-API-Key": TEST_API_KEY})
    # Code 422 attendu (valeur non autorisée)
    assert r.status_code == 422