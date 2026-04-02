# Ce code définit des tests d'intégration pour une API Fast API en vérifiant la présence du fichier modèle, 
# le bon fonctionnement de l'endpoint /predict avec des données réalistes, et la disponibilité de l'endpoint /health, 
# tout en permettant d'activer ces tests uniquement via une option spécifique "--integration".

# tests/test_api_integration.py : test en conditions réelles (on teste l'intégration complète) :
# - FastAPI démarre
# - le modèle est chargé depuis models/logreg_pipeline.joblib
# - la transformation en Dataframe se fait
# - le pipeline fait une vraie prédiction)
# - la protection par clé API sur les endpoints sécurisés
# permet de s'assurer que tout marche vraiment

from pathlib import Path
import pytest

import api

# Clé API utilisée pour les tests d'intégration
TEST_API_KEY = "cle-integration-valide"

# Récupère le chemin du fichier modèle défini dans api.py
MODEL_PATH = Path(api.MODEL_PATH)

# Fonction utilitaire : ignore les tests si l'option --integration n'est pas activée
def _skip_unless_integration_enabled(request):
    if not request.config.getoption("--integration"):
        pytest.skip("Integration tests skipped (use --integration to run).")

# Test d'intégration : vérifie que le fichier modèle existe
@pytest.mark.integration
def test_model_file_exists(request):
    _skip_unless_integration_enabled(request)
    assert MODEL_PATH.exists(), f"Missing model file: {MODEL_PATH}"

# Test d'intégration : vérifie que l'endpoint /predict refuse l'accès sans clé API
@pytest.mark.integration
def test_predict_integration_requires_api_key(request, client):
    _skip_unless_integration_enabled(request)

    # Exemple de payload réaliste envoyé à l'API
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

    # Envoie une requête POST à l'endpoint /predict
    r = client.post("/predict", json=payload)
    assert r.status_code == 401

# Test d'intégration : vérifie que l'endpoint /predict fonctionne correctement avec une clé valide
@pytest.mark.integration
def test_predict_integration_ok(request, client, monkeypatch):
    _skip_unless_integration_enabled(request)

    # Injecte une clé API de test dans l'environnement
    monkeypatch.setenv("API_KEY", TEST_API_KEY)

    # Exemple de payload réaliste envoyé à l'API
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

    # Envoie une requête POST à l'endpoint /predict
    r = client.post("/predict", json=payload, headers={"X-API-Key": TEST_API_KEY})
    
    # Vérifie que la requête a réussi
    assert r.status_code == 200

    # Récupère la réponse JSON
    data = r.json()

    # Vérifie la présence et la validité des champs retournés
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert "probability" in data
    if data["probability"] is not None:
        assert 0.0 <= data["probability"] <= 1.0 
# Envoie un payload réel à /predict (attend 200 + une précition 0/1 + probabilité ou none)

# Test d'integration : vérifie que l'endpoint /health répond correctement
@pytest.mark.integration
def test_health_integration(request, client):
    _skip_unless_integration_enabled(request)
    # Envoie une requête GET à /health
    r = client.get("/health")
    # Vérifie que l'API répond correctement
    assert r.status_code == 200
# /health répond (pour vérifier l'API)