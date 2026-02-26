# tests/test_api_integration.py : test en conditions réelles (on teste l'intégration complète :
# - FastAPI démarre
# - le modèle est chargé depuis models/logreg_pipeline.joblib
# - la transformation en Dataframe se fait
# - le pipeline fait une vraie prédiction)
# permet de s'assurer que tout marche vraiment

from pathlib import Path
import pytest

import api


MODEL_PATH = Path(api.MODEL_PATH)


def _skip_unless_integration_enabled(request):
    if not request.config.getoption("--integration"):
        pytest.skip("Integration tests skipped (use --integration to run).")


@pytest.mark.integration
def test_model_file_exists(request):
    _skip_unless_integration_enabled(request)
    assert MODEL_PATH.exists(), f"Missing model file: {MODEL_PATH}"
# Vérifie que le fichier modèle existe bien

@pytest.mark.integration
def test_predict_integration_ok(request, client):
    _skip_unless_integration_enabled(request)

    payload = {
        "age": 35,
        "revenu_mensuel": 2500,
        "frequence_deplacement": "Occasionnel",
        "departement": "Sales",
        "poste": "Commercial",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200

    data = r.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert "probability" in data
    if data["probability"] is not None:
        assert 0.0 <= data["probability"] <= 1.0
# Envoie un payload réel à /predict (attend 200 + une précition 0/1 + probabilité ou none)

@pytest.mark.integration
def test_health_integration(request, client):
    _skip_unless_integration_enabled(request)
    r = client.get("/health")
    assert r.status_code == 200
# /health répond (pour vérifier l'API)