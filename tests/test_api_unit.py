# tests/test_api_unit.py : tests rapides et isolés qui ne dépendent pas du vrai modèle (on teste l'API et la validation sans se soucier du modèle réel)
# permet de détecter les bugs de code/validation

import pytest
from fastapi.testclient import TestClient

import api

# On remplace par un "faux" modèle (Dummy) qui renvoie toujours une prédicition connue
class DummyModel:
    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        return [[0.25, 0.75]]


@pytest.fixture()
def unit_client(monkeypatch):
    # On remplace get_model pour éviter de dépendre du fichier joblib
    monkeypatch.setattr(api, "get_model", lambda: DummyModel())

    # (optionnel) réinitialise le cache global si jamais
    monkeypatch.setattr(api, "_model", None)

    with TestClient(api.app) as c:
        yield c


def test_health_unit(unit_client):
    r = unit_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
# /health renvoie 200 + status : ok


def test_model_info_unit(unit_client):
    r = unit_client.get("/model-info")
    assert r.status_code == 200
    data = r.json()
    assert "expected_columns" in data
    assert "age" in data["expected_columns"]
# /model-info renvoie bien la liste expected_columns

def test_predict_unit_ok(unit_client):
    payload = {
        "age": 40,
        "revenu_mensuel": 3000,
        "frequence_deplacement": "Occasionnel",
        "departement": "Sales",
        "poste": "Commercial",
    }
    r = unit_client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] == 1
    assert data["probability"] == 0.75
# /predict renvoie le bon JSON (prédiction + probabilité) avec le modèle mocké

def test_predict_unit_reject_unknown_field(unit_client):
    r = unit_client.post("/predict", json={"age": 30, "champ_inconnu": 123})
    assert r.status_code == 422  # Pydantic extra="forbid"
# si on envoie un champ non prévu (422)

def test_predict_unit_invalid_age(unit_client):
    r = unit_client.post("/predict", json={"age": 10})
    assert r.status_code == 422
# âge trop petit (422)

def test_predict_unit_invalid_enum(unit_client):
    r = unit_client.post("/predict", json={"frequence_deplacement": "Souvent"})
    assert r.status_code == 422
# valeur inconnue (422)