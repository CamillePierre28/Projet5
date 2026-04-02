# Ce fichier configure l'environnement de test avec pytest en ajoutant le projet au PYTHONPATH, 
# en fournissant un client FastAPI réutilisable pour tester l'API, et en définissant une option ainsi qu'un marqueur pour distinguer les tests d'intégration. 

import sys
from pathlib import Path

# Récupère le chemin du dossier racine du projet 
ROOT_DIR = Path(__file__).resolve().parents[1]
# Ajoute ce dossier au PYTHONPATH pour permettre les imports du projet
sys.path.insert(0, str(ROOT_DIR))

import pytest
from fastapi.testclient import TestClient

# Importe l'application FastAPI définie dans api.py
import api

# Ficture pytest qui fournit un client de test pour l'API
@pytest.fixture()
def client():
    # Initialise un client de test FastAPI
    with TestClient(api.app) as c:
        # Rend ce client disponible dans les tests
        yield c

# Ajoute une option personnalisée à pytest (--integration)
def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",    # option booléenne (présente ou non)
        default=False,          # par défaut désactivée
        help="Run integration tests (require the real model file).",
    )

# Déclare un marqueur personalisé "integration" pour les tests
def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark a test as integration test")