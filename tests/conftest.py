# tests/conftest.py : prépare l'environnement de test (créer un lient réutilisable dans les tests)
import sys
from pathlib import Path

# Ajoute le dossier racine du projet au PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture()
def client():
    with TestClient(api.app) as c:
        yield c


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests (require the real model file).",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark a test as integration test")