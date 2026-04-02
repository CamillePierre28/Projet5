from enum import Enum
from pathlib import Path
from typing import Optional

import os
import secrets
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, ConfigDict
from contextlib import asynccontextmanager
from db.save_predictions import save_prediction

# Ce code : 
# Charge un modèle de régression logistique sauvegardé
# Valide les données reçues via Pydantic
# Transforme les données en DataFrame
# Effectue une prédiction
# Expose le tout via une API FastAPI

####################################### CHARGEMENT DU MODELE #######################################

# Définit le chemin vers le fichier du modèle (models/logreg_pipeline.joblib)
# __file__ = fichier actuel
# resolve() = chemin absolu
# parent = dossier du fichier
MODEL_PATH = Path(__file__).resolve().parent / "models" / "logreg_pipeline.joblib"

# Variable globale pour stocker le modèle en mémoire
_model = None

# Liste des colonnes que le modèle attend exactement
EXPECTED_COLUMNS = [
    "age",
    "revenu_mensuel",
    "frequence_deplacement",
    "departement",
    "poste",
    "satisfaction_globale",
    "distance_domicile_travail",
    "annees_dans_l_entreprise",
    "nb_formations_suivies",
    "nombre_experiences_precedentes",
    "domaine_etude",
    "genre",
    "note_evaluation",
    "annees_depuis_la_derniere_promotion",
    "niveau_hierarchique_poste",
    "augmentation_salaire_precedente",
    "heure_supplementaires",
    "nombre_participation_pee",
    "statut_marital",
    "niveau_education",
]

# Fonction qui charge le modèle une seule fois (lazy loading)
def get_model():
    global _model       # on utilise la variable globale
    # Si le modèle n’est pas encore chargé
    if _model is None:
        # Vérifie que le fichier existe
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")
        # Charge le modèle en mémoire
        _model = joblib.load(MODEL_PATH)
    # Retourne le modèle (déjà chargé ou nouvellement chargé)
    return _model

# Fonction qui prépare les données et effectue la prédiction
def predict_payload(payload: dict):
    # Récupère le modèle
    model = get_model()

    # Crée un dictionnaire contenant toutes les colonnes attendues
    # Si une colonne est absente du payload → valeur None
    row = {col: payload.get(col, None) for col in EXPECTED_COLUMNS}

    # Crée un DataFrame pandas avec UNE seule ligne
    X = pd.DataFrame([row], columns=EXPECTED_COLUMNS)

    # Prédiction (0 ou 1 par exemple)
    pred = int(model.predict(X)[0])
    # Probabilité de la classe 1 si le modèle le permet
    proba = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else None
    # Retourne prédiction + probabilité
    return pred, proba

####################################### AUTHENTIFICATION #######################################

# Définit le header HTTP attendu côté client pour transmettre la clé API
# Exemple côté client : X-API-Key: ma_cle_secrete
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def load_api_keys() -> set[str]:
    """
    Charge les clés API depuis les variables d'environnement.

    Supporte deux formats :
    - API_KEYS=key1,key2,key3
    - API_KEY=key_unique
    """
    keys = set()

    # Récupère plusieurs clés éventuelles séparées par des virgules
    raw_keys = os.getenv("API_KEYS")

    # Récupère une clé unique éventuelle
    raw_single = os.getenv("API_KEY")

    # Si plusieurs clés sont fournies, on les découpe et on enlève les espaces inutiles
    if raw_keys:
        keys.update(k.strip() for k in raw_keys.split(",") if k.strip())

    # Si une seule clé est fournie, on l'ajoute aussi
    if raw_single:
        keys.add(raw_single.strip())

    return keys


# Dépendance FastAPI qui vérifie si la clé API envoyée par le client est valide
def require_api_key(api_key: str = Security(api_key_header)):
    # Charge les clés autorisées côté serveur
    valid_keys = load_api_keys()

    # Si aucune clé n'est configurée, c'est un problème serveur
    if not valid_keys:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Aucune clé API configurée côté serveur."
        )

    # Si le client n'a pas fourni de clé API
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API manquante. Ajoute le header X-API-Key."
        )

    # Vérifie que la clé envoyée correspond à une clé valide
    # compare_digest est plus sûr qu'une comparaison classique
    if not any(secrets.compare_digest(api_key, valid) for valid in valid_keys):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clé API invalide."
        )

    # Retourne la clé si elle est valide
    return api_key

####################################### VALIDATION DES DONNEES #######################################
# Enumération pour limiter les valeurs possibles (empêche l'utilisateur d'envoyer autre chose que ces trois valeurs)
class FrequenceDeplacement(str, Enum):
    Aucun = "Aucun"
    Occasionnel = "Occasionnel"
    Frequent = "Frequent"

# Modèle de requête (données envoyées à l’API)
class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: Optional[float] = Field(default=None, ge=18, le=60)
    revenu_mensuel: Optional[float] = Field(default=None, ge=0)
    frequence_deplacement: Optional[FrequenceDeplacement] = None
    departement: Optional[str] = None
    poste: Optional[str] = None

    satisfaction_globale: Optional[float] = None
    distance_domicile_travail: Optional[float] = Field(default=None, ge=0)
    annees_dans_l_entreprise: Optional[float] = Field(default=None, ge=0)
    nb_formations_suivies: Optional[float] = Field(default=None, ge=0)
    nombre_experiences_precedentes: Optional[float] = Field(default=None, ge=0)
    domaine_etude: Optional[str] = None
    genre: Optional[float] = None
    note_evaluation: Optional[float] = None
    annees_depuis_la_derniere_promotion: Optional[float] = Field(default=None, ge=0)
    niveau_hierarchique_poste: Optional[float] = Field(default=None, ge=0)
    augmentation_salaire_precedente: Optional[float] = None
    heure_supplementaires: Optional[float] = None
    nombre_participation_pee: Optional[float] = Field(default=None, ge=0)
    statut_marital: Optional[str] = None
    niveau_education: Optional[float] = None

# Modèle de réponse renvoyé par l’API
class PredictResponse(BaseModel):
    # Prédiction (0 ou 1)
    prediction: int
    # Probabilité entre 0 et 1
    probability: Optional[float] = Field(default=None, ge=0, le=1)

####################################### API #######################################
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()  # charge le modèle au démarrage
    yield        # l'app tourne

# Création de l’application FastAPI
app = FastAPI(
    title="API Attrition - Régression Logistique",
    version="1.0.0",
    lifespan=lifespan,
)

# Route GET simple pour vérifier que l’API fonctionne
@app.get("/health")
def health():
    return {"status": "ok"}

# Route POST pour faire une prédiction
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, _: str = Depends(require_api_key)):
    try:
        input_data = req.model_dump()

        pred, proba = predict_payload(input_data)

        save_prediction(
            input_data=input_data,
            output_data={
                "prediction": pred,
                "probability": proba,
            },
            model_name="logreg_pipeline",
            model_version="1.0.0",
        )

        return PredictResponse(prediction=pred, probability=proba)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Route pour voir les colonnes attendues par le modèle
@app.get("/model-info", tags=["debug"], summary="Infos sur le modèle (colonnes attendues)")
def model_info(_: str = Depends(require_api_key)):
    return {"expected_columns": EXPECTED_COLUMNS}

# Pour afficher quelque chose de propre dans la page principale sur Hugging Face
@app.get("/")
def root():
    return {
        "message": "API Attrition en ligne",
        "endpoints": ["/health", "/docs", "/predict", "/model-info"],
        "auth": "Ajouter le header X-API-Key pour accéder aux endpoints protégés"
    }
