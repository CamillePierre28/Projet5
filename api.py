from enum import Enum
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from contextlib import asynccontextmanager

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

####################################### VALIDATION DES DONNEES #######################################
# Enumération pour limiter les valeurs possibles (empêche l'utilisateur d'envoyer autre chose que ces trois valeurs)
class FrequenceDeplacement(str, Enum):
    Aucun = "Aucun"
    Occasionnel = "Occasionnel"
    Frequent = "Frequent"

# Modèle de requête (données envoyées à l’API)
# class PredictRequest(BaseModel):
#     # Interdit les champs non définis dans le modèle
#     model_config = ConfigDict(extra="forbid")

#     # Champ optionnel, entre 18 et 60
#     age: Optional[float] = Field(default=None, ge=18, le=60)
#     # Revenu >= 0
#     revenu_mensuel: Optional[float] = Field(default=None, ge=0)
#     # Enum défini plus haut
#     frequence_deplacement: Optional[FrequenceDeplacement] = None
#     # Champs texte optionnels
#     departement: Optional[str] = None
#     poste: Optional[str] = None


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
def predict(req: PredictRequest):
    try:
        # Convertit l’objet Pydantic en dictionnaire
        pred, proba = predict_payload(req.model_dump())
        # Retourne la réponse structurée
        return PredictResponse(prediction=pred, probability=proba)
    # Si erreur → retourne erreur HTTP 400
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Route pour voir les colonnes attendues par le modèle
@app.get("/model-info", tags=["debug"], summary="Infos sur le modèle (colonnes attendues)")
def model_info():
    return {"expected_columns": EXPECTED_COLUMNS}
