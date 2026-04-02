# Permet d'enregistrer les prédictions du modèle dans la base PostgreSQL
# À exécuter dans le terminal : python -m db.save_predictions

# Ce code permet d’enregistrer dans une base PostgreSQL les résultats d’une prédiction de modèle (entrées, sorties, nom et version du modèle),
# en stockant les données au format JSON et en assurant une gestion sécurisée des transactions et des erreurs.

from datetime import datetime
from psycopg2.extras import Json

from db.connection import get_connection

# Fonction qui sauvegarde une prédiction en base de données
def save_prediction(input_data: dict, output_data: dict, model_name: str, model_version: str = None):
    conn = None
    try:
        # Ouvre une connexion à la base
        conn = get_connection()
        cur = conn.cursor()

        # Exécute une requête d'insertion dans la table predictions
        cur.execute(
            """
            INSERT INTO predictions (
                prediction_date,
                model_name,
                model_version,
                input_data,
                output_data
            )
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                datetime.utcnow(),  # Date/heure actuelle (UTC)
                model_name,         # Nom du modèle utilisé
                model_version,      # Version du modèle (optionnelle)
                Json(input_data),   # Données d'entrée au format JSON
                Json(output_data),  # Résultat de la prédiction au format JSON
            ),
        )

        # Valide l'insertion
        conn.commit()
        
        # Ferme le curseur
        cur.close()
        
        print("Prédiction sauvegardée.")

    except Exception as e:
        # Annule la transaction en cas d'erreur
        if conn:
            conn.rollback()
        print(f"Erreur sauvegarde prédiction : {e}")
        raise
    
    finally:
        # Ferme la connexion à la base
        if conn:
            conn.close()

# Exemple d'utilisation du script
if __name__ == "__main__":
    save_prediction(
        input_data={"texte": "demande de formation"},
        output_data={"classe_predite": "sirh", "score": 0.95},
        model_name="classifieur_documents",
        model_version="v1",
    )