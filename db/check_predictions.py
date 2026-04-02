# Permet de consulter les dernières prédictions enregistrées en base de données

# Ce code interroge la base PostgreSQL pour récupérer les 10 dernières prédictions enregistrées,
# puis les affiche de manière lisible (id, date, modèle, entrées et résultats).

from db.connection import get_connection

# Fonction qui récupère et affiche les dernières prédictions
def check_predictions():
    conn = None
    try:
        # Ouvre une connexion à la base PostgreSQL
        conn = get_connection()
        cur = conn.cursor()

        # Exécute une requête pour récupérer les 10 dernières prédictions
        cur.execute("""
            SELECT id, prediction_date, model_name, input_data, output_data
            FROM predictions
            ORDER BY id DESC
            LIMIT 10;
        """)

        # Récupère toutes les lignes retournées
        rows = cur.fetchall()

        # Parcourt et affiche chaque prédiction
        for row in rows:
            print("\n---")
            print(f"id: {row[0]}")      # Identifiant
            print(f"date: {row[1]}")    # Date de la prédiction
            print(f"model: {row[2]}")   # Nom du modèle
            print(f"input: {row[3]}")   # Données d'entrée
            print(f"output: {row[4]}")  # Résultat de la prédiction

        # Ferme le curseur
        cur.close()

    finally:
        # Ferme la connexion à la base si elle est ouverte
        if conn:
            conn.close()

# Point d'entrée du script
if __name__ == "__main__":
    check_predictions()