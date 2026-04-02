# Permet de tester la connexion à la base PostgreSQL
# À exécuter dans le terminal : python -m db.test_connection

# Ce code permet de tester la connexion à une base de données PostgreSQL en utilisant les paramètres configurés,
# en exécutant une requête simple pour vérifier que la connexion fonctionne, puis en affichant la version du serveur ou une erreur en cas de problème.

import psycopg2
from db.config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    DB_SSLROOTCERT,
)

# Fonction qui teste la connexion à la base de données
def test_connection():
    conn = None
    try:
        # Établit une connexion à PostgreSQL avec SSL sécurisé
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            sslmode="verify-full",          # Vérifie complètement le certificat SSL
            sslrootcert=DB_SSLROOTCERT,     # Chemin vers le certificat SSL
        )

        # Crée un curseur pour exécuter des requêtes SQL
        cur = conn.cursor()
        # Exécute une requête simple pour vérifier que la connexion fonctionne
        cur.execute("SELECT version();")
        # Récupère le résultat (version de PostgreSQL)
        version = cur.fetchone()[0]
        # Affiche un message de succès et la version
        print("Connexion réussie")
        print(version)
        # Ferme le curseur
        cur.close()
    
    except Exception as e:
        # Affiche une erreur en cas de problème de connexion
        print(f"Erreur de connexion : {e}")
        raise
    
    finally:
        # Ferme la connexion si elle a été ouverte
        if conn:
            conn.close()

# Point d'entrée du script
if __name__ == "__main__":
    test_connection()