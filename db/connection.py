# Fournit une fonction de connexion réutilisable à la base PostgreSQL

# Ce code définit une fonction réutilisable qui crée et retourne une connexion sécurisée à une base de données PostgreSQL,
# afin de pouvoir l’utiliser facilement dans différents scripts du projet.

import psycopg2
from db.config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    DB_SSLROOTCERT,
)

# Fonction qui retourne une connexion à la base de données
def get_connection():
    return psycopg2.connect(
        host=DB_HOST,                   # Adresse du serveur PostgreSQL
        port=DB_PORT,                   # Port de connexion
        database=DB_NAME,               # Nom de la base de données
        user=DB_USER,                   # Utilisateur
        password=DB_PASSWORD,           # Mot de passe
        sslmode="verify-full",          # Vérification complète du certificat SSL
        sslrootcert=DB_SSLROOTCERT,     # Chemin vers le certificat SSL
    )