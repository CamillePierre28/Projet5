# Connexion à PostgreSQL hébergé sur AWS
# Ce code charge les variables d'environnement nécessaires pour se connecter à une base de données PostegreSQL sur AWS,
# puis construit les paramètres de connexion ainsi que le chemin absolu vers le certificat SSL requis pour sécuriser la connexion
import os
from pathlib import Path
from dotenv import load_dotenv

# Charge les variables d'environnement depuis un fichier .env
load_dotenv()

# Définit le répertoire de base du projet (deux niveaux au-dessus de ce fichier)
BASE_DIR = Path(__file__).resolve().parent.parent

# Récupère les paramètres de connexion à la base de données depuis les variables d'environnement
DB_HOST = os.getenv("DB_HOST")                  # Adresse du serveur PostgreSQL
DB_PORT = int(os.getenv("DB_PORT", "5432"))     # Port (5432 par défaut)
DB_NAME = os.getenv("DB_NAME")                  # Nom de la base de données
DB_USER = os.getenv("DB_USER")                  # Utilisateur
DB_PASSWORD = os.getenv("DB_PASSWORD")          # Mot de passe

# Récupère le chemin relatif du certificat SSL
ssl_cert_relative = os.getenv("DB_SSLROOTCERT", "certs/global-bundle.pem")
# Construit le chemin absolu vers le certificat SSL
DB_SSLROOTCERT = str((BASE_DIR / ssl_cert_relative).resolve())