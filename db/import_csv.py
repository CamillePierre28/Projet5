# Permet d'importer les fichiers CSV dans la base PostgreSQL
# À exécuter dans le terminal : python -m db.import_csv

# Ce code lit plusieurs fichiers CSV avec pandas et insère leurs données dans des tables PostgreSQL correspondantes
# en les stockant au format JSON, tout en gérant les erreurs et les transactions pour garantir un import fiable.

import json
from pathlib import Path

import pandas as pd
from psycopg2.extras import Json

from db.connection import get_connection

# Dossier contenant les fichiers CSV
DATA_DIR = Path("data")

# Association entre les fichiers CSV et les tables cibles
FILES_AND_TABLES = {
    "extrait_eval.csv": "extrait_eval_raw",
    "extrait_sirh.csv": "extrait_sirh_raw",
    "extrait_sondage.csv": "extrait_sondage_raw",
}

# Fonction qui importe un CSV dans une table donnée
def import_csv_to_table(csv_path: Path, table_name: str):
    conn = None
    try:
        # Lit le fichier CSV avec pandas
        df = pd.read_csv(csv_path)

        # Ouvre une connexion à la base de données
        conn = get_connection()
        cur = conn.cursor()

        # Requête d'insertion (les données sont stockées en JSONB)
        query = f"INSERT INTO {table_name} (data) VALUES (%s)"

        # Parcourt chaque ligne du DataFrame et l'insère en base
        for row in df.to_dict(orient="records"):
            cur.execute(query, [Json(row)])

        # Valide les insertions
        conn.commit()

        # Ferme le curseur
        cur.close()
        
        print(f"{csv_path.name} importé dans {table_name} ({len(df)} lignes)")
    
    except Exception as e:
        # Annule les insertions en cas d'erreur
        if conn:
            conn.rollback()
        print(f"Erreur import {csv_path.name}: {e}")
        raise
    
    finally:
        # Ferme la connexion à la base
        if conn:
            conn.close()

# Fonction principale : parcourt tous les fichiers à importer
def main():
    for filename, table_name in FILES_AND_TABLES.items():
        csv_path = DATA_DIR / filename

        # Vérifie si le fichier existe avant import
        if csv_path.exists():
            import_csv_to_table(csv_path, table_name)
        else:
            print(f"Fichier introuvable : {csv_path}")

# Point d'entrée du script
if __name__ == "__main__":
    main()