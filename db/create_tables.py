# Permet de créer les tables dans la base PostgreSQL
# À exécuter dans le terminal : python -m db.create_tables

# Ce code crée plusieurs tables dans une base PostgreSQL (si elles n’existent pas déjà) pour stocker des données brutes
# et des prédictions au format JSON, en gérant correctement la connexion, les transactions et les erreurs.

from db.connection import get_connection

# Script SQL pour créer les différentes tables si elles n'existent pas déjà
CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS extrait_eval_raw (
    id BIGSERIAL PRIMARY KEY,                   -- Identifiant unique
    data JSONB NOT NULL,                        -- Données brutes au format JSON
    imported_at TIMESTAMP DEFAULT NOW()         -- Date d'import (par défaut : maintenant)
);

CREATE TABLE IF NOT EXISTS extrait_sirh_raw (
    id BIGSERIAL PRIMARY KEY,
    data JSONB NOT NULL,
    imported_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS extrait_sondage_raw (
    id BIGSERIAL PRIMARY KEY,
    data JSONB NOT NULL,
    imported_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    prediction_date TIMESTAMP NOT NULL DEFAULT NOW(),   -- Date de la prédiction
    model_name TEXT NOT NULL,                           -- Nom du modèle utilisé
    model_version TEXT,                                 -- Version du modèle
    input_data JSONB NOT NULL,                          -- Données d'entrée
    output_data JSONB NOT NULL                          -- Résultat de la prédiction
);
"""
# Fonction qui exécute le script de création des tables
def create_tables():
    conn = None
    try:
        # Ouvre une connexion à la base de données
        conn = get_connection()
        cur = conn.cursor()
        # Exécute le script SQL
        cur.execute(CREATE_TABLES_SQL)
        # Valide les changements (commit)
        conn.commit()
        # Ferme le curseur
        cur.close()
        print("Tables créées avec succès.")
    except Exception as e:
        # Annule les changements en cas d'erreur
        if conn:
            conn.rollback()
        print(f"Erreur lors de la création des tables : {e}")
        raise
    finally:
        # Ferme la connexion à la base
        if conn:
            conn.close()

# Point d'entrée du script
if __name__ == "__main__":
    create_tables()