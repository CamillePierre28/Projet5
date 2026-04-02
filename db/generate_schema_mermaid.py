from pathlib import Path
from db.connection import get_connection

# Définition du fichier de sortie (Markdown contenant du Mermaid)
OUTPUT_FILE = Path("docs/schema_bdd_mermaid.md")

# Fonction qui récupère la liste des tables du schéma public
def fetch_tables(cursor):
    # Requête SQL pour lister les tables
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    # Retourne une liste des noms de tables
    return [row[0] for row in cursor.fetchall()]

# Fonction qui récupère les colonnes d'une table (nom + type uniquement)
def fetch_columns(cursor, table_name):
    # Requête SQL sur information_schema pour obtenir les colonnes
    cursor.execute("""
        SELECT
            column_name,
            data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
        ORDER BY ordinal_position;
    """, (table_name,))     # Paramètre sécurisé
    # Retourne toutes les colonnes
    return cursor.fetchall()

# Fonction qui récupère les clés JSONB d'une colonne (par défaut "data")
def fetch_jsonb_keys(cursor, table_name, jsonb_column="data"):
    # Vérifie que la colonne existe et est bien de type JSONB
    cursor.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
          AND column_name = %s
          AND data_type = 'jsonb';
    """, (table_name, jsonb_column))

    # Si la colonne n'existe pas : pas de clés JSON
    if cursor.fetchone() is None:
        return []

    # Requête pour extraire les clés JSON de premier niveau
    query = f"""
        SELECT DISTINCT jsonb_object_keys({jsonb_column})
        FROM {table_name}
        WHERE {jsonb_column} IS NOT NULL
        ORDER BY 1;
    """
    # Exécution
    cursor.execute(query)
    # Retourne la liste des clés JSON
    return [row[0] for row in cursor.fetchall()]

# Fonction utilitaire pour nettoyer les noms (compatibilité Mermaid)
def sanitize_name(name: str) -> str:
    return (
        name.replace(" ", "_")  # Remplace les espaces
            .replace("-", "_")  # Remplace les tirets
            .replace(".", "_")  # Remplace les points
            .replace("/", "_")  # Remplace les slashs
    )

# Fonction principale qui génère le diagramme Mermaid
def generate_mermaid():
    conn = None     # Initialisation de la connexion
    try:
        # Ouverture de la connexion à la base
        conn = get_connection()
        # Création du curseur SQL
        cur = conn.cursor()

        # Récupération des tables
        tables = fetch_tables(cur)

        # Liste des lignes du fichier Markdown
        lines = []
        # Titre du document
        lines.append("# Diagramme Mermaid de la base")
        lines.append("")
        # Début du bloc Mermaid
        lines.append("```mermaid")
        lines.append("erDiagram")   # Type de diagramme (Entity-Relationship)

        # Boucle sur chaque table
        for table in tables:
            # Nom de la table nettoyé pour Mermaid
            lines.append(f"    {sanitize_name(table)} {{")

            # Récupération des colonnes SQL
            columns = fetch_columns(cur, table)
            # Ajout de chaque colonne dans le diagramme
            for column_name, data_type in columns:
                # Nettoyage du type et du nom pour éviter les caractères invalides
                safe_type = sanitize_name(data_type)
                safe_col = sanitize_name(column_name)
                # Ajout ligne Mermaid : "type nom_colonne"
                lines.append(f"        {safe_type} {safe_col}")

            # Récupération des clés JSONB dans "data"
            jsonb_keys = fetch_jsonb_keys(cur, table, "data")
            # Ajout des clés JSON comme champs supplémentaires
            for key in jsonb_keys:
                # Préfixe "data_" pour éviter les collisions de noms
                safe_key = sanitize_name(f"data_{key}")
                # On les force en type "string" (choix arbitraire)
                lines.append(f"        string {safe_key}")

            # Fermeture du bloc table
            lines.append("    }")

        # Fin du bloc Mermaid
        lines.append("```")

        # Création du dossier si nécessaire
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Écriture du fichier Markdown
        OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")

        # Log console
        print(f"Diagramme Mermaid généré : {OUTPUT_FILE}")
        # Fermeture du curseur
        cur.close()

    finally:
        if conn:
            # Fermeture de la connexion
            conn.close()

# Point d'entrée du script
if __name__ == "__main__":
    # Lance la génération du diagramme si exécuté directement
    generate_mermaid()