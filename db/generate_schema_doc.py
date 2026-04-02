from pathlib import Path
from db.connection import get_connection

# Définition du chemin du fichier de sortie Markdown
OUTPUT_FILE = Path("docs/schema_bdd.md")

# Fonction qui récupère la liste des tables de la base (schéma public)
def fetch_tables(cursor):
    # Exécution d'une requête SQL pour récupérer les noms des tables
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    # Retourne une liste contenant uniquement les noms de tables
    return [row[0] for row in cursor.fetchall()]

# Fonction qui récupère les colonnes d'une table donnée
def fetch_columns(cursor, table_name):
    # Requête SQL pour récupérer les métadonnées des colonnes
    cursor.execute("""
        SELECT
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
        ORDER BY ordinal_position;
    """, (table_name,))     # Paramètre sécurisé pour éviter les injections SQL
    # Retourne toutes les lignes récupérées
    return cursor.fetchall()

# Fonction qui récupère les colonnes constituant la clé primaire
def fetch_primary_keys(cursor, table_name):
    # Requête SQL pour trouver les contraintes de type PRIMARY KEY
    cursor.execute("""
        SELECT kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        WHERE tc.table_schema = 'public'
          AND tc.table_name = %s
          AND tc.constraint_type = 'PRIMARY KEY'
        ORDER BY kcu.ordinal_position;
    """, (table_name,))
    # Retourne un set (ensemble) de noms de colonnes (plus rapide pour les tests "in")
    return {row[0] for row in cursor.fetchall()}

# Fonction qui récupère les clés JSONB d'une colonne (par défaut "data")
def fetch_jsonb_keys(cursor, table_name, jsonb_column="data"):
    """
    Récupère toutes les clés de premier niveau présentes dans une colonne JSONB.
    """
    # Vérifie que la colonne existe bien et qu'elle est de type JSONB
    cursor.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
          AND column_name = %s
          AND data_type = 'jsonb';
    """, (table_name, jsonb_column))

    # Si la colonne n'existe pas ou n'est pas JSONB → on retourne une liste vide
    if cursor.fetchone() is None:
        return []

    # Construction d'une requête dynamique pour extraire les clés JSONB
    query = f"""
        SELECT DISTINCT jsonb_object_keys({jsonb_column})
        FROM {table_name}
        WHERE {jsonb_column} IS NOT NULL
        ORDER BY 1;
    """
    # Exécution de la requête
    cursor.execute(query)
    # Retourne toutes les clés JSON trouvées
    return [row[0] for row in cursor.fetchall()]

# Fonction principale qui génère le fichier Markdown
def generate_markdown():
    conn = None     # Initialisation de la connexion
    try:
        # Ouverture de la connexion à la base de données
        conn = get_connection()
        # Création d'un curseur pour exécuter les requêtes SQL
        cur = conn.cursor()

        # Récupération de toutes les tables
        tables = fetch_tables(cur)

        # Liste qui va contenir toutes les lignes du fichier Markdown
        lines = []
        # Titre principal du document
        lines.append("# Schéma de la base de données")
        lines.append("")
        lines.append("Documentation générée automatiquement depuis PostgreSQL.")
        lines.append("")

        # Boucle sur chaque table
        for table in tables:
            # Ajout du nom de la table
            lines.append(f"## Table `{table}`")
            lines.append("")

            # Récupération des clés primaires
            pk_columns = fetch_primary_keys(cur, table)
            # Récupération des colonnes
            columns = fetch_columns(cur, table)

            # Section des colonnes SQL
            lines.append("### Colonnes SQL")
            lines.append("")
            # En-tête du tableau Markdown
            lines.append("| Colonne | Type | Nullable | Clé primaire | Valeur par défaut |")
            lines.append("|---|---|---|---|---|")

            # Boucle sur chaque colonne
            for column_name, data_type, is_nullable, column_default in columns:
                # Conversion du nullable en "Oui/Non"
                nullable = "Oui" if is_nullable == "YES" else "Non"
                # Indique si la colonne est une clé primaire
                pk = "Oui" if column_name in pk_columns else "Non"
                # Gestion de la valeur par défaut (None : vide)
                default = column_default if column_default is not None else ""
                # Ajout d'une ligne dans le tableau Markdown
                lines.append(
                    f"| `{column_name}` | `{data_type}` | {nullable} | {pk} | `{default}` |"
                )

            lines.append("")

            # Récupération des clés JSONB dans la colonne "data"
            jsonb_keys = fetch_jsonb_keys(cur, table, "data")
            # Si des clés JSONB existent
            if jsonb_keys:
                lines.append("### Champs détectés dans `data` (JSONB)")
                lines.append("")
                # Liste des clés JSON sous forme de liste Markdown
                for key in jsonb_keys:
                    lines.append(f"- `data.{key}`")
                lines.append("")

        # Création du dossier si nécessaire (docs/)
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Écriture du fichier Markdown
        OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")

        # Affichage console
        print(f"Documentation générée : {OUTPUT_FILE}")
        # Fermeture du curseur
        cur.close()

    finally:
        # Fermeture de la connexion si elle existe
        if conn:
            conn.close()

# Point d'entrée du script
if __name__ == "__main__":
    # Lance la génération du Markdown si le fichier est exécuté directement
    generate_markdown()