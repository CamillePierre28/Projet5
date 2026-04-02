# Schéma de la base de données

Documentation générée automatiquement depuis PostgreSQL.

## Table `extrait_eval_raw`

### Colonnes SQL

| Colonne | Type | Nullable | Clé primaire | Valeur par défaut |
|---|---|---|---|---|
| `id` | `bigint` | Non | Oui | `nextval('extrait_eval_raw_id_seq'::regclass)` |
| `data` | `jsonb` | Non | Non | `` |
| `imported_at` | `timestamp without time zone` | Oui | Non | `now()` |

### Champs détectés dans `data` (JSONB)

- `data.augementation_salaire_precedente`
- `data.eval_number`
- `data.heure_supplementaires`
- `data.niveau_hierarchique_poste`
- `data.note_evaluation_actuelle`
- `data.note_evaluation_precedente`
- `data.satisfaction_employee_environnement`
- `data.satisfaction_employee_equilibre_pro_perso`
- `data.satisfaction_employee_equipe`
- `data.satisfaction_employee_nature_travail`

## Table `extrait_sirh_raw`

### Colonnes SQL

| Colonne | Type | Nullable | Clé primaire | Valeur par défaut |
|---|---|---|---|---|
| `id` | `bigint` | Non | Oui | `nextval('extrait_sirh_raw_id_seq'::regclass)` |
| `data` | `jsonb` | Non | Non | `` |
| `imported_at` | `timestamp without time zone` | Oui | Non | `now()` |

### Champs détectés dans `data` (JSONB)

- `data.age`
- `data.annee_experience_totale`
- `data.annees_dans_l_entreprise`
- `data.annees_dans_le_poste_actuel`
- `data.departement`
- `data.genre`
- `data.id_employee`
- `data.nombre_experiences_precedentes`
- `data.nombre_heures_travailless`
- `data.poste`
- `data.revenu_mensuel`
- `data.statut_marital`

## Table `extrait_sondage_raw`

### Colonnes SQL

| Colonne | Type | Nullable | Clé primaire | Valeur par défaut |
|---|---|---|---|---|
| `id` | `bigint` | Non | Oui | `nextval('extrait_sondage_raw_id_seq'::regclass)` |
| `data` | `jsonb` | Non | Non | `` |
| `imported_at` | `timestamp without time zone` | Oui | Non | `now()` |

### Champs détectés dans `data` (JSONB)

- `data.annees_depuis_la_derniere_promotion`
- `data.annes_sous_responsable_actuel`
- `data.a_quitte_l_entreprise`
- `data.ayant_enfants`
- `data.code_sondage`
- `data.distance_domicile_travail`
- `data.domaine_etude`
- `data.frequence_deplacement`
- `data.nb_formations_suivies`
- `data.niveau_education`
- `data.nombre_employee_sous_responsabilite`
- `data.nombre_participation_pee`

## Table `predictions`

### Colonnes SQL

| Colonne | Type | Nullable | Clé primaire | Valeur par défaut |
|---|---|---|---|---|
| `id` | `bigint` | Non | Oui | `nextval('predictions_id_seq'::regclass)` |
| `prediction_date` | `timestamp without time zone` | Non | Non | `now()` |
| `model_name` | `text` | Non | Non | `` |
| `model_version` | `text` | Oui | Non | `` |
| `input_data` | `jsonb` | Non | Non | `` |
| `output_data` | `jsonb` | Non | Non | `` |
