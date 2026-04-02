import pandas as pd
import json

# Ce code charge le fichier csv df1 avec pandas, sélectionne une ligne spécifique du dataset (donc un individu), 
# la convertit en dictionnaire compatible JSON en remplaçant les valeurs manquantes (NaN) par None, 
# puis affiche le résultat formaté en JSON afin qu'il puisse être utilisable dans l'API.  

DATA_PATH = "data/df1.csv"

# Charge le dataset
df = pd.read_csv(DATA_PATH)

# Choisis l'individu à tester (ex: ligne 0, ou un index précis)
row = df.iloc[1]

# Convertit la ligne en dict compatible JSON
payload = row.to_dict()

# IMPORTANT: remplace les NaN par None (sinon JSON invalide)
payload = {k: (None if pd.isna(v) else v) for k, v in payload.items()}

print(json.dumps(payload, ensure_ascii=False, indent=2))