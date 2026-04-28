import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

print("Début du Preprocessing...")

# 1. Chargement
df = pd.read_csv('customer_churn_business_dataset.csv')

# 2. Nettoyage de base (EDA a montré des valeurs manquantes ici)
df['complaint_type'] = df['complaint_type'].fillna('Aucune Plainte')

# 3. Séparation features / cible
drop_list = ['churn', 'customer_id']
X = df.drop([c for c in drop_list if c in df.columns], axis=1)
y = df['churn']

# 4. Train / Test Split (Avec Stratification pour préserver le déséquilibre)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Sauvegarde des colonnes pour le dashboard et les modèles
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
medians = X.median(numeric_only=True).to_dict()

# On sauvegarde tout ce dont les modèles auront besoin
joblib.dump({
    'X_train': X_train, 'X_test': X_test, 
    'y_train': y_train, 'y_test': y_test,
    'num_cols': num_cols, 'cat_cols': cat_cols,
    'all_cols': X.columns.tolist(),
    'medians': medians
}, 'data_preprocessed.pkl')

print("Preprocessing terminé ! Les données sont prêtes ('data_preprocessed.pkl').")
