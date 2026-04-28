import joblib
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

print("Entraînement de XGBoost...")

# 1. Chargement des données préparées
data = joblib.load('data_preprocessed.pkl')
X_train, y_train = data['X_train'], data['y_train']
num_cols, cat_cols = data['num_cols'], data['cat_cols']

# 2. Preprocessor SANS mise à l'échelle (inutile pour XGBoost)
pre_unscaled = ColumnTransformer([
    ('num', 'passthrough', num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# 3. Modèle et Rééchantillonnage (SMOTE)
model = XGBClassifier(random_state=42)
p = Pipeline([
    ('pre', pre_unscaled),
    ('smote', SMOTE(random_state=42)),
    ('model', model)
])
p.fit(X_train, y_train)

# 4. Sauvegarde dans le dossier courant
joblib.dump(p, 'model_XGBoost.pkl')

print("Fini ! 'model_XGBoost.pkl' sauvegardé.")
