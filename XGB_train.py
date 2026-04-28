import joblib
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

print("Entraînement de XGBoost...")

# 1. Chargement des données préparées
data = joblib.load('data_preprocessed.pkl')
X_train, y_train = data['X_train'], data['y_train']
num_cols, cat_cols = data['num_cols'], data['cat_cols']

# 2. Preprocessor impose: scaling numeriques + OHE categorielles
pre_scaled = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
])

# 3. Pipeline obligatoire: preprocessor -> SMOTETomek -> modele
model = XGBClassifier(random_state=42, eval_metric='logloss')
p = Pipeline([
    ('pre', pre_scaled),
    ('balance', SMOTETomek(random_state=42)),
    ('model', model)
])
p.fit(X_train, y_train)

# 4. Sauvegarde dans le dossier courant
joblib.dump(p, 'model_XGBoost.pkl')

print("Fini ! 'model_XGBoost.pkl' sauvegardé.")
