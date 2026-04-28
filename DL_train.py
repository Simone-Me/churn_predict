import joblib
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier

print("Entraînement du Deep Learning (MLP)...")

# 1. Chargement des données préparées
data = joblib.load('data_preprocessed.pkl')
X_train, y_train = data['X_train'], data['y_train']
num_cols, cat_cols = data['num_cols'], data['cat_cols']

# 2. Preprocessor AVEC mise à l'échelle (indispensable pour les réseaux de neurones)
pre_scaled = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# 3. Modèle et Rééchantillonnage (SMOTE)
model = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=500, random_state=42)
p = Pipeline([
    ('pre', pre_scaled),
    ('smote', SMOTE(random_state=42)),
    ('model', model)
])
p.fit(X_train, y_train)

# 4. Sauvegarde dans le dossier courant
joblib.dump(p, 'model_DeepLearning.pkl')

print("Fini ! 'model_DeepLearning.pkl' sauvegardé.")
