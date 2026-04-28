import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

print("Debut du preprocessing...")

NUMERIC_COLS_REQUIRED = [
    "age",
    "tenure",
    "monthly_charges",
    "total_revenue",
    "payment_failures",
    "support_tickets",
    "session_duration",
    "login_frequency",
    "nps_score",
]

CATEGORICAL_COLS_REQUIRED = [
    "gender",
    "contract_type",
    "payment_method",
]


def normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    df_local = dataframe.copy()
    df_local.columns = [c.strip().lower().replace(" ", "_") for c in df_local.columns]

    # Alignement noms colonnes selon le cahier des charges.
    if "tenure_months" in df_local.columns and "tenure" not in df_local.columns:
        df_local["tenure"] = df_local["tenure_months"]
    if "monthly_fee" in df_local.columns and "monthly_charges" not in df_local.columns:
        df_local["monthly_charges"] = df_local["monthly_fee"]

    return df_local


df = pd.read_csv("customer_churn_business_dataset.csv")
df = normalize_columns(df)

if "complaint_type" in df.columns:
    df["complaint_type"] = df["complaint_type"].fillna("Aucune Plainte")

drop_list = ["churn", "customer_id"]
X = df.drop([c for c in drop_list if c in df.columns], axis=1)
y = df["churn"]

# Feature engineering utile pour dashboard et tests scenario
if "tenure" in X.columns:
    tenure_safe = X["tenure"].replace(0, pd.NA)
    if {"support_tickets", "tenure"}.issubset(X.columns):
        X["tickets_per_tenure"] = (X["support_tickets"] / tenure_safe).fillna(0)
    if {"total_revenue", "tenure"}.issubset(X.columns):
        X["revenue_per_month"] = (X["total_revenue"] / tenure_safe).fillna(0)
    if {"payment_failures", "tenure"}.issubset(X.columns):
        X["payment_failure_rate"] = (X["payment_failures"] / tenure_safe).fillna(0)

# Colonnes strictes pour la tache 1 (garde seulement celles disponibles).
num_cols = [c for c in NUMERIC_COLS_REQUIRED if c in X.columns]
cat_cols = [c for c in CATEGORICAL_COLS_REQUIRED if c in X.columns]

required_features = num_cols + cat_cols
X = X[required_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

medians = X.median(numeric_only=True).to_dict()

joblib.dump(
    {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "all_cols": X.columns.tolist(),
        "medians": medians,
    },
    "data_preprocessed.pkl",
)

print("Preprocessing termine: data_preprocessed.pkl genere.")
