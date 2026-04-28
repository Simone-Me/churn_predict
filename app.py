from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

st.set_page_config(page_title="Dashboard Churn EFREI", layout="wide")
st.title("Dashboard Churn Client")
st.markdown("Version metier: prediction scenario + comparaison avancee des modeles.")


def find_project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data" / "customer_churn.csv").exists():
            return candidate
        if (candidate / "customer_churn_business_dataset.csv").exists():
            return candidate
    raise FileNotFoundError("Impossible de localiser les fichiers projet.")


@st.cache_data
def load_dataset() -> pd.DataFrame:
    root = find_project_root()
    primary = root / "data" / "customer_churn.csv"
    fallback = root / "customer_churn_business_dataset.csv"
    path = primary if primary.exists() else fallback
    df = pd.read_csv(path)
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])
    return df


@st.cache_data
def load_info(df_raw: pd.DataFrame) -> dict:
    root = find_project_root()
    info_path = root / "data_preprocessed.pkl"
    if info_path.exists():
        return joblib.load(info_path)

    # Fallback minimal si le fichier preprocess n'a pas encore ete genere.
    X = df_raw.drop(columns=["churn"]) if "churn" in df_raw.columns else df_raw.copy()
    y = df_raw["churn"] if "churn" in df_raw.columns else pd.Series(dtype=float)
    return {
        "X_train": X.copy(),
        "X_test": X.copy(),
        "y_train": y.copy(),
        "y_test": y.copy(),
        "num_cols": X.select_dtypes(exclude=["object"]).columns.tolist(),
        "cat_cols": X.select_dtypes(include=["object"]).columns.tolist(),
        "all_cols": X.columns.tolist(),
        "medians": X.median(numeric_only=True).to_dict(),
    }


@st.cache_resource
def load_models() -> dict:
    root = find_project_root()
    files = {
        "Logistic Regression": root / "model_LogisticRegression.pkl",
        "MLP": root / "model_DeepLearning.pkl",
        "XGBoost": root / "model_XGBoost.pkl",
        "Random Forest": root / "model_RandomForest.pkl",
    }
    loaded = {}
    for name, path in files.items():
        if path.exists():
            loaded[name] = joblib.load(path)
    return loaded


def predict_proba_safe(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    raise ValueError("Modele incompatible: ni predict_proba ni decision_function.")


def build_modes(df: pd.DataFrame) -> dict:
    modes = {}
    for col in df.columns:
        if df[col].dropna().empty:
            modes[col] = "None"
        else:
            modes[col] = df[col].mode(dropna=True).iloc[0]
    return modes


def apply_feature_engineering(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    tenure_col = "tenure_months" if "tenure_months" in data.columns else "tenure"
    if tenure_col in data.columns:
        tenure_safe = data[tenure_col].replace(0, np.nan)
        if {"support_tickets", tenure_col}.issubset(data.columns):
            data["tickets_per_tenure"] = (data["support_tickets"] / tenure_safe).fillna(0)
        if {"total_revenue", tenure_col}.issubset(data.columns):
            data["revenue_per_month"] = (data["total_revenue"] / tenure_safe).fillna(0)
        if {"payment_failures", tenure_col}.issubset(data.columns):
            data["payment_failure_rate"] = (data["payment_failures"] / tenure_safe).fillna(0)
    return data


def build_input_row(info: dict, medians: dict, modes: dict) -> dict:
    row = {}
    for col in info["all_cols"]:
        if col in info["num_cols"]:
            row[col] = float(medians.get(col, 0.0))
        else:
            row[col] = modes.get(col, "None")
    return row


def top_feature_importance(model, reference_x: pd.DataFrame) -> pd.Series | None:
    if not hasattr(model, "named_steps") or "pre" not in model.named_steps:
        return None
    pre = model.named_steps["pre"]
    inner = model.named_steps["model"]
    feature_names = pre.get_feature_names_out()

    if hasattr(inner, "feature_importances_"):
        vals = pd.Series(inner.feature_importances_, index=feature_names)
        return vals.sort_values(ascending=False).head(15)

    if hasattr(inner, "coef_"):
        coef = np.abs(inner.coef_).ravel()
        vals = pd.Series(coef, index=feature_names)
        return vals.sort_values(ascending=False).head(15)

    return None


df_raw = load_dataset()
info = load_info(df_raw)
models = load_models()
if not models:
    st.error("Aucun modele charge. Lance d'abord les scripts d'entrainement.")
    st.stop()

modes = build_modes(df_raw.drop(columns=["churn"], errors="ignore"))
seuil = st.sidebar.slider("Seuil de decision", 0.1, 0.9, 0.35, 0.05)
selected_model_name = st.sidebar.selectbox("Modele actif", sorted(models.keys()))
model = models[selected_model_name]

tab1, tab2 = st.tabs(["Prediction Scenario", "Comparaison & Analyse"])

with tab1:
    st.header(f"Scenario avec {selected_model_name}")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 35)
        tenure = st.number_input("Anciennete (mois)", 0, 120, 12)
        contract = st.selectbox(
            "Contrat",
            ["Monthly", "Yearly", "Two-Year"],
        )
    with col2:
        fee = st.number_input("Frais mensuels (EUR)", 0, 1000, 50)
        complaint = st.selectbox(
            "Plainte",
            ["Aucune Plainte", "Service", "Billing", "Technical"],
        )
        segment = st.selectbox("Segment", ["Individual", "SME", "Enterprise"])

    if st.button("Predire le risque"):
        medians = info.get("medians", {})
        data = build_input_row(info, medians, modes)

        if "age" in data:
            data["age"] = float(age)
        if "tenure_months" in data:
            data["tenure_months"] = float(tenure)
        if "tenure" in data:
            data["tenure"] = float(tenure)
        if "monthly_fee" in data:
            data["monthly_fee"] = float(fee)
        if "monthly_charges" in data:
            data["monthly_charges"] = float(fee)
        if "contract_type" in data:
            data["contract_type"] = contract
        if "complaint_type" in data:
            data["complaint_type"] = complaint
        if "customer_segment" in data:
            data["customer_segment"] = segment

        df_test = pd.DataFrame([data])
        df_test = apply_feature_engineering(df_test)
        for col in info["all_cols"]:
            if col not in df_test.columns:
                df_test[col] = data.get(col, 0.0)
        df_test = df_test[info["all_cols"]]

        prob = float(predict_proba_safe(model, df_test)[0])
        st.session_state["last_prob"] = prob

        st.subheader("Resultat")
        if prob >= seuil:
            st.error(f"Risque eleve de depart ({prob:.1%}) - au dessus du seuil {seuil:.1%}")
        else:
            st.success(f"Client probablement fidele ({prob:.1%}) - sous le seuil {seuil:.1%}")

        # Comparaison au reste des clients
        X_test = info["X_test"]
        peer_probs = predict_proba_safe(model, X_test)
        percentile = float((peer_probs < prob).mean() * 100)
        st.info(f"Comparaison: ce client est plus risque que {percentile:.1f}% des clients test.")

with tab2:
    st.header("Analyse comparative")
    X_test = info["X_test"]
    y_test = info["y_test"]

    st.subheader("1. Performances des modeles sur le jeu de test")
    st.caption("Ce tableau montre le compromis precision/recall/f1/ROC-AUC pour comparer les modeles.")

    metrics_data = []
    roc_curves = {}
    for nom, m in models.items():
        y_pred_proba = predict_proba_safe(m, X_test)
        y_pred = (y_pred_proba >= seuil).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        metrics_data.append(
            {
                "Modele": nom,
                "Accuracy": f"{acc:.2f}",
                "Precision": f"{prec:.2f}",
                "Recall": f"{rec:.2f}",
                "F1-Score": f"{f1:.2f}",
                "ROC-AUC": f"{roc_auc:.2f}",
            }
        )
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_curves[nom] = (fpr, tpr, roc_auc)

    st.table(pd.DataFrame(metrics_data))

    st.subheader("2. Comparaison des courbes ROC")
    st.caption("La courbe ROC visualise la capacite a separer churners et non churners.")
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    for nom, (fpr, tpr, roc_auc) in roc_curves.items():
        ax_roc.plot(fpr, tpr, label=f"{nom} (AUC = {roc_auc:.2f})")
    ax_roc.plot([0, 1], [0, 1], "k--", label="Aleatoire")
    ax_roc.set_xlabel("Taux de faux positifs (FPR)")
    ax_roc.set_ylabel("Taux de vrais positifs (TPR)")
    ax_roc.set_title("Courbes ROC")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    st.subheader(f"3. Importance des variables ({selected_model_name})")
    st.caption("Ce graphe indique les variables qui influencent le plus la decision du modele selectionne.")
    imp = top_feature_importance(model, X_test)
    if imp is None:
        st.info("Importance indisponible pour ce modele.")
    else:
        fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
        imp.sort_values(ascending=True).plot(kind="barh", ax=ax_imp, color="#4C78A8")
        ax_imp.set_xlabel("Importance")
        ax_imp.set_ylabel("Variables")
        st.pyplot(fig_imp)

    st.subheader("4. Segmentation avancee: churn par segment et contrat")
    st.caption(
        "Ce heatmap montre les zones de risque par combinaison segment x contrat, utile pour cibler les campagnes."
    )
    if {"customer_segment", "contract_type", "churn"}.issubset(df_raw.columns):
        heat = (
            df_raw.groupby(["customer_segment", "contract_type"])["churn"]
            .mean()
            .unstack(fill_value=0)
        )
        st.dataframe(heat.style.format("{:.1%}"))
    else:
        st.info("Colonnes segment/contrat manquantes pour ce graphe.")

    st.subheader("5. Cohortes d'anciennete")
    st.caption("Ce graphe compare le churn selon l'anciennete, pour identifier les phases critiques du cycle client.")
    tenure_col = "tenure_months" if "tenure_months" in df_raw.columns else "tenure"
    if tenure_col in df_raw.columns:
        cohort = df_raw.copy()
        cohort["tenure_bin"] = pd.cut(
            cohort[tenure_col], bins=[-1, 3, 6, 12, 24, 60, 200], include_lowest=True
        )
        churn_by_bin = cohort.groupby("tenure_bin")["churn"].mean()
        fig_bin, ax_bin = plt.subplots(figsize=(8, 4))
        churn_by_bin.plot(kind="bar", ax=ax_bin, color="#F28E2B")
        ax_bin.set_ylabel("Taux de churn")
        ax_bin.set_xlabel("Cohorte anciennete")
        st.pyplot(fig_bin)
    else:
        st.info("Colonne d'anciennete absente.")

    st.subheader("6. Impact des plaintes")
    st.caption("Ce graphe montre quels types de plaintes sont les plus associes au churn.")
    if {"complaint_type", "churn"}.issubset(df_raw.columns):
        complaint_churn = df_raw.groupby("complaint_type")["churn"].mean().sort_values(ascending=False)
        st.bar_chart(complaint_churn)
    else:
        st.info("Colonne complaint_type absente.")

    st.subheader("7. Comparaison avec autres clients (percentiles de risque)")
    st.caption("La distribution des probabilites situe le client saisi par rapport a la population test.")
    peer_probs = predict_proba_safe(model, X_test)
    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
    ax_hist.hist(peer_probs, bins=30, color="#59A14F")
    if "last_prob" in st.session_state:
        ax_hist.axvline(st.session_state["last_prob"], color="red", linestyle="--", linewidth=2)
    ax_hist.set_xlabel("Probabilite de churn")
    ax_hist.set_ylabel("Nombre de clients")
    st.pyplot(fig_hist)
