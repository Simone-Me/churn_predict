from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from feature_engineering import prepare_customer_features


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"
PREPROCESSED_PATH = BASE_DIR / "data_preprocessed.pkl"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
REPORT_PATH = BASE_DIR / "reports" / "model_comparison.csv"
DEFAULT_THRESHOLD = 0.35


st.set_page_config(page_title="Dashboard Churn Client", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    info = joblib.load(PREPROCESSED_PATH) if PREPROCESSED_PATH.exists() else None
    raw_data = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None
    comparison = pd.read_csv(REPORT_PATH) if REPORT_PATH.exists() else None
    return model, info, raw_data, comparison


def predict_scores(model, X):
    return model.predict_proba(X)[:, 1]


def build_default_customer(info, raw_data):
    medians = info.get("medians", {})
    modes = info.get("modes", {})
    values = {}
    for col in info.get("input_cols", info["all_cols"]):
        if col in medians:
            values[col] = float(medians.get(col, 0.0))
        else:
            values[col] = modes.get(col, "Unknown")
    if raw_data is not None:
        values["total_revenue"] = float(raw_data["total_revenue"].median())
    return values


def format_feature_name(name):
    return name.replace("num__", "").replace("cat__", "").replace("_", " ").title()


def model_metrics(model, X_test, y_test, threshold):
    scores = predict_scores(model, X_test)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, scores),
        "Faux negatifs": fn,
        "Faux positifs": fp,
        "Vrais positifs": tp,
        "Vrais negatifs": tn,
    }


model, info, raw_data, comparison = load_artifacts()

st.title("Pilotage du risque de churn client")

if model is None or info is None or raw_data is None:
    st.error("Artefacts manquants. Lance d'abord le notebook `06_entrainement_complet.ipynb`.")
    st.stop()

threshold = st.sidebar.slider(
    "Seuil d'alerte churn",
    min_value=0.35,
    max_value=0.90,
    value=DEFAULT_THRESHOLD,
    step=0.05,
)
st.sidebar.caption("Modele retenu : Logistic Regression, choisi pour son recall.")

X_test = info["X_test"]
y_test = info["y_test"]

tab_business, tab_simulation, tab_explain = st.tabs(
    ["Vue metier", "Simulation client", "Explication"]
)

with tab_business:
    st.subheader("Indicateurs de retention")

    scored = raw_data.copy()
    features = prepare_customer_features(scored)[info["all_cols"]]
    scored["proba_churn"] = predict_scores(model, features)
    scored["client_a_risque"] = scored["proba_churn"] >= threshold
    scored["revenu_a_risque"] = scored["monthly_fee"] * scored["proba_churn"]

    risky = scored[scored["client_a_risque"]]
    churn_rate = raw_data["churn"].mean()
    total_risk_revenue = scored["revenu_a_risque"].sum()
    metrics = model_metrics(model, X_test, y_test, threshold)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Clients", f"{len(scored):,}".replace(",", " "))
    kpi2.metric("Churn observe", f"{churn_rate:.1%}")
    kpi3.metric("Clients a risque", f"{len(risky):,}".replace(",", " "))
    kpi4.metric("Revenu mensuel a risque", f"{total_risk_revenue:,.0f} EUR".replace(",", " "))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recall modele", f"{metrics['Recall']:.1%}")
    m2.metric("Precision", f"{metrics['Precision']:.1%}")
    m3.metric("F1-score", f"{metrics['F1-score']:.3f}")
    m4.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("#### Clients a prioriser")
        priority_cols = [
            "customer_id",
            "customer_segment",
            "contract_type",
            "monthly_fee",
            "total_revenue",
            "proba_churn",
            "revenu_a_risque",
            "nps_score",
            "support_tickets",
        ]
        st.dataframe(
            scored.sort_values("revenu_a_risque", ascending=False)[priority_cols]
            .head(20)
            .style.format({"proba_churn": "{:.1%}", "revenu_a_risque": "{:.0f} EUR"}),
            use_container_width=True,
        )

    with right:
        st.markdown("#### Risque par segment")
        segment_risk = (
            scored.groupby("customer_segment", as_index=False)
            .agg(
                proba_churn=("proba_churn", "mean"),
                revenu_a_risque=("revenu_a_risque", "sum"),
            )
            .sort_values("proba_churn", ascending=False)
        )
        st.bar_chart(segment_risk, x="customer_segment", y="proba_churn")
        st.dataframe(
            segment_risk.style.format(
                {"proba_churn": "{:.1%}", "revenu_a_risque": "{:.0f} EUR"}
            ),
            use_container_width=True,
        )

with tab_simulation:
    st.subheader("Simulation d'un client")
    values = build_default_customer(info, raw_data)

    col1, col2, col3 = st.columns(3)
    with col1:
        values["age"] = st.number_input("Age", 18, 100, int(values.get("age", 35)))
        values["tenure_months"] = st.number_input(
            "Anciennete (mois)", 0, 120, int(values.get("tenure_months", 12))
        )
        values["contract_type"] = st.selectbox(
            "Type de contrat", ["Monthly", "Quarterly", "Yearly"]
        )
    with col2:
        values["monthly_fee"] = st.number_input(
            "Frais mensuels", 0, 500, int(values.get("monthly_fee", 50))
        )
        values["payment_failures"] = st.number_input(
            "Echecs de paiement", 0, 10, int(values.get("payment_failures", 0))
        )
        values["price_increase_last_3m"] = st.selectbox(
            "Hausse de prix recente", ["No", "Yes"]
        )
    with col3:
        values["monthly_logins"] = st.number_input(
            "Connexions mensuelles", 0, 100, int(values.get("monthly_logins", 15))
        )
        values["support_tickets"] = st.number_input(
            "Tickets support", 0, 30, int(values.get("support_tickets", 2))
        )
        values["nps_score"] = st.number_input(
            "NPS", -100, 100, int(values.get("nps_score", 20))
        )

    values["complaint_type"] = st.selectbox(
        "Type de plainte", ["Aucune Plainte", "Service", "Billing", "Technical"]
    )
    values["customer_segment"] = st.selectbox("Segment", ["Individual", "SME", "Enterprise"])

    scenario = prepare_customer_features(pd.DataFrame([values]))[info["all_cols"]]
    probability = float(predict_scores(model, scenario)[0])
    expected_loss = float(values["monthly_fee"] * probability)

    result_col, action_col = st.columns([1, 1])
    with result_col:
        st.metric("Probabilite de churn", f"{probability:.1%}")
        st.metric("Perte mensuelle attendue", f"{expected_loss:.0f} EUR")
    with action_col:
        if probability >= threshold:
            st.error("Client a traiter en priorite.")
            st.write("Actions recommandees : verifier le support, proposer une offre ciblee, suivre le paiement.")
        else:
            st.success("Risque sous le seuil d'alerte.")
            st.write("Actions recommandees : maintenir l'engagement et surveiller les signaux faibles.")

with tab_explain:
    st.subheader("Pourquoi le modele alerte certains clients ?")
    st.write(
        "Cette partie sert a expliquer les variables qui aident le plus le modele a detecter le churn."
    )

    sample_size = min(500, len(X_test))
    result = permutation_importance(
        model,
        X_test.iloc[:sample_size],
        y_test.iloc[:sample_size],
        scoring="recall",
        n_repeats=5,
        random_state=42,
    )
    top = (
        pd.Series(result.importances_mean, index=X_test.columns)
        .sort_values(ascending=False)
        .head(15)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    top.rename(index=format_feature_name).plot(kind="barh", ax=ax, color="#2563eb")
    ax.set_title("Variables les plus utiles pour le recall")
    st.pyplot(fig)

    if comparison is not None:
        st.markdown("#### Rappel du choix du modele")
        st.dataframe(
            comparison[["modele", "test_recall", "test_precision", "test_f1", "test_roc_auc"]]
            .sort_values("test_recall", ascending=False)
            .style.format(
                {
                    "test_recall": "{:.3f}",
                    "test_precision": "{:.3f}",
                    "test_f1": "{:.3f}",
                    "test_roc_auc": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

    st.caption(
        "Ces resultats montrent des associations apprises par le modele. Ils ne prouvent pas une causalite."
    )
