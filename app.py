from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dashboard Churn EFREI", layout="wide")
st.title("Dashboard Churn Client")

st.markdown(
    "Ce dashboard est concu pour une lecture metier. "
    "On privilegie la coherence (Recall, PR-AUC) plutot qu'un score parfait."
)


def find_project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data" / "customer_churn.csv").exists():
            return candidate
        if (candidate / "customer_churn_business_dataset.csv").exists():
            return candidate
    raise FileNotFoundError("Impossible de localiser le fichier data")


@st.cache_data
def load_dataset() -> pd.DataFrame:
    root = find_project_root()
    primary = root / "data" / "customer_churn.csv"
    fallback = root / "customer_churn_business_dataset.csv"
    data_path = primary if primary.exists() else fallback
    df = pd.read_csv(data_path)
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])
    return df


MODEL_FILENAME = "best_model.pkl"
FINAL_MODEL_LABEL = f"Modele final ({MODEL_FILENAME})"


@st.cache_resource
def load_best_model():
    root = find_project_root()
    model_path = root / "models" / MODEL_FILENAME
    if not model_path.exists():
        return None, f"Modele introuvable: {model_path}"
    try:
        model = joblib.load(model_path)
        return model, None
    except Exception as exc:
        return None, f"Erreur chargement modele: {exc}"


def get_model_label(model) -> str:
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"].__class__.__name__
    return model.__class__.__name__


def apply_feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    temp = data.copy()
    if "tenure_months" in temp.columns:
        tenure_safe = temp["tenure_months"].replace(0, np.nan)
    else:
        tenure_safe = None

    if {"support_tickets", "tenure_months"}.issubset(temp.columns):
        temp["tickets_per_tenure"] = (temp["support_tickets"] / tenure_safe).fillna(0)
    if {"total_revenue", "tenure_months"}.issubset(temp.columns):
        temp["revenue_per_month"] = (temp["total_revenue"] / tenure_safe).fillna(0)
    if {"payment_failures", "tenure_months"}.issubset(temp.columns):
        temp["payment_failure_rate"] = (temp["payment_failures"] / tenure_safe).fillna(0)
    return temp


def build_defaults(df: pd.DataFrame) -> dict:
    defaults = {}
    for col in df.columns:
        if df[col].dtype.kind in "iufc":
            defaults[col] = float(df[col].median())
        else:
            defaults[col] = df[col].mode().iloc[0]
    return defaults


def predict_proba_safe(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    raise ValueError("Le modele ne supporte pas predict_proba ni decision_function")


df_raw = load_dataset()
df_engineered = apply_feature_engineering(df_raw)
raw_defaults = build_defaults(df_raw.drop(columns=["churn"]))

best_model, best_model_error = load_best_model()
best_model_available = best_model_error is None

st.sidebar.markdown("## Reglages")
threshold = st.sidebar.slider(
    "Seuil decision (priorite Recall)", min_value=0.1, max_value=0.9, value=0.35, step=0.05
)
st.sidebar.caption(
    "Choix metier: un seuil bas detecte plus de churners, au prix de plus de faux positifs."
)

tabs = st.tabs(
    [
        "1. Vue d'ensemble",
        "2. Exploration et analyse",
        "3. Prediction en temps reel",
        "4. Simulation de scenarios",
    ]
)

with tabs[0]:
    st.header("Vue d'ensemble")
    st.markdown(
        "On affiche des KPIs clairs pour un public non technique. "
        "Le focus reste sur le risque et l'impact revenu."
    )

    churn_rate = float(df_raw["churn"].mean())
    col1, col2, col3 = st.columns(3)
    col1.metric("Taux de churn observe", f"{churn_rate:.1%}")

    if best_model_available:
        model_label = get_model_label(best_model)
        st.info(f"Modele actif: {FINAL_MODEL_LABEL} - {model_label}.")
        st.caption(
            "Choix metier: on privilegie la detection des churners (Recall) et la PR-AUC "
            "pour les classes desequilibrees."
        )

        X_all = df_engineered.drop(columns=["churn"])
        proba_all = predict_proba_safe(best_model, X_all)
        at_risk_mask = proba_all >= threshold
        at_risk_count = int(at_risk_mask.sum())

        revenue_col = (
            "total_revenue" if "total_revenue" in df_engineered.columns else "monthly_fee"
        )
        revenue_at_risk = float(df_engineered.loc[at_risk_mask, revenue_col].sum())

        col2.metric("Clients a risque", f"{at_risk_count:,}")
        col3.metric("Revenu a risque (estime)", f"{revenue_at_risk:,.0f}")
        st.caption("Les KPIs s'appuient sur le modele final, pas seulement sur l'accuracy.")
    else:
        col2.metric("Clients a risque", "-")
        col3.metric("Revenu a risque (estime)", "-")
        st.warning(best_model_error)

with tabs[1]:
    st.header("Exploration des donnees")
    st.markdown(
        "Filtres simples pour comprendre les segments a risque. "
        "Objectif: trouver des actions metier concretes."
    )

    segment_options = sorted(df_raw["customer_segment"].dropna().unique())
    contract_options = sorted(df_raw["contract_type"].dropna().unique())

    selected_segments = st.multiselect(
        "Segment client", segment_options, default=segment_options
    )
    selected_contracts = st.multiselect(
        "Type de contrat", contract_options, default=contract_options
    )

    filtered = df_raw[
        df_raw["customer_segment"].isin(selected_segments)
        & df_raw["contract_type"].isin(selected_contracts)
    ]

    st.write(f"Lignes selectionnees: {len(filtered):,}")

    churn_by_contract = (
        filtered.groupby("contract_type")["churn"].mean().sort_values(ascending=False)
    )
    st.subheader("Taux de churn par contrat")
    st.bar_chart(churn_by_contract)

    st.subheader("Distribution tenure (mois)")
    fig, ax = plt.subplots()
    ax.hist(filtered["tenure_months"], bins=20, color="#4C78A8")
    ax.set_xlabel("Tenure (mois)")
    ax.set_ylabel("Nb clients")
    st.pyplot(fig)

    if "nps_score" in filtered.columns:
        st.subheader("NPS par statut churn")
        fig_nps, ax_nps = plt.subplots()
        filtered.boxplot(column="nps_score", by="churn", ax=ax_nps)
        ax_nps.set_title("")
        ax_nps.set_xlabel("Churn")
        ax_nps.set_ylabel("NPS")
        st.pyplot(fig_nps)

    if best_model_available:
        st.subheader("Distribution du risque predit")
        risk_scores = predict_proba_safe(best_model, df_engineered.drop(columns=["churn"]))
        fig_risk, ax_risk = plt.subplots()
        ax_risk.hist(risk_scores, bins=20, color="#F28E2B")
        ax_risk.set_xlabel("Probabilite de churn")
        ax_risk.set_ylabel("Nb clients")
        st.pyplot(fig_risk)

        st.subheader("Risque moyen par segment")
        risk_df = df_raw.copy()
        risk_df["risk_score"] = risk_scores
        risk_by_segment = (
            risk_df.groupby("customer_segment")["risk_score"]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(risk_by_segment)
    else:
        st.info("Le modele final est requis pour afficher les scores de risque.")

with tabs[2]:
    st.header("Prediction en temps reel")
    st.markdown(
        "Formulaire simple. Les champs non remplis sont completes par des valeurs "
        "par defaut (medianes et modes)."
    )

    if not best_model_available:
        st.error(best_model_error)
    else:
        with st.form("predict_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age", 18, 100, int(raw_defaults.get("age", 35)))
                tenure = st.number_input(
                    "Tenure (mois)", 0, 120, int(raw_defaults.get("tenure_months", 12))
                )
                contract_type = st.selectbox(
                    "Contract type",
                    sorted(df_raw["contract_type"].dropna().unique()),
                )
            with col2:
                monthly_fee = st.number_input(
                    "Monthly fee", 0, 1000, int(raw_defaults.get("monthly_fee", 50))
                )
                payment_failures = st.number_input(
                    "Payment failures", 0, 20, int(raw_defaults.get("payment_failures", 0))
                )
                nps_score = st.number_input(
                    "NPS score", -100, 100, int(raw_defaults.get("nps_score", 0))
                )
            with col3:
                support_tickets = st.number_input(
                    "Support tickets", 0, 50, int(raw_defaults.get("support_tickets", 0))
                )
                customer_segment = st.selectbox(
                    "Segment",
                    sorted(df_raw["customer_segment"].dropna().unique()),
                )
                complaint_type = st.selectbox(
                    "Complaint type",
                    sorted(df_raw["complaint_type"].dropna().unique()),
                )

            submitted = st.form_submit_button("Predire le risque")

        if submitted:
            row = raw_defaults.copy()
            row.update(
                {
                    "age": float(age),
                    "tenure_months": float(tenure),
                    "contract_type": contract_type,
                    "monthly_fee": float(monthly_fee),
                    "payment_failures": float(payment_failures),
                    "nps_score": float(nps_score),
                    "support_tickets": float(support_tickets),
                    "customer_segment": customer_segment,
                    "complaint_type": complaint_type,
                }
            )
            X_input = apply_feature_engineering(pd.DataFrame([row]))
            prob = float(predict_proba_safe(best_model, X_input)[0])

            st.subheader("Resultat")
            if prob >= threshold:
                st.error(f"Risque eleve de churn: {prob:.1%}")
            else:
                st.success(f"Risque modere: {prob:.1%}")

            st.markdown("**Explication locale (SHAP)**")
            try:
                import shap

                if hasattr(best_model, "named_steps") and "pre" in best_model.named_steps:
                    pre = best_model.named_steps["pre"]
                    inner = best_model.named_steps["model"]
                    X_bg = pre.transform(df_engineered.drop(columns=["churn"]).head(200))
                    X_row = pre.transform(X_input)
                    feature_names = pre.get_feature_names_out()

                    explainer = shap.Explainer(inner, X_bg, feature_names=feature_names)
                    shap_values = explainer(X_row)

                    fig = plt.figure()
                    shap.waterfall_plot(shap_values[0], show=False)
                    st.pyplot(fig)
                else:
                    st.info("SHAP local non disponible: pipeline incomplet.")
            except Exception as exc:
                st.info(f"SHAP non disponible: {exc}")

with tabs[3]:
    st.header("Simulation de scenarios")
    st.markdown(
        "On modifie quelques leviers metier (engagement, NPS, tickets) pour voir l'impact "
        "sur la proba de churn."
    )
    if not best_model_available:
        st.error(best_model_error)
    else:
        base_row = raw_defaults.copy()
        baseline_df = apply_feature_engineering(pd.DataFrame([base_row]))
        baseline_prob = float(predict_proba_safe(best_model, baseline_df)[0])

        col1, col2 = st.columns(2)
        with col1:
            sim_tenure = st.slider(
                "Tenure (mois)", 0, 120, int(base_row.get("tenure_months", 12))
            )
            sim_nps = st.slider("NPS score", -100, 100, int(base_row.get("nps_score", 0)))
            sim_failures = st.slider(
                "Payment failures", 0, 20, int(base_row.get("payment_failures", 0))
            )
        with col2:
            sim_tickets = st.slider(
                "Support tickets", 0, 50, int(base_row.get("support_tickets", 0))
            )
            sim_fee = st.slider("Monthly fee", 0, 1000, int(base_row.get("monthly_fee", 50)))

        sim_row = base_row.copy()
        sim_row.update(
            {
                "tenure_months": float(sim_tenure),
                "nps_score": float(sim_nps),
                "payment_failures": float(sim_failures),
                "support_tickets": float(sim_tickets),
                "monthly_fee": float(sim_fee),
            }
        )
        sim_df = apply_feature_engineering(pd.DataFrame([sim_row]))
        sim_prob = float(predict_proba_safe(best_model, sim_df)[0])

        st.metric("Baseline", f"{baseline_prob:.1%}")
        st.metric("Scenario", f"{sim_prob:.1%}", delta=f"{(sim_prob - baseline_prob):.1%}")
        st.caption("Plus le delta est positif, plus le risque augmente.")
