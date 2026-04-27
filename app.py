import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard Churn EFREI")

st.title("Analyse du Churn Client")

@st.cache_resource
def load_all():
    try:
        models = joblib.load('all_models.pkl')
        info = joblib.load('features_info.pkl')
        return models, info
    except:
        return None, None

models, info = load_all()

if models is None:
    st.error("Erreur : Lancez le notebook pour générer 'all_models.pkl'")
    st.stop()

# Selection du modele en haut de page
selected_model_name = st.sidebar.selectbox("Modèle utilisé pour la prédiction :", list(models.keys()))
model = models[selected_model_name]

tab1, tab2 = st.tabs(["Prédiction Scénario", "Comparaison & Analyse"])

with tab1:
    st.header(f"Scénario avec {selected_model_name}")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Age', 18, 100, 35)
        tenure = st.number_input('Ancienneté (mois)', 0, 100, 12)
        contract = st.selectbox('Contrat', ['Monthly', 'Yearly', 'Two-Year'])
    with col2:
        fee = st.number_input('Frais mensuels (€)', 0, 500, 50)
        complaint = st.selectbox('Plainte', ['Aucune Plainte', 'Service', 'Billing', 'Technical'])
        segment = st.selectbox('Segment', ['Individual', 'SME', 'Enterprise'])

    if st.button('Prédire le risque'):
        medians = info.get('medians', {})
        data = {c: float(medians.get(c, 0.0)) if c in info['num'] else 'None' for c in info['all']}
        data.update({
            'age': float(age), 'tenure_months': float(tenure), 'monthly_fee': float(fee),
            'contract_type': contract, 'complaint_type': complaint, 'customer_segment': segment
        })
        df_test = pd.DataFrame([data])[info['all']]
        prob = model.predict_proba(df_test)[0][1]
        
        st.subheader("Résultat :")
        if prob > 0.5:
            st.error(f"Risque élevé de départ ({prob:.1%})")
        else:
            st.success(f"Client probablement fidèle ({prob:.1%})")

with tab2:
    st.header("Analyse comparative")
    
    st.subheader("1. Rappel des scores (Notebook)")
    df_comp = pd.DataFrame({
        'Modèle': ['RandomForest', 'XGBoost', 'LogisticRegression', 'DeepLearning (MLP)'],
        'Score Moyen (CV)': [0.88, 0.86, 0.82, 0.83]
    })
    st.table(df_comp)

    st.subheader(f"2. Importance des variables ({selected_model_name})")
    try:
        # On essaie de recuperer l'importance si le modele le permet (ex: RF, XGB)
        inner_model = model.steps[1][1]
        preprocessor = model.steps[0][1]
        
        if hasattr(inner_model, 'feature_importances_'):
            noms_colonnes = preprocessor.get_feature_names_out()
            feat_imp = pd.Series(inner_model.feature_importances_, index=noms_colonnes).sort_values(ascending=False).head(10)
            fig, ax = plt.subplots()
            feat_imp.plot(kind='barh', ax=ax, color='skyblue')
            ax.set_title(f"Top 10 - {selected_model_name}")
            st.pyplot(fig)
        else:
            st.write(f"L'importance des variables n'est pas disponible pour {selected_model_name} (uniquement RandomForest et XGBoost).")
    except Exception as e:
        st.write(f"Graphique indisponible : {e}")
