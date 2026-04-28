import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

st.set_page_config(page_title="Dashboard Churn EFREI")

st.title("Analyse du Churn Client")

@st.cache_resource
def load_all():
    try:
        models = {
            'LogisticRegression': joblib.load('model_LogisticRegression.pkl'),
            'RandomForest': joblib.load('model_RandomForest.pkl'),
            'XGBoost': joblib.load('model_XGBoost.pkl'),
            'DeepLearning (MLP)': joblib.load('model_DeepLearning.pkl')
        }
        info = joblib.load('data_preprocessed.pkl')
        return models, info
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        return None, None

models, info = load_all()

if models is None:
    st.error("Erreur : Lancez les scripts d'entraînement pour générer les modèles.")
    st.stop()

# Selection du modele et du seuil en haut de page
selected_model_name = st.sidebar.selectbox("Modèle utilisé pour la prédiction :", list(models.keys()))
model = models[selected_model_name]

st.sidebar.markdown("---")
seuil = st.sidebar.slider("Seuil de décision (Optimisation Recall)", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
st.sidebar.caption("Un seuil plus bas augmente le Recall (détecte plus de risques) mais augmente les Faux Positifs.")

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
        data = {c: float(medians.get(c, 0.0)) if c in info['num_cols'] else 'None' for c in info['all_cols']}
        data.update({
            'age': float(age), 'tenure_months': float(tenure), 'monthly_fee': float(fee),
            'contract_type': contract, 'complaint_type': complaint, 'customer_segment': segment
        })
        df_test = pd.DataFrame([data])[info['all_cols']]
        prob = model.predict_proba(df_test)[0][1]
        
        st.subheader("Résultat :")
        if prob >= seuil:
            st.error(f"Risque élevé de départ ({prob:.1%}) - Dépasse le seuil de {seuil:.1%}")
        else:
            st.success(f"Client probablement fidèle ({prob:.1%}) - Sous le seuil de {seuil:.1%}")

with tab2:
    st.header("Analyse comparative")
    
    st.subheader("1. Performances des Modèles sur le Jeu de Test")
    
    # Récupération des données de test
    X_test = info['X_test']
    y_test = info['y_test']
    
    metrics_data = []
    
    # On va stocker les courbes ROC pour le graphique
    roc_curves = {}
    
    for nom, m in models.items():
        y_pred_proba = m.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= seuil).astype(int)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        metrics_data.append({
            'Modèle': nom,
            'Accuracy': f"{acc:.2f}",
            'Precision': f"{prec:.2f}",
            'Recall': f"{rec:.2f}",
            'F1-Score': f"{f1:.2f}",
            'ROC-AUC': f"{roc_auc:.2f}"
        })
        
        # Calcul courbe ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_curves[nom] = (fpr, tpr, roc_auc)
        
    df_comp = pd.DataFrame(metrics_data)
    st.table(df_comp)
    
    st.subheader("2. Comparaison des Courbes ROC")
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    for nom, (fpr, tpr, roc_auc) in roc_curves.items():
        ax_roc.plot(fpr, tpr, label=f"{nom} (AUC = {roc_auc:.2f})")
    
    ax_roc.plot([0, 1], [0, 1], 'k--', label="Aléatoire")
    ax_roc.set_xlabel("Taux de Faux Positifs (FPR)")
    ax_roc.set_ylabel("Taux de Vrais Positifs (TPR / Recall)")
    ax_roc.set_title("Courbes ROC")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    st.subheader(f"3. Importance des variables ({selected_model_name})")
    try:
        # On essaie de recuperer l'importance si le modele le permet (ex: RF, XGB)
        inner_model = model.named_steps['model']
        preprocessor = model.named_steps['pre']
        
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
