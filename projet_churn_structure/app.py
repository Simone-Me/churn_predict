from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from feature_engineering import prepare_customer_features

try:
    import shap
except ImportError:
    shap = None


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"
PREPROCESSED_PATH = BASE_DIR / "data_preprocessed.pkl"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
REPORT_PATH = BASE_DIR / "reports" / "model_comparison.csv"
THRESHOLD_REPORT_PATH = BASE_DIR / "reports" / "threshold_analysis.csv"
BASELINE_REPORT_PATH = BASE_DIR / "reports" / "baseline_analysis.csv"
DEFAULT_THRESHOLD = 0.50


st.set_page_config(page_title="Dashboard Churn Client", layout="wide")


@st.cache_resource
def load_artifacts():
    """Charge les artefacts generes par le notebook 04 une seule fois."""
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    info = joblib.load(PREPROCESSED_PATH) if PREPROCESSED_PATH.exists() else None
    raw_data = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None
    comparison = pd.read_csv(REPORT_PATH) if REPORT_PATH.exists() else None
    thresholds = pd.read_csv(THRESHOLD_REPORT_PATH) if THRESHOLD_REPORT_PATH.exists() else None
    baseline = pd.read_csv(BASELINE_REPORT_PATH) if BASELINE_REPORT_PATH.exists() else None
    return model, info, raw_data, comparison, thresholds, baseline


def predict_scores(model, X):
    """Retourne la probabilite de churn, necessaire pour appliquer un seuil metier."""
    return model.predict_proba(X)[:, 1]


def build_default_customer(info, raw_data):
    """Construit un client moyen pour pre-remplir la simulation."""
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
    """Rend les noms de variables plus lisibles dans les graphiques."""
    return name.replace("num__", "").replace("cat__", "").replace("_", " ").title()


def get_transformed_feature_names(model, X_sample):
    """Recupere les noms apres preprocessing pour les importances natives et SHAP."""
    preprocessor = model.named_steps.get("pre")
    if preprocessor is None:
        return X_sample.columns.tolist()
    try:
        return preprocessor.get_feature_names_out()
    except AttributeError:
        return X_sample.columns.tolist()


def model_metrics(model, X_test, y_test, threshold):
    """Calcule les metriques au seuil choisi par l'utilisateur metier."""
    scores = predict_scores(model, X_test)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, scores),
        "PR-AUC": average_precision_score(y_test, scores),
        "Faux negatifs": fn,
        "Faux positifs": fp,
        "Vrais positifs": tp,
        "Vrais negatifs": tn,
    }


GRID_FIGSIZE = (4.3, 3.6)


def style_grid_axes(ax, title, xlabel=None, ylabel=None):
    """Applique une taille de police compacte adaptee a une grille de petits graphiques."""
    ax.set_title(title, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)


def render_chart_grid(figures, n_cols=3):
    """Affiche une liste de figures matplotlib en grille (n_cols par ligne) pour eviter
    des graphiques trop grands qui obligent a scroller."""
    for start in range(0, len(figures), n_cols):
        row_figures = figures[start : start + n_cols]
        columns = st.columns(n_cols)
        for col, fig in zip(columns, row_figures):
            with col:
                st.pyplot(fig, width="stretch")
                plt.close(fig)


def plot_baseline_metrics(baseline):
    """Graphique du notebook 05 : l'accuracy seule peut cacher un recall nul."""
    fig, ax = plt.subplots(figsize=GRID_FIGSIZE)
    metrics = ["accuracy", "recall", "precision", "f1"]
    x = np.arange(len(baseline["modele"]))
    width = 0.18

    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, baseline[metric], width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(baseline["modele"], rotation=25, ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    style_grid_axes(ax, "Baseline : accuracy elevee, recall faible", ylabel="Score")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.32), fontsize=7)
    fig.tight_layout()
    return fig


def plot_model_recall(comparison):
    """Graphique du notebook 05 : recall par modele."""
    plot_df = comparison.sort_values("test_recall", ascending=True)
    fig, ax = plt.subplots(figsize=GRID_FIGSIZE)
    ax.barh(plot_df["modele"], plot_df["test_recall"], color="#2563eb")
    style_grid_axes(ax, "Recall par modele (seuil 0.5)", xlabel="Recall")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig


def plot_model_prauc(comparison):
    """Graphique du notebook 05 : PR-AUC par modele."""
    plot_df = comparison.sort_values("test_pr_auc", ascending=True)
    fig, ax = plt.subplots(figsize=GRID_FIGSIZE)
    ax.barh(plot_df["modele"], plot_df["test_pr_auc"], color="#16a34a")
    style_grid_axes(ax, "PR-AUC par modele", xlabel="PR-AUC")
    ax.set_xlim(0, max(0.4, plot_df["test_pr_auc"].max() + 0.05))
    fig.tight_layout()
    return fig


def plot_metric_comparison(comparison):
    """Graphique du notebook 05 : comparaison multi-metriques des meilleurs modeles."""
    metric_cols = [
        "test_accuracy",
        "test_recall",
        "test_precision",
        "test_f1",
        "test_roc_auc",
        "test_pr_auc",
    ]
    metric_labels = ["Accuracy", "Recall", "Precision", "F1", "ROC-AUC", "PR-AUC"]
    ordered = comparison.sort_values("test_recall", ascending=False).head(6)

    fig, ax = plt.subplots(figsize=GRID_FIGSIZE)
    x = np.arange(len(ordered))
    width = 0.12

    for i, col in enumerate(metric_cols):
        ax.bar(x + (i - 2.5) * width, ordered[col], width, label=metric_labels[i])

    ax.set_xticks(x)
    ax.set_xticklabels(ordered["modele"], rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    style_grid_axes(ax, "Comparaison multi-metriques (top recall)", ylabel="Score")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.38), fontsize=6.5)
    fig.tight_layout()
    return fig


def plot_error_bars(comparison):
    """Graphique du notebook 05 : faux negatifs et faux positifs par modele."""
    error_df = comparison.sort_values("test_fn", ascending=True)
    fig, ax = plt.subplots(figsize=GRID_FIGSIZE)
    x = np.arange(len(error_df))
    width = 0.38

    ax.bar(x - width / 2, error_df["test_fn"], width, label="Faux negatifs", color="#dc2626")
    ax.bar(x + width / 2, error_df["test_fp"], width, label="Faux positifs", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(error_df["modele"], rotation=30, ha="right", fontsize=7)
    style_grid_axes(ax, "Erreurs par modele (seuil 0.5)", ylabel="Nombre de clients")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_threshold_scores(thresholds, final_model, recommended_threshold):
    """Graphique du notebook 05 : precision/recall/F1 selon le seuil."""
    final_thresholds = thresholds[thresholds["modele"] == final_model].sort_values("threshold")
    fig, ax = plt.subplots(figsize=GRID_FIGSIZE)

    ax.plot(final_thresholds["threshold"], final_thresholds["recall"], marker="o", label="Recall")
    ax.plot(
        final_thresholds["threshold"],
        final_thresholds["precision"],
        marker="o",
        label="Precision",
    )
    ax.plot(final_thresholds["threshold"], final_thresholds["f1"], marker="o", label="F1")
    ax.axvline(recommended_threshold, color="black", linestyle="--", label="Seuil retenu")
    style_grid_axes(ax, "Precision / Recall / F1 selon le seuil", xlabel="Seuil", ylabel="Score")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_threshold_errors(thresholds, final_model, recommended_threshold):
    """Graphique du notebook 05 : faux positifs et faux negatifs selon le seuil."""
    final_thresholds = thresholds[thresholds["modele"] == final_model].sort_values("threshold")
    fig, ax = plt.subplots(figsize=GRID_FIGSIZE)

    ax.plot(
        final_thresholds["threshold"],
        final_thresholds["fp"],
        marker="o",
        label="Faux positifs",
        color="#f59e0b",
    )
    ax.plot(
        final_thresholds["threshold"],
        final_thresholds["fn"],
        marker="o",
        label="Faux negatifs",
        color="#dc2626",
    )
    ax.axvline(recommended_threshold, color="black", linestyle="--", label="Seuil retenu")
    style_grid_axes(
        ax, "Faux positifs / faux negatifs selon le seuil", xlabel="Seuil", ylabel="Nombre de clients"
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_training_time(comparison):
    """Cout de calcul par modele (ecoresponsabilite) : temps d'entrainement en secondes."""
    plot_df = comparison.sort_values("temps_entrainement_secondes", ascending=True)
    fig, ax = plt.subplots(figsize=GRID_FIGSIZE)
    ax.barh(plot_df["modele"], plot_df["temps_entrainement_secondes"], color="#0891b2")
    style_grid_axes(
        ax, "Temps d'entrainement (cout de calcul)", xlabel="Secondes (CV 3 folds + entrainement)"
    )
    fig.tight_layout()
    return fig


def plot_radar(comparison):
    """Graphique etoile du notebook 05 pour comparer les meilleurs modeles."""
    radar_metrics = [
        "test_accuracy",
        "test_recall",
        "test_precision",
        "test_f1",
        "test_roc_auc",
        "test_pr_auc",
    ]
    radar_labels = ["Accuracy", "Recall", "Precision", "F1", "ROC-AUC", "PR-AUC"]
    radar_models = comparison.sort_values(
        ["modele_final", "test_recall", "test_pr_auc"], ascending=False
    ).head(4)
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=GRID_FIGSIZE, subplot_kw={"polar": True})
    for _, row in radar_models.iterrows():
        values = row[radar_metrics].astype(float).tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["modele"].replace("_", " "))
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=7)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=7)
    ax.set_title("Comparaison multi-metriques (etoile)", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=6.5)
    fig.tight_layout()
    return fig


def plot_native_feature_importance(model, X_sample, top_n=10):
    """Importance native apres entrainement, disponible pour les modeles d'arbres."""
    estimator = model.named_steps.get("model")
    if estimator is None or not hasattr(estimator, "feature_importances_"):
        return None

    names = get_transformed_feature_names(model, X_sample)
    importances = pd.Series(estimator.feature_importances_, index=names)
    top = importances.sort_values(ascending=False).head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=GRID_FIGSIZE)
    top.rename(index=format_feature_name).plot(kind="barh", ax=ax, color="#0f766e")
    style_grid_axes(ax, "Feature importances natives", xlabel="Importance")
    fig.tight_layout()
    return fig


def plot_permutation_importance(top_series):
    """Permutation importance : variables les plus utiles pour le recall."""
    fig, ax = plt.subplots(figsize=GRID_FIGSIZE)
    top_series.rename(index=format_feature_name).plot(kind="barh", ax=ax, color="#2563eb")
    style_grid_axes(ax, "Permutation importance (recall)", xlabel="Perte de recall")
    fig.tight_layout()
    return fig


def plot_shap_importance(model, X_sample, top_n=10):
    """SHAP sur le modele final selectionne, pour une explication plus avancee."""
    if shap is None:
        return None

    estimator = model.named_steps.get("model")
    preprocessor = model.named_steps.get("pre")
    if estimator is None or preprocessor is None:
        return None

    transformed = preprocessor.transform(X_sample)
    names = get_transformed_feature_names(model, X_sample)

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    top = pd.Series(mean_abs, index=names).sort_values(ascending=False).head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=GRID_FIGSIZE)
    top.rename(index=format_feature_name).plot(kind="barh", ax=ax, color="#7c3aed")
    style_grid_axes(ax, "SHAP moyen absolu", xlabel="Impact moyen sur la prediction")
    fig.tight_layout()
    return fig


def plot_roc_curve_final(y_test, scores, model_name):
    """Courbe ROC (taux de vrais positifs vs taux de faux positifs) du modele final."""
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc = roc_auc_score(y_test, scores)

    fig, ax = plt.subplots(figsize=GRID_FIGSIZE)
    ax.plot(fpr, tpr, color="#2563eb", linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Modele aleatoire")
    style_grid_axes(
        ax, f"Courbe ROC ({model_name.replace('_', ' ')})",
        xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


def plot_pr_curve_final(y_test, scores, model_name):
    """Courbe Precision-Recall du modele final, plus pertinente que la ROC sur une classe rare."""
    precision, recall, _ = precision_recall_curve(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)
    baseline_rate = y_test.mean()

    fig, ax = plt.subplots(figsize=GRID_FIGSIZE)
    ax.plot(recall, precision, color="#16a34a", linewidth=2, label=f"PR-AUC = {pr_auc:.3f}")
    ax.axhline(
        baseline_rate, color="gray", linestyle="--", linewidth=1,
        label=f"Modele aleatoire ({baseline_rate:.1%})",
    )
    style_grid_axes(
        ax, f"Courbe Precision-Recall ({model_name.replace('_', ' ')})",
        xlabel="Recall", ylabel="Precision",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def plot_risk_gauge(probability, threshold):
    """Jauge visuelle du risque pour un client simule : plus parlant qu'un chiffre seul."""
    color = "#dc2626" if probability >= threshold else "#16a34a"
    fig, ax = plt.subplots(figsize=(GRID_FIGSIZE[0] * 1.6, 1.8))
    ax.barh([0], [1.0], color="#e5e7eb", height=0.5)
    ax.barh([0], [probability], color=color, height=0.5)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2)
    ax.text(
        threshold, 0.65, f"Seuil {threshold:.2f}", ha="center", va="bottom", fontsize=8,
    )
    ax.text(
        probability, 0, f" {probability:.1%}", va="center",
        fontsize=11, fontweight="bold", color=color,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Probabilite de churn", fontsize=9)
    ax.set_title("Niveau de risque de ce client", fontsize=10, pad=14)
    fig.tight_layout()
    return fig


def plot_client_shap_waterfall(model, scenario, top_n=8):
    """Explication locale : quelles variables poussent CE client vers le churn ou non."""
    if shap is None:
        return None

    estimator = model.named_steps.get("model")
    preprocessor = model.named_steps.get("pre")
    if estimator is None or preprocessor is None:
        return None

    transformed = preprocessor.transform(scenario)
    names = get_transformed_feature_names(model, scenario)

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    contributions = pd.Series(shap_values[0], index=names)
    top_names = contributions.abs().sort_values(ascending=False).head(top_n).index
    top_contributions = contributions[top_names].sort_values()
    colors = ["#dc2626" if v > 0 else "#16a34a" for v in top_contributions]

    fig, ax = plt.subplots(figsize=GRID_FIGSIZE)
    top_contributions.rename(index=format_feature_name).plot(kind="barh", ax=ax, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    style_grid_axes(
        ax, "Pourquoi ce client ? (facteurs SHAP)", xlabel="Impact sur la probabilite de churn"
    )
    fig.tight_layout()
    return fig


def plot_risk_distribution(scored, threshold):
    """Distribution du risque de churn sur le portefeuille filtre."""
    fig, ax = plt.subplots(figsize=GRID_FIGSIZE)
    ax.hist(scored["proba_churn"], bins=30, color="#2563eb", alpha=0.85)
    ax.axvline(threshold, color="#dc2626", linestyle="--", linewidth=1.2, label=f"Seuil {threshold:.2f}")
    style_grid_axes(
        ax, "Distribution du risque (portefeuille filtre)",
        xlabel="Probabilite de churn", ylabel="Nombre de clients",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_priority_matrix(scored, threshold):
    """Matrice de priorisation : risque de churn vs revenu mensuel a risque."""
    fig, ax = plt.subplots(figsize=GRID_FIGSIZE)
    colors = np.where(scored["client_a_risque"], "#dc2626", "#2563eb")
    ax.scatter(
        scored["proba_churn"], scored["revenu_a_risque"],
        c=colors, alpha=0.35, s=12, linewidths=0,
    )
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1)
    style_grid_axes(
        ax, "Matrice de priorisation (risque x revenu)",
        xlabel="Probabilite de churn", ylabel="Revenu mensuel a risque (EUR)",
    )
    fig.tight_layout()
    return fig


model, info, raw_data, comparison, thresholds, baseline = load_artifacts()

st.title("Pilotage du risque de churn client")
st.caption(
    "Tableau de bord pense pour un utilisateur non technique : chaque indicateur cle est "
    "affiche en chiffre (pas seulement par une couleur), les graphiques utilisent des couleurs "
    "contrastees et distinguables (bleu/orange/rouge/vert), et le texte reste en langage metier."
)

if model is None or info is None or raw_data is None:
    st.error("Artefacts manquants. Lance d'abord le notebook `04_entrainement_complet.ipynb`.")
    st.stop()

recommended_threshold = float(info.get("threshold_recommande", DEFAULT_THRESHOLD))
selected_model_name = info.get("modele_retenu", "modele retenu")
selected_strategy = info.get("strategie_retenue", "strategie non renseignee")

X_test = info["X_test"]
y_test = info["y_test"]

tab_business, tab_simulation, tab_explain = st.tabs(
    ["Vue metier", "Simulation client", "Explication"]
)

with tab_business:
    st.subheader("Indicateurs de retention")
    st.caption(
        f"Modele retenu : {selected_model_name}. Strategie : {selected_strategy}. "
        f"Seuil recommande : {recommended_threshold:.2f}."
    )

    # Le seuil est un choix metier : plus il baisse, plus on alerte de clients.
    threshold = st.slider(
        "Seuil d'alerte churn",
        min_value=0.10,
        max_value=0.90,
        value=recommended_threshold,
        step=0.05,
    )

    # Drill-down metier : la commission CRM cible souvent un sous-ensemble de clients.
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        segment_filter = st.multiselect(
            "Filtrer par segment client",
            options=sorted(raw_data["customer_segment"].unique()),
            default=sorted(raw_data["customer_segment"].unique()),
        )
    with filter_col2:
        contract_filter = st.multiselect(
            "Filtrer par type de contrat",
            options=sorted(raw_data["contract_type"].unique()),
            default=sorted(raw_data["contract_type"].unique()),
        )

    # On score tous les clients pour prioriser les actions CRM.
    scored = raw_data.copy()
    features = prepare_customer_features(scored)[info["all_cols"]]
    scored["proba_churn"] = predict_scores(model, features)
    scored["client_a_risque"] = scored["proba_churn"] >= threshold
    scored["revenu_a_risque"] = scored["monthly_fee"] * scored["proba_churn"]

    # Les KPI et tableaux ne portent que sur la selection courante (drill-down).
    scored = scored[
        scored["customer_segment"].isin(segment_filter)
        & scored["contract_type"].isin(contract_filter)
    ]

    risky = scored[scored["client_a_risque"]]
    churn_rate = scored["churn"].mean() if len(scored) else 0.0
    total_risk_revenue = scored["revenu_a_risque"].sum()
    metrics = model_metrics(model, X_test, y_test, threshold)

    # KPI de pilotage : portefeuille, churn observe, volume d'alertes et revenu a risque.
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Clients", f"{len(scored):,}".replace(",", " "))
    kpi2.metric("Churn observe", f"{churn_rate:.1%}")
    kpi3.metric("Clients a risque", f"{len(risky):,}".replace(",", " "))
    kpi4.metric("Revenu mensuel a risque", f"{total_risk_revenue:,.0f} EUR".replace(",", " "))

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Recall modele", f"{metrics['Recall']:.1%}")
    m2.metric("Precision", f"{metrics['Precision']:.1%}")
    m3.metric("F1-score", f"{metrics['F1-score']:.3f}")
    m4.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")
    m5.metric("PR-AUC", f"{metrics['PR-AUC']:.3f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Faux negatifs", metrics["Faux negatifs"])
    c2.metric("Faux positifs", metrics["Faux positifs"])
    c3.metric("Vrais positifs", metrics["Vrais positifs"])
    c4.metric("Vrais negatifs", metrics["Vrais negatifs"])

    st.markdown("#### Vue graphique du portefeuille")
    st.caption(
        "Combien de clients sont a risque (a gauche), et lesquels concentrent le plus "
        "de revenu a risque (a droite, points rouges = clients alertes)."
    )
    render_chart_grid(
        [plot_risk_distribution(scored, threshold), plot_priority_matrix(scored, threshold)],
        n_cols=2,
    )

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
        # Vue agregee pour identifier les segments les plus exposes.
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

    # La simulation applique le meme feature engineering que l'entrainement.
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

    st.markdown("#### Pourquoi cette prediction ?")
    gauge_col, shap_col = st.columns(2)
    with gauge_col:
        st.pyplot(plot_risk_gauge(probability, threshold), width="stretch")
    with shap_col:
        client_shap_fig = plot_client_shap_waterfall(model, scenario)
        if client_shap_fig is not None:
            st.pyplot(client_shap_fig, width="stretch")
        else:
            st.info("SHAP n'est pas disponible pour cet environnement ou ce modele.")

with tab_explain:
    st.subheader("Pourquoi le modele alerte certains clients ?")
    st.write(
        "Cette partie sert a expliquer les variables qui aident le plus le modele a detecter le churn. "
        "Les graphiques sont regroupes en grille pour rester lisibles a l'ecran."
    )

    sample_size = min(500, len(X_test))

    st.markdown("#### 1-3. Variables les plus importantes")
    st.write(
        "Trois methodes complementaires : l'importance native du modele (rapide, moins fiable), "
        "la permutation importance (perte de recall quand on melange une variable) et SHAP "
        "(impact moyen sur chaque prediction)."
    )

    # Permutation importance : on melange une variable et on observe la perte de recall.
    result = permutation_importance(
        model,
        X_test.iloc[:sample_size],
        y_test.iloc[:sample_size],
        scoring="recall",
        n_repeats=5,
        random_state=42,
    )
    permutation_top = (
        pd.Series(result.importances_mean, index=X_test.columns)
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )

    importance_figs = []
    native_fig = plot_native_feature_importance(model, X_test.iloc[:sample_size])
    if native_fig is not None:
        importance_figs.append(native_fig)
    importance_figs.append(plot_permutation_importance(permutation_top))
    shap_fig = plot_shap_importance(model, X_test.iloc[:sample_size])
    if shap_fig is not None:
        importance_figs.append(shap_fig)
    else:
        st.info("SHAP n'est pas disponible pour cet environnement ou ce modele.")

    render_chart_grid(importance_figs, n_cols=3)

    if comparison is not None:
        st.markdown("#### Rappel du choix du modele")
        comparison_cols = [
            "modele",
            "strategie_desequilibre",
            "test_recall",
            "test_precision",
            "test_f1",
            "test_roc_auc",
            "test_pr_auc",
            "test_fp",
            "test_fn",
            "temps_entrainement_secondes",
        ]
        comparison_cols = [col for col in comparison_cols if col in comparison.columns]
        st.dataframe(
            comparison[comparison_cols]
            .sort_values("test_recall", ascending=False)
            .style.format(
                {
                    "test_recall": "{:.3f}",
                    "test_precision": "{:.3f}",
                    "test_f1": "{:.3f}",
                    "test_roc_auc": "{:.3f}",
                    "test_pr_auc": "{:.3f}",
                    "temps_entrainement_secondes": "{:.2f} s",
                }
            ),
            use_container_width=True,
        )

        st.markdown("#### Courbes ROC et Precision-Recall (modele final)")
        st.write(
            "Les KPI n'affichent que le chiffre de ROC-AUC et PR-AUC : voici les courbes "
            "completes derriere ces deux chiffres, pour le modele retenu."
        )
        final_scores = predict_scores(model, X_test)
        curve_figs = [
            plot_roc_curve_final(y_test, final_scores, selected_model_name),
            plot_pr_curve_final(y_test, final_scores, selected_model_name),
        ]
        render_chart_grid(curve_figs, n_cols=2)

        st.markdown("#### Graphiques de choix du modele")
        st.write(
            "Ces graphiques reprennent l'analyse du notebook `05_modelisation_evaluation.ipynb`, "
            "y compris le cout de calcul (ecoresponsabilite) : XGBoost obtient le meilleur recall "
            "pour un temps d'entrainement nettement plus faible que le Random Forest ou le MLP."
        )

        comparison_figs = []
        if baseline is not None:
            comparison_figs.append(plot_baseline_metrics(baseline))
        comparison_figs.append(plot_model_recall(comparison))
        comparison_figs.append(plot_model_prauc(comparison))
        comparison_figs.append(plot_metric_comparison(comparison))
        comparison_figs.append(plot_error_bars(comparison))
        comparison_figs.append(plot_radar(comparison))
        if "temps_entrainement_secondes" in comparison.columns:
            comparison_figs.append(plot_training_time(comparison))

        render_chart_grid(comparison_figs, n_cols=3)

        if thresholds is not None:
            st.markdown("#### Impact du seuil d'alerte")
            threshold_figs = [
                plot_threshold_scores(thresholds, selected_model_name, recommended_threshold),
                plot_threshold_errors(thresholds, selected_model_name, recommended_threshold),
            ]
            render_chart_grid(threshold_figs, n_cols=2)

    st.caption(
        "Ces resultats montrent des associations apprises par le modele. Ils ne prouvent pas une causalite."
    )
