# Projet de prediction du churn client

Ce projet repond a la consigne EFREI : predire le churn client, traiter le desequilibre de classes, comparer plusieurs strategies de modelisation et proposer un dashboard decisionnel.

## Objectif

L'objectif est d'aider une equipe marketing/CRM a reperer les clients qui risquent de resilier leur abonnement.

La cible est `churn` :

- `0` : le client reste ;
- `1` : le client resilie.

La cible est desequilibree : le dataset contient `8 979` non-churners et `1 021` churners, soit un ratio majoritaire/minoritaire de `8.79`. Le **recall** est donc prioritaire, car rater un churner est plus couteux que contacter un client en trop.

## Structure

```text
projet_churn_structure/
  app.py
  feature_engineering.py
  README.md
  requirements.txt
  data/
    customer_churn.csv
  docs/
    consigne_projet.pdf
    DOCUMENTATION_PROJET.md
    CYCLE_DE_VIE_PROJET.md
    TEXTE_A_DIRE.md
    presentation_churn_notes.pptx
  models/
    best_model.pkl
    model_*.pkl
  notebooks/
    01_EDA.ipynb
    02_preparation_donnees.ipynb
    03_modelisation_evaluation.ipynb
    04_dashboard.ipynb
    05_feature_engineering_details.ipynb
    06_entrainement_complet.ipynb
    07_application_metier.ipynb
  reports/
    baseline_analysis.csv
    model_comparison.csv
    threshold_analysis.csv
```

## Lancer le projet

```powershell
pip install -r requirements.txt
jupyter notebook notebooks/06_entrainement_complet.ipynb
streamlit run app.py
```

Le notebook `06_entrainement_complet.ipynb` regenere les modeles, les rapports et le fichier `data_preprocessed.pkl`.

## Analyse du desequilibre

Une baseline majoritaire obtient `0.898` d'accuracy, mais `0.000` de recall : elle ne detecte aucun churner. Une regression logistique sans reequilibrage au seuil `0.5` obtient aussi une accuracy elevee (`0.896`), mais seulement `0.059` de recall.

C'est pour cela que le projet utilise des metriques adaptees :

- recall : detecter les clients qui vont vraiment churner ;
- precision : eviter trop d'alertes inutiles ;
- F1-score : compromis precision/recall ;
- ROC-AUC : qualite globale du classement ;
- PR-AUC : metrique fortement utile quand la classe positive est rare.

## Strategies comparees

Le notebook final compare :

- baseline sans reequilibrage ;
- `RandomOverSampler` ;
- `SMOTE` ;
- `RandomUnderSampler` ;
- `class_weight="balanced"` pour Logistic Regression ;
- `class_weight="balanced_subsample"` pour Random Forest ;
- `scale_pos_weight` pour XGBoost ;
- MLP avec SMOTE.

Toutes les validations croisees utilisent `StratifiedKFold`, afin de preserver la proportion de churners dans chaque fold.

## Resultat final

Le modele retenu est **XGBoost avec `scale_pos_weight`**, au seuil recommande `0.20`.

| Modele final | Strategie | Seuil | Recall | Precision | F1 | ROC-AUC | PR-AUC | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | scale_pos_weight | 0.20 | 0.882 | 0.211 | 0.340 | 0.797 | 0.282 | 674 | 24 |

Ce choix detecte `180` churners sur `204` dans le test set. Le compromis accepte plus de faux positifs pour reduire les faux negatifs, ce qui est coherent avec un contexte CRM ou une action de retention coute moins cher qu'un client perdu.

## Idee principale

Le gain vient de la combinaison suivante :

- variables metier explicables ;
- validation stratifiee ;
- comparaison explicite des techniques de desequilibre ;
- choix de metriques adaptees ;
- ajustement du seuil de decision.
