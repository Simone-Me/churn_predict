# Projet de prediction du churn client

Ce projet repond a la consigne EFREI : predire le churn client, comparer plusieurs modeles et proposer un dashboard decisionnel.

## Objectif

L'objectif est d'aider une equipe marketing/CRM a reperer les clients qui risquent de resilier leur abonnement.

La cible est `churn` :

- `0` : le client reste ;
- `1` : le client resilie.

Comme les churners sont minoritaires, la metrique prioritaire est le **recall**.

## Structure

```text
churn_predict/
  app.py
  feature_engineering.py
  README.md
  ETAT_DES_LIEUX.md
  requirements.txt
  data/
    customer_churn.csv
  docs/
    consigne_projet.pdf
    DOCUMENTATION_PROJET.md
    CYCLE_DE_VIE_PROJET.md
    TEXTE_A_DIRE.md
    SLIDE_NOTES.txt
    presentation_churn_notes.pptx
  models/
    best_model.pkl
    model_LogisticRegression.pkl
    model_RandomForest.pkl
    model_XGBoost.pkl
    model_DeepLearning.pkl
  notebooks/
    01_EDA.ipynb
    02_preparation_donnees.ipynb
    03_modelisation_evaluation.ipynb
    04_dashboard.ipynb
    05_feature_engineering_details.ipynb
    06_entrainement_complet.ipynb
    07_application_metier.ipynb
  reports/
    model_comparison.csv
    threshold_analysis.csv
```

## Lancer le projet

```powershell
pip install -r requirements.txt
jupyter notebook notebooks/06_entrainement_complet.ipynb
streamlit run app.py
```

Le dashboard s'ouvre generalement sur :

```text
http://localhost:8501
```

## Notebooks

Les notebooks sont prevus pour etre executes et compris par vous deux :

- `01_EDA.ipynb` : analyse exploratoire ;
- `02_preparation_donnees.ipynb` : preparation des donnees ;
- `03_modelisation_evaluation.ipynb` : comparaison des modeles ;
- `04_dashboard.ipynb` : utilisation du dashboard ;
- `05_feature_engineering_details.ipynb` : detail des variables metier ;
- `06_entrainement_complet.ipynb` : execution du pipeline final ;
- `07_application_metier.ipynb` : explication de l'application finale.

Le notebook `06_entrainement_complet.ipynb` regenere les modeles, les rapports et le fichier `data_preprocessed.pkl`.

Les fichiers `.py` restants sont seulement ceux necessaires au dashboard ou aux fonctions partagees.

## Choix du modele

Le projet compare 4 modeles :

- Logistic Regression ;
- Random Forest ;
- XGBoost ;
- Deep Learning MLP.

Le modele retenu est la **Logistic Regression**, car elle obtient le meilleur recall au seuil fixe `0.35`.

Dans le dashboard final, l'utilisateur metier ne choisit pas le modele. L'application utilise directement le modele retenu, car le but est d'aider a prioriser les clients, pas de comparer les algorithmes.

## Resultat principal

| Modele | Recall | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.882 | 0.165 | 0.278 | 0.750 |
| XGBoost | 0.843 | 0.215 | 0.343 | 0.784 |
| Random Forest | 0.809 | 0.259 | 0.392 | 0.798 |
| Deep Learning MLP | 0.353 | 0.257 | 0.298 | 0.699 |

## Idee principale

Le gain vient surtout de la preparation des donnees :

- activite client ;
- satisfaction ;
- pression support ;
- risque paiement ;
- type de contrat ;
- anciennete et revenu.

Ces variables sont simples a expliquer et correspondent a une logique metier CRM.
