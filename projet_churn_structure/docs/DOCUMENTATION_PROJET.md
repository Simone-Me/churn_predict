# Documentation du projet churn client

## 1. Contexte

Le projet consiste a predire le churn client, c'est-a-dire la probabilite qu'un client resilie son abonnement.

Dans un contexte CRM, ce modele sert a identifier les clients a risque, prioriser les actions de retention, reduire la perte de revenu et aider les equipes marketing a prendre une decision.

## 2. Dataset et desequilibre

Le dataset contient `10 000` clients. La cible est `churn`.

Repartition observee :

| Classe | Signification | Effectif | Proportion |
|---|---|---:|---:|
| 0 | client qui reste | 8 979 | 89.79 % |
| 1 | client qui resilie | 1 021 | 10.21 % |

Le ratio majoritaire/minoritaire est donc `8.79`. Ce desequilibre oblige a comparer plusieurs strategies de reequilibrage.

## 3. Pourquoi l'accuracy ne suffit pas

Une baseline qui predit toujours `pas de churn` obtient `0.898` d'accuracy, mais `0.000` de recall. Elle ne detecte aucun client perdu.

Une regression logistique sans reequilibrage au seuil `0.5` obtient `0.896` d'accuracy, mais seulement `0.059` de recall avec `192` faux negatifs.

Dans ce contexte, l'accuracy est trompeuse. Les metriques suivies sont :

- recall : prioritaire, car il mesure la detection des vrais churners ;
- precision : utile pour controler le nombre d'alertes inutiles ;
- F1-score : compromis entre precision et recall ;
- ROC-AUC : capacite generale a classer les clients ;
- PR-AUC : fortement pertinente car la classe churn est rare.

## 4. Preparation des donnees

La preparation est faite dans `feature_engineering.py`.

Variables metier ajoutees :

- `has_complaint` ;
- `payment_risk` ;
- `low_satisfaction` ;
- `inactive_customer` ;
- `monthly_contract` ;
- `tickets_per_tenure` ;
- `fee_per_login` ;
- `support_pressure` ;
- `engagement_score` ;
- `satisfaction_score`.

Ces variables representent des signaux CRM lisibles : activite, satisfaction, support, paiement, contrat et revenu.

## 5. Strategies testees

Le notebook `04_entrainement_complet.ipynb` compare plusieurs approches.

Approches data-level :

- `RandomOverSampler` : duplique la classe minoritaire, mais peut favoriser l'overfitting ;
- `SMOTE` : cree des exemples synthetiques, mais peut ajouter du bruit ;
- `RandomUnderSampler` : reduit la classe majoritaire, mais peut perdre de l'information.

Approches model-level :

- `class_weight="balanced"` pour Logistic Regression ;
- `class_weight="balanced_subsample"` pour Random Forest ;
- `scale_pos_weight` pour XGBoost.

La validation croisee utilise `StratifiedKFold` pour garder la meme proportion de churners dans chaque fold.

## 6. Resultats principaux

Au seuil par defaut `0.5`, XGBoost avec `scale_pos_weight` obtient deja le meilleur compromis global :

| Modele | Strategie | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| XGBoost | scale_pos_weight | 0.696 | 0.267 | 0.386 | 0.797 | 0.282 |
| Logistic Regression | class_weight | 0.672 | 0.197 | 0.305 | 0.750 | 0.251 |
| Random Forest | class_weight | 0.377 | 0.316 | 0.344 | 0.798 | 0.262 |

## 7. Ajustement du seuil

Le seuil `0.5` n'est pas optimal avec une classe positive rare. Le notebook teste des seuils de `0.10` a `0.90`.

Le seuil final choisi est `0.20` pour XGBoost avec `scale_pos_weight`.

| Seuil | Recall | Precision | F1 | Faux positifs | Faux negatifs | Vrais positifs |
|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.882 | 0.211 | 0.340 | 674 | 24 | 180 |

Ce seuil reduit fortement les faux negatifs. Le prix a payer est un volume plus eleve de faux positifs, acceptable si les actions de retention restent peu couteuses.

## 8. Choix final

Le modele retenu est **XGBoost avec `scale_pos_weight`**, seuil `0.20`.

Raisons :

- meilleur compromis recall/precision sous contrainte metier ;
- tres bonne detection des churners : `180` sur `204` dans le test set ;
- PR-AUC et ROC-AUC solides pour un dataset desequilibre ;
- conservation de toutes les donnees, contrairement a l'under-sampling.

## 9. Dashboard

Le dashboard Streamlit utilise `best_model.pkl` et le seuil recommande stocke dans `data_preprocessed.pkl`.

Il permet de :

- afficher les KPI principaux ;
- voir les clients a risque ;
- estimer le revenu mensuel a risque ;
- simuler un client ;
- comparer les strategies ;
- afficher le recall, la precision, le F1-score, la ROC-AUC, la PR-AUC et les faux positifs/faux negatifs.

Commande :

```powershell
streamlit run app.py
```
