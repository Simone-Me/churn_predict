# Cycle de vie du projet

## 1. Probleme metier

Le but est d'aider une equipe CRM a identifier les clients qui risquent de resilier leur abonnement, afin d'agir avant le depart.

La variable cible est `churn` :

- `0` : le client reste ;
- `1` : le client part.

## 2. Exploration des donnees

L'EDA montre un desequilibre important :

- `8 979` clients restent ;
- `1 021` clients churnent ;
- le ratio majoritaire/minoritaire est `8.79`.

Environ `10.21 %` des clients churnent. Cette proportion change la maniere d'evaluer les modeles.

## 3. Accuracy insuffisante

Avec ce dataset, une baseline qui predit toujours la classe majoritaire obtient `0.898` d'accuracy, mais `0.000` de recall.

Cela veut dire qu'elle ne detecte aucun churner. Pour le metier, ce modele est inutile.

Le projet suit donc surtout :

- le recall, pour minimiser les faux negatifs ;
- la precision, pour surveiller les alertes inutiles ;
- le F1-score, pour mesurer le compromis ;
- la ROC-AUC et la PR-AUC, pour evaluer la qualite du scoring.

La PR-AUC est particulierement utile parce que la classe positive est rare.

## 4. Preparation et feature engineering

Les donnees sont preparees dans `feature_engineering.py`.

Nous creons des variables metier faciles a expliquer :

- plainte client ;
- risque paiement ;
- faible satisfaction ;
- inactivite ;
- contrat mensuel ;
- pression support ;
- engagement ;
- satisfaction globale.

Ces variables traduisent des signaux concrets que les equipes CRM peuvent comprendre.

## 5. Validation stratifiee

Le split train/test est stratifie, et la validation croisee utilise `StratifiedKFold`.

Ce choix preserve les proportions de classes dans chaque fold. C'est important car, sans stratification, certains folds pourraient contenir trop peu de churners et rendre le recall instable.

## 6. Gestion du desequilibre

Plusieurs techniques sont testees.

Data-level :

- `RandomOverSampler` ;
- `SMOTE` ;
- `RandomUnderSampler`.

Model-level :

- `class_weight="balanced"` ;
- `class_weight="balanced_subsample"` ;
- `scale_pos_weight` pour XGBoost.

Les effets attendus sont differents : l'over-sampling peut overfitter, SMOTE peut ajouter du bruit, l'under-sampling peut perdre de l'information, et la ponderation garde toutes les observations.

## 7. Entrainement et comparaison

Le notebook `06_entrainement_complet.ipynb` entraine les experiences et regenere :

- `reports/baseline_analysis.csv` ;
- `reports/model_comparison.csv` ;
- `reports/threshold_analysis.csv` ;
- les modeles dans `models/` ;
- `data_preprocessed.pkl`.

Au seuil `0.5`, XGBoost avec `scale_pos_weight` donne un bon compromis :

- recall : `0.696` ;
- precision : `0.267` ;
- F1-score : `0.386` ;
- ROC-AUC : `0.797` ;
- PR-AUC : `0.282`.

## 8. Optimisation du seuil

Le seuil de classification par defaut est `0.5`. Dans un probleme desequilibre, il n'est pas forcement optimal.

Le projet teste des seuils de `0.10` a `0.90`.

Le seuil retenu est `0.20` pour XGBoost avec `scale_pos_weight` :

- recall : `0.882` ;
- precision : `0.211` ;
- F1-score : `0.340` ;
- faux positifs : `674` ;
- faux negatifs : `24` ;
- vrais positifs : `180`.

Le seuil plus bas detecte beaucoup plus de churners, au prix d'un nombre plus eleve de clients alertes.

## 9. Choix final

Le modele final est **XGBoost avec `scale_pos_weight` au seuil `0.20`**.

Ce choix est pertinent en contexte metier car il detecte la majorite des clients qui vont partir. Le compromis precision/recall est acceptable si le cout d'un contact de retention est inferieur au cout d'un client perdu.

## 10. Dashboard

Le dashboard Streamlit transforme le modele en outil decisionnel :

- KPI metier ;
- clients a prioriser ;
- revenu mensuel a risque ;
- simulation client ;
- comparaison des strategies ;
- affichage des erreurs FP/FN.

## 11. Conclusion

Le projet montre qu'un bon modele de churn ne se choisit pas seulement avec l'accuracy. Il faut adapter les metriques, traiter le desequilibre, utiliser une validation stratifiee et ajuster le seuil en fonction du besoin metier.
