# Documentation du projet churn client

## 1. Contexte

Le projet consiste a predire le churn client, c'est-a-dire la probabilite qu'un client resilie son abonnement.

Dans un contexte CRM, ce type de modele sert a :

- identifier les clients a risque ;
- prioriser les actions de retention ;
- reduire la perte de revenu ;
- aider les equipes marketing a prendre une decision.

## 2. Dataset

Le dataset contient 10 000 clients.

Il contient plusieurs familles de variables :

- profil client : age, genre, pays, ville, segment ;
- contrat : type de contrat, anciennete, frais mensuels ;
- usage : connexions, jours actifs, temps de session, fonctionnalites utilisees ;
- paiement : moyen de paiement, echecs de paiement, hausse de prix ;
- support : tickets, temps de resolution, type de plainte ;
- satisfaction : CSAT, NPS, reponse au sondage ;
- cible : `churn`.

## 3. Pourquoi le recall est prioritaire

La classe `churn = 1` est minoritaire : environ 10 % des clients.

Dans ce cas, l'accuracy peut etre trompeuse. Un modele qui predit toujours "pas de churn" aurait deja une bonne accuracy, mais il serait inutile.

Le recall est plus adapte parce qu'il repond a la question :

> Parmi les clients qui churnent vraiment, combien le modele arrive-t-il a detecter ?

Pour ce projet, manquer un client a risque est considere plus grave que contacter un client qui ne churnera pas.

## 4. Preparation des donnees

La preparation est faite dans `feature_engineering.py`.

L'idee est de creer des variables simples et metier.

### Variables ajoutees

`has_complaint`

Indique si le client a deja eu une plainte. Une plainte peut signaler une insatisfaction.

`payment_risk`

Indique s'il y a eu un echec de paiement ou une hausse de prix recente. Ces evenements peuvent augmenter le risque de churn.

`low_satisfaction`

Indique si le client a un NPS negatif, un CSAT faible ou une reponse insatisfaite.

`inactive_customer`

Indique si le client utilise peu le service ou ne s'est pas connecte recemment.

`monthly_contract`

Indique si le client a un contrat mensuel. Un contrat mensuel est souvent plus facile a resilier.

`tickets_per_tenure`

Nombre de tickets support rapporte a l'anciennete du client.

`fee_per_login`

Frais mensuels rapportes au nombre de connexions. Cela donne une idee du cout percu par rapport a l'utilisation.

`support_pressure`

Combine le nombre de tickets et le temps de resolution. Cela represente la pression support.

`engagement_score`

Score simple qui resume l'activite client : connexions, jours actifs, fonctionnalites utilisees, croissance d'usage et recence.

`satisfaction_score`

Score simple qui resume la satisfaction : CSAT, NPS, escalades et plaintes.

## 5. Modeles

Quatre modeles sont entraines dans le notebook `06_entrainement_complet.ipynb`.

### Logistic Regression

Modele simple et interpretable. Il sert de baseline solide.

Dans ce projet, il donne le meilleur recall.

### Random Forest

Modele d'arbres qui capture des relations non lineaires.

Il donne un bon F1-score et une meilleure precision que la regression logistique.

### XGBoost

Modele de gradient boosting performant sur donnees tabulaires.

Il donne un bon compromis entre recall, precision et ROC-AUC.

### MLP

Modele de reseau de neurones simple.

Il est inclus pour respecter l'exigence d'un modele Deep Learning.

## 6. Resultats

Le seuil de decision est garde a `0.35`.

| Modele | Recall | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.882 | 0.165 | 0.278 | 0.750 |
| XGBoost | 0.843 | 0.215 | 0.343 | 0.784 |
| Random Forest | 0.809 | 0.259 | 0.392 | 0.798 |
| Deep Learning MLP | 0.353 | 0.257 | 0.298 | 0.699 |

## 7. Choix final

Le modele retenu est la Logistic Regression.

Raisons :

- meilleur recall ;
- modele simple a expliquer ;
- bon choix quand l'objectif est de detecter un maximum de clients a risque ;
- plus transparent qu'un modele complexe.

Limite :

- la precision est faible ;
- cela veut dire que certains clients alertes ne churneront pas vraiment.

Cette limite est acceptable si le cout d'une action de retention est plus faible que le cout d'un client perdu.

## 8. Dashboard

Le dashboard Streamlit permet de :

- afficher les KPI principaux ;
- voir les clients a risque ;
- estimer le revenu mensuel a risque ;
- simuler un client ;
- comparer les modeles ;
- afficher les variables importantes.

Commande :

```powershell
streamlit run app.py
```

## 9. Ce qu'il faut retenir pour la presentation

La logique du projet est :

1. Comprendre le desequilibre du churn.
2. Choisir le recall comme metrique principale.
3. Creer des variables metier simples.
4. Comparer quatre modeles.
5. Retenir le modele avec le meilleur recall.
6. Mettre le resultat dans un dashboard utilisable.
