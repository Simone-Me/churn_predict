# Cycle de vie du projet

Ce document explique le projet comme une histoire logique, de l'idee de depart jusqu'au dashboard final.

## 1. Comprendre le probleme metier

Au depart, la consigne demandait de construire une solution autour de la retention client.

Le probleme metier est simple :

- une entreprise veut savoir quels clients risquent de partir ;
- elle veut agir avant que le client resilie ;
- elle veut prioriser les actions de retention ;
- elle veut limiter la perte de chiffre d'affaires.

Nous avons donc choisi la tache la plus naturelle du dataset :

> Predire le churn client, c'est-a-dire savoir si un client risque de resilier son abonnement.

La variable cible est `churn` :

- `0` : le client reste ;
- `1` : le client part.

## 2. Explorer les donnees

La premiere etape a ete l'EDA, c'est-a-dire l'analyse exploratoire.

Nous avons regarde :

- la taille du dataset ;
- les colonnes disponibles ;
- les valeurs manquantes ;
- la repartition de la cible `churn` ;
- les liens possibles entre churn et contrat, support, paiement, satisfaction, activite.

Le premier constat important a ete le desequilibre de la cible :

> Il y a beaucoup plus de clients qui restent que de clients qui partent.

Environ 10 % des clients churnent. Cela change completement la maniere d'evaluer les modeles.

## 3. Comprendre que l'accuracy ne suffit pas

Au debut, on pourrait penser qu'un modele avec une bonne accuracy est un bon modele.

Mais avec un dataset desequilibre, ce n'est pas suffisant.

Exemple :

Si 90 % des clients restent, un modele qui predit toujours "le client reste" peut avoir environ 90 % d'accuracy, mais il ne detecte aucun client qui part.

Pour notre sujet, ce n'est pas utile.

Nous avons donc choisi de regarder surtout le **recall**.

Le recall repond a la question :

> Parmi les clients qui partent vraiment, combien le modele arrive-t-il a detecter ?

Comme le but est de ne pas rater les clients a risque, le recall est la metrique la plus importante pour nous.

## 4. Nettoyer et preparer les donnees

Ensuite, nous avons prepare les donnees pour les modeles.

Nous avons :

- remplace les valeurs manquantes de `complaint_type` par "Aucune Plainte" ;
- retire `customer_id`, car c'est juste un identifiant ;
- separe les variables explicatives `X` et la cible `y` ;
- fait un split train/test stratifie pour garder la meme proportion de churners dans train et test ;
- encode les variables categorielles ;
- standardise les variables numeriques pour les modeles qui en ont besoin.

Le split stratifie est important car il evite d'avoir un jeu d'entrainement ou de test avec une proportion de churn trop differente.

## 5. Faire du feature engineering

Nous nous sommes rendu compte que les variables brutes ne suffisaient pas toujours.

Par exemple, une plainte seule ne raconte pas toute l'histoire. Il est plus parlant de savoir :

- si le client a deja eu une plainte ;
- s'il est peu actif ;
- s'il a eu des problemes de paiement ;
- s'il a une faible satisfaction ;
- s'il utilise peu le service par rapport a ce qu'il paie.

Nous avons donc cree des variables metier simples dans `feature_engineering.py` :

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

Ces variables sont faciles a expliquer car elles correspondent a une logique CRM.

## 6. Choisir plusieurs modeles

La consigne demandait de comparer plusieurs modeles, dont un modele de Deep Learning.

Nous avons choisi quatre modeles :

### Logistic Regression

Modele simple, interpretable et utile comme baseline.

Il est interessant car on peut expliquer facilement pourquoi il sert de reference.

### Random Forest

Modele base sur plusieurs arbres de decision.

Il est robuste, fonctionne bien avec des donnees tabulaires et gere naturellement des relations non lineaires.

### XGBoost

Modele de boosting performant sur les donnees tabulaires.

Il corrige progressivement ses erreurs et donne souvent de bons resultats.

### MLP

Petit reseau de neurones.

Il permet de respecter la demande d'inclure un modele Deep Learning, meme si ce n'est pas forcement le meilleur choix pour ce dataset.

## 7. Adapter le preprocessing aux modeles

Nous avons aussi compris que tous les modeles n'ont pas les memes besoins.

Pour Logistic Regression et MLP :

- les variables numeriques doivent etre standardisees ;
- ces modeles sont sensibles aux echelles des variables.

Pour Random Forest et XGBoost :

- la standardisation n'est pas indispensable ;
- les arbres fonctionnent avec des seuils et sont moins sensibles aux echelles.

C'est pour cela que les pipelines ne sont pas exactement identiques selon les modeles.

## 8. Gerer le desequilibre

Comme les churners sont minoritaires, il fallait eviter que les modeles apprennent surtout la classe majoritaire.

Nous avons utilise une ponderation de classe pour certains modeles :

- `class_weight='balanced'` pour Logistic Regression ;
- `class_weight='balanced_subsample'` pour Random Forest ;
- `scale_pos_weight` pour XGBoost.

L'idee est de dire au modele :

> Les erreurs sur les clients qui churnent coutent plus cher, donc il faut leur donner plus d'importance pendant l'entrainement.

Nous avons prefere cette approche car elle reste simple a expliquer.

## 9. Evaluer les modeles

Nous avons compare les modeles avec plusieurs metriques :

- accuracy ;
- precision ;
- recall ;
- F1-score ;
- ROC-AUC.

Mais pour choisir le modele final, nous avons surtout regarde le recall.

Resultats principaux au seuil fixe 0.35 :

| Modele | Recall | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.882 | 0.165 | 0.278 | 0.750 |
| XGBoost | 0.843 | 0.215 | 0.343 | 0.784 |
| Random Forest | 0.809 | 0.259 | 0.392 | 0.798 |
| MLP | 0.353 | 0.257 | 0.298 | 0.699 |

Le meilleur recall est obtenu par la Logistic Regression.

## 10. Choisir le modele final

Nous avons retenu la Logistic Regression.

Pourquoi ?

- elle a le meilleur recall ;
- elle est simple a expliquer ;
- elle correspond bien a l'objectif metier : detecter le plus possible de clients a risque ;
- elle sert de modele clair pour une presentation.

La limite est sa precision faible.

Cela veut dire que certains clients alertes ne partiront pas vraiment.

Mais dans un contexte de retention, ce compromis peut etre acceptable :

> Il vaut mieux contacter quelques clients en trop que rater beaucoup de clients qui allaient partir.

## 11. Construire le dashboard

La derniere etape a ete de transformer le travail de modelisation en outil utilisable.

Le dashboard Streamlit permet de :

- voir les indicateurs principaux ;
- estimer le nombre de clients a risque ;
- calculer un revenu mensuel a risque ;
- simuler un profil client ;
- comparer les modeles ;
- afficher les variables influentes.

L'objectif n'est pas seulement d'avoir un modele, mais d'avoir une aide a la decision.

## 12. Ce qu'on a appris en avançant

Au debut, le projet semblait etre seulement un probleme de prediction.

En avançant, nous avons compris que les points importants etaient :

- comprendre le desequilibre de la cible ;
- choisir une metrique adaptee ;
- preparer les donnees selon les modeles ;
- creer des variables metier utiles ;
- comparer plusieurs modeles ;
- expliquer les resultats simplement ;
- transformer le modele en outil decisionnel.

## 13. Limites et ameliorations possibles

Le projet peut encore etre ameliore.

Pistes possibles :

- analyser plus precisement les faux negatifs ;
- tester d'autres variables metier ;
- optimiser les hyperparametres ;
- tester RobustScaler pour les modeles sensibles aux outliers ;
- ajouter SHAP pour expliquer les predictions plus finement.

## 14. Phrase de conclusion

Nous avons construit un cycle complet de projet Data Science :

> comprehension du probleme, exploration des donnees, preparation, feature engineering, entrainement, evaluation, choix du modele et dashboard.

Le projet montre que la meilleure solution n'est pas seulement celle qui a la meilleure accuracy, mais celle qui repond le mieux au besoin metier : detecter les clients a risque pour agir avant leur depart.
