# Texte a dire pendant la presentation

## Slide 1 - Titre

Bonjour, nous allons presenter notre projet de prediction du churn client. L'objectif est d'aider une equipe CRM a identifier les clients qui risquent de resilier leur abonnement, afin de prioriser les actions de retention.

## Slide 2 - Probleme metier

Dans une entreprise par abonnement, perdre un client coute cher. Le but n'est pas seulement de predire une classe, mais d'aider les equipes metier a savoir quels clients contacter en priorite.

## Slide 3 - Dataset

Nous avons utilise un dataset de 10 000 clients. Il contient des informations sur le profil client, le contrat, l'usage du service, les paiements, le support client et la satisfaction. La variable cible est `churn`, avec 0 pour non churn et 1 pour churn.

## Slide 4 - Desequilibre de la cible

Le point important est que la cible est desequilibree : environ 10 % des clients churnent. Donc l'accuracy seule n'est pas suffisante. Un modele pourrait avoir une bonne accuracy tout en ratant la majorite des churners.

## Slide 5 - Cycle de vie du projet

Nous avons suivi un cycle classique de projet Data Science. D'abord, nous avons compris le probleme metier. Ensuite, nous avons explore les donnees, prepare les variables, entraine plusieurs modeles, compare les resultats, puis transforme le modele en dashboard.

## Slide 6 - Ce qu'on a constate

Pendant l'EDA, nous avons vu que le dataset etait desequilibre. Nous avons aussi remarque que les signaux etaient repartis sur plusieurs familles de variables : activite, satisfaction, paiement et support. Donc il fallait creer des variables metier plus lisibles.

## Slide 7 - Metrique principale

Nous avons choisi le recall comme metrique principale. Le recall repond a la question : parmi les clients qui churnent vraiment, combien sont detectes par le modele ? Dans notre cas, rater un client a risque est plus grave que contacter un client en trop.

## Slide 8 - Preparation des donnees

Pour ameliorer les modeles, nous avons cree des variables metier simples. Par exemple, un indicateur de plainte, un indicateur de risque paiement, un indicateur de faible satisfaction, un indicateur d'inactivite, et des scores d'engagement et de satisfaction.

## Slide 9 - Pourquoi ces variables

Ces variables sont logiques dans un contexte CRM. Un client peu actif, insatisfait, qui a eu des echecs de paiement ou beaucoup de tickets support a plus de chances de churner. L'objectif etait de rendre les donnees plus lisibles pour les modeles.

## Slide 10 - Modeles compares

Nous avons compare quatre modeles : Logistic Regression, Random Forest, XGBoost et un MLP. Cela permet de comparer un modele simple, des modeles d'arbres, un modele boosting, et un modele de type Deep Learning.

## Slide 11 - Resultats

Au seuil fixe de 0.35, la Logistic Regression obtient le meilleur recall avec 88,2 %. XGBoost et Random Forest ont une meilleure precision ou un meilleur F1-score, mais detectent un peu moins de churners.

## Slide 12 - Choix final

Nous retenons la Logistic Regression, car notre objectif principal est le recall. Elle est aussi simple a expliquer et plus interpretable. La limite est que la precision reste faible, donc certains clients alertes ne churneront pas. Mais dans un contexte retention, cela peut etre acceptable.

## Slide 13 - Dashboard

Nous avons construit un dashboard Streamlit. Il permet de voir les KPI, les clients a risque, le revenu mensuel a risque, de simuler un client et de comparer les modeles. L'objectif est de transformer le modele en outil de decision.

## Slide 14 - Conclusion

Pour conclure, notre projet va de l'analyse des donnees jusqu'au dashboard. La partie la plus importante a ete la preparation des donnees et le choix d'une metrique adaptee. Le modele final detecte environ 88 % des churners, ce qui correspond bien a notre objectif de retention.
