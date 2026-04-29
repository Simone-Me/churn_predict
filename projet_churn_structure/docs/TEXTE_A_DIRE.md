# Texte a dire pendant la presentation

## Slide 1 - Titre

Bonjour, nous allons presenter notre projet de prediction du churn client. L'objectif est d'aider une equipe CRM a identifier les clients qui risquent de resilier leur abonnement, afin de prioriser les actions de retention.

## Slide 2 - Probleme metier

Dans une entreprise par abonnement, perdre un client coute cher. Le but n'est pas seulement de predire une classe, mais d'aider les equipes metier a savoir quels clients contacter en priorite.

## Slide 3 - Dataset

Nous avons utilise un dataset de 10 000 clients. La variable cible est `churn`, avec 0 pour non churn et 1 pour churn.

## Slide 4 - Desequilibre de la cible

La cible est desequilibree : 8 979 clients restent et 1 021 clients churnent. Le ratio majoritaire/minoritaire est donc de 8,79. Cela veut dire que l'accuracy seule peut etre trompeuse.

## Slide 5 - Pourquoi l'accuracy ne suffit pas

Une baseline qui predit toujours "pas de churn" obtient environ 89,8 % d'accuracy, mais elle detecte 0 churner. Donc elle est inutile pour la retention.

## Slide 6 - Metriques choisies

Nous avons choisi le recall comme metrique principale, car il mesure combien de vrais churners sont detectes. Nous suivons aussi la precision, le F1-score, la ROC-AUC et la PR-AUC. La PR-AUC est importante ici parce que la classe churn est rare.

## Slide 7 - Preparation des donnees

Nous avons cree des variables metier simples : risque paiement, faible satisfaction, inactivite, plainte, pression support, engagement et satisfaction globale.

## Slide 8 - Techniques de desequilibre

Nous avons compare plusieurs techniques : Random Over-Sampling, SMOTE, Random Under-Sampling, class weights et `scale_pos_weight` pour XGBoost. Chaque technique a un compromis : overfitting possible, bruit synthetique possible, perte d'information possible, ou ponderation des erreurs.

## Slide 9 - Validation

Nous avons utilise une validation croisee stratifiee. Cela garde la meme proportion de churners dans chaque fold et rend l'evaluation plus stable.

## Slide 10 - Resultats au seuil 0.5

Au seuil par defaut 0.5, XGBoost avec `scale_pos_weight` donne un bon compromis : recall 69,6 %, precision 26,7 %, F1 0,386, ROC-AUC 0,797 et PR-AUC 0,282.

## Slide 11 - Ajustement du seuil

Comme le seuil 0.5 n'est pas optimal dans un contexte desequilibre, nous avons teste plusieurs seuils entre 0.10 et 0.90. Le seuil retenu est 0.20.

## Slide 12 - Choix final

Le modele final est XGBoost avec `scale_pos_weight`, au seuil 0.20. Il detecte 180 churners sur 204 dans le test set, soit un recall de 88,2 %. Il reste 24 faux negatifs, mais il y a 674 faux positifs.

## Slide 13 - Compromis metier

Ce compromis est acceptable en retention client : il vaut mieux contacter certains clients en trop que rater trop de clients qui allaient partir, tant que le cout de contact reste raisonnable.

## Slide 14 - Dashboard

Nous avons construit un dashboard Streamlit. Il affiche les KPI, les clients a risque, le revenu mensuel a risque, une simulation client, les metriques du modele et les faux positifs/faux negatifs.

## Slide 15 - Conclusion

Pour conclure, notre projet montre qu'il faut traiter explicitement le desequilibre : analyser la cible, comparer plusieurs techniques, utiliser des metriques adaptees et ajuster le seuil selon le besoin metier.
