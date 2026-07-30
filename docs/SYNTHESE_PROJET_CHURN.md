# Synthèse — Prédiction du churn client (Projet 2/3, rétention client)

Auteurs : MELOTTI Simone, MEDJDOUB Amina
Promotion : M1 Data Engineering & IA — Année scolaire 2025-2026
Tuteur / Formatrice : Sarah MALAEB
Projet certifiant RNCP40875 — Expert en Ingénierie de Données — Bloc 2 : Pilotage et implémentation de solutions IA

*Version synthétique du rapport complet (`RAPPORT_PROJET.md`, 20 sections + annexes A à AL), condensée pour une lecture combinée avec les deux autres projets de la promotion.*

---

## 1. Résumé exécutif

Ce projet vise à prédire le churn client pour aider une équipe CRM/marketing à prioriser ses actions de rétention. Le contexte est celui d'une entreprise par abonnement (SaaS/télécom), où chaque départ entraîne une perte de revenu récurrent et un coût de remplacement. Le dataset (Kaggle, 10 000 clients) présente une cible binaire fortement déséquilibrée : 8 979 non-churners contre 1 021 churners (10,21 %, ratio 8,79). Ce déséquilibre rend l'accuracy trompeuse et impose des métriques orientées détection : recall, precision, F1, ROC-AUC et surtout PR-AUC.

La démarche suit une chaîne Data Science complète : EDA, feature engineering métier, comparaison de 4 familles de modèles (Logistic Regression, Random Forest, XGBoost, MLP/Deep Learning) croisées avec 5 stratégies de gestion du déséquilibre, validation croisée stratifiée, optimisation du seuil de décision, puis mise à disposition dans un dashboard Streamlit décisionnel.

Le modèle final retenu est **XGBoost avec `scale_pos_weight`, seuil 0.20**. Il détecte 178 churners sur 204 dans le jeu de test (recall 87,3 %), pour une précision de 21,2 %. Ce compromis est assumé : dans un contexte CRM, rater un churner coûte généralement plus cher que contacter à tort un client fidèle. Le dashboard rend la solution actionnable : KPI de rétention, revenu à risque, simulation client avec explication SHAP individuelle, comparaison des modèles et impact du seuil.

Le pipeline présenté ici est une version corrigée : la revue du projet a fait apparaître deux limites (redondance dans le feature engineering, règle de sélection du modèle final incomplète), traitées et vérifiées avant application (section 10).

---

## 2. Contexte métier et utilisateur cible

La rétention client est généralement moins coûteuse que l'acquisition. Un départ non anticipé implique une perte de revenu, un effort commercial de remplacement et un risque d'érosion du portefeuille. L'objectif du projet n'est pas seulement d'entraîner un modèle mais de construire un outil d'aide à la décision exploitable par une équipe métier.

**Utilisateur final** : un responsable CRM, marketing ou financier chargé de la rétention, avec des contraintes de budget (coût des campagnes), de temps (priorisation rapide) et de compréhension (besoin d'indicateurs clairs, pas de jargon statistique).

**Scénarios d'usage** :
- préparer une campagne de rétention ciblant 5 à 10 % de la base client, ordonnée par risque ;
- estimer le revenu mensuel à risque pour justifier une allocation budgétaire ;
- identifier les segments les plus sensibles pour adapter le discours commercial.

**Décisions à soutenir** : prioriser les clients à contacter (risque × valeur), estimer le revenu à risque, choisir un seuil de décision selon le coût métier, interpréter les facteurs de churn pour adapter l'action (offre, support proactif, appel personnalisé).

---

## 3. Données

- **Source** : Kaggle (`customer_churn.csv`), 10 000 clients, données synthétiques générées selon une logique métier réaliste.
- **Familles de variables** : contrat et facturation (type de contrat, frais mensuels, revenu total, hausse de prix) ; engagement et usage (connexions, jours actifs, croissance d'usage, récence de connexion) ; support et incidents (tickets, temps de résolution, escalades) ; satisfaction (NPS, CSAT, réponse à enquête).
- **Cible** : `churn` (0 = reste, 1 = résilié), déséquilibrée à 10,21 % de churners.
- **Qualité** : vérifiée (notebook `01_EDA`) — 0 ligne dupliquée, 0 `customer_id` dupliqué. Une seule colonne avec valeurs manquantes, `complaint_type` (20,45 %), interprétée comme « pas de plainte connue » plutôt que supprimée. Les valeurs extrêmes (revenu, tickets, échecs de paiement) sont conservées car elles correspondent à des profils clients réels, pas à des erreurs de saisie.
- **Limite principale** : dataset synthétique, sans historique temporel ni coût réel d'une action CRM — le choix du seuil de décision (section 7) ne peut donc pas s'appuyer sur une vraie matrice de coût.

---

## 4. Méthodologie et pipeline

Organisation en 4 phases : (1) compréhension du besoin et du dataset, (2) EDA + préparation + feature engineering, (3) entraînement multi-modèles et évaluation comparative, (4) sélection du modèle final, ajustement du seuil, dashboard.

```
Données brutes -> EDA -> Feature engineering -> Split stratifié
  -> Entraînement multi-modèles (8 expériences) -> Validation croisée stratifiée
  -> Optimisation du seuil -> Modèle final + artefacts -> Dashboard Streamlit
```

Points méthodologiques clés :
- **Anti data-leakage** : séparation X/y et retrait de `customer_id` avant toute transformation ; aucune transformation apprenante n'est ajustée avant le split train/test.
- **Stratification** : split et validation croisée (`StratifiedKFold`, 3 folds) conservent la proportion de churners dans chaque sous-ensemble, ce qui stabilise le recall malgré la rareté de la classe positive.
- **Reproductibilité** : notebooks versionnés, script `feature_engineering.py` unique, rapports CSV et artefacts modèles régénérés à chaque exécution du notebook central.

---

## 5. Préparation et feature engineering

Huit variables métier sont ajoutées par `prepare_customer_features()` :

| Variable | Logique |
|---|---|
| `has_complaint` | Plainte connue (`complaint_type != "Aucune Plainte"`) |
| `payment_risk` | Échec de paiement récent OU hausse de prix récente |
| `monthly_contract` | Contrat mensuel (plus fragile qu'un contrat long) |
| `tickets_per_tenure` | Tickets support rapportés à l'ancienneté |
| `fee_per_login` | Frais mensuels rapportés à l'usage |
| `support_pressure` | Volume de tickets × temps de résolution |
| `engagement_score` | Score pondéré (connexions, jours actifs, usage, récence) |
| `satisfaction_score` | Score pondéré (CSAT, NPS, escalades, plaintes) |

Deux vérifications de rigueur ont été menées par cross-validation répétée (20 entraînements XGBoost par variante) avant d'être écartées ou appliquées : un recalibrage de `payment_risk` (aucun gain mesurable) et une interaction « nouveau client × faible satisfaction » (aucun gain net). Le jeu de variables présenté ci-dessus est la version corrigée du feature engineering, après suppression de 3 variables redondantes identifiées lors de la revue du projet (détail et réflexion en section 10).

Constat marquant sur le signal réel : `payment_failures` ne bouge presque pas entre 0 et 1 échec (~8,6 % de churn) mais saute à 21-33 % dès 2 échecs ; `price_increase_last_3m` et `has_complaint`, en revanche, ne séparent presque pas les deux populations dans ce dataset.

---

## 6. Modélisation et comparaison des modèles

Quatre familles de modèles sont comparées, croisées avec les techniques de gestion du déséquilibre (aucune, sur-échantillonnage, sous-échantillonnage, SMOTE, pondération des classes) :

| Modèle | Meilleure stratégie | Recall | Precision | F1 | ROC-AUC | PR-AUC | Temps (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| **XGBoost** | scale_pos_weight | **0,667** | 0,264 | 0,378 | 0,792 | 0,275 | 1,71 |
| Logistic Regression | random_over_sampling | 0,652 | 0,192 | 0,297 | 0,737 | 0,237 | 0,36 |
| Random Forest | class_weight_balanced_subsample | 0,319 | 0,310 | 0,314 | 0,803 | 0,282 | 10,05 |
| MLP (Deep Learning) | smote | 0,255 | 0,214 | 0,233 | 0,667 | 0,179 | 12,11 |
| Baseline naïve / LogReg sans rééquilibrage | - | 0,000 / 0,044 | - / 0,450 | - | - | - | - |

Pourquoi l'accuracy ne suffit pas : la baseline naïve obtient 89,8 % d'accuracy mais 0 % de recall (elle ne détecte aucun churner) ; une régression logistique sans rééquilibrage obtient un profil similaire (accuracy élevée, recall quasi nul). C'est pourquoi le recall et la PR-AUC pilotent la comparaison plutôt que l'accuracy.

XGBoost avec `scale_pos_weight` offre le meilleur compromis global dès le seuil par défaut (0.50) et reste environ 6 fois plus rapide à entraîner que Random Forest et 7 fois plus rapide que le MLP — un argument d'écoresponsabilité qui pèse explicitement dans le choix final, pas seulement une note en bas de page.

---

## 7. Choix du modèle final et optimisation du seuil

Le seuil 0.50 n'est pas optimal pour une classe positive rare. Un balayage entre 0.10 et 0.90 est testé pour chaque modèle :

| Seuil | Recall | Precision | F1 | Faux positifs | Faux négatifs | Vrais positifs |
|---:|---:|---:|---:|---:|---:|---:|
| 0,50 | 0,667 | 0,264 | 0,378 | 379 | 68 | 136 |
| **0,20 (retenu)** | **0,873** | 0,212 | 0,341 | 662 | 26 | 178 |

**Modèle final : XGBoost (`scale_pos_weight`), seuil 0.20.** Il détecte 178 churners sur 204, contre 136 au seuil 0.50. Le compromis est assumé : plus de faux positifs (662 clients contactés à tort) pour minimiser les faux négatifs (26 clients perdus non détectés) — cohérent avec un contexte CRM où une action de rétention coûte généralement moins cher qu'un client perdu.

**Règle de sélection (modèle + seuil)** : parmi les combinaisons modèle/seuil respectant une précision minimale (≥ 20 %), celle qui maximise le recall est retenue, avec un critère de coût de calcul en cas d'égalité. Après le nettoyage des variables, XGBoost et Random Forest se sont retrouvés à égalité quasi parfaite de recall (178/204). Dans ce cas, le modèle le plus rapide/sobre à entraîner est préféré plutôt que de départager sur la seule précision : XGBoost (1,71 s) l'emporte sur Random Forest (10,05 s), 5,9 fois plus lent pour un recall identique. Cette règle a été renforcée suite à la revue du projet (section 10).

---

## 8. Interprétabilité

Trois niveaux d'explication sont disponibles : `feature_importances_` natif (rapide, modèles d'arbres uniquement), permutation importance (agnostique au modèle, mesure la perte de recall), et SHAP (impact individuel et global, sur le modèle final). Les variables les plus influentes identifiées sont `csat_score`, `payment_failures`, `tenure_months`, `monthly_logins` et `total_revenue` — cohérentes avec la logique métier : un client insatisfait, en difficulté de paiement, récent et peu actif est plus susceptible de partir.

Cette double lecture (globale via feature importance/SHAP, locale via l'explication par client dans le dashboard) permet de transformer un score statistique en argument actionnable pour un responsable CRM : pourquoi ce client précis est-il à risque, et quelle action y répondre.

---

## 9. Dashboard décisionnel (Streamlit)

Le dashboard charge uniquement le modèle final (`best_model.pkl`) — un utilisateur métier n'a pas besoin de choisir entre plusieurs modèles. Trois onglets :

- **Vue métier** : KPI de rétention (clients, churn observé, clients à risque, revenu mensuel à risque), slider de seuil d'alerte, filtres par segment et type de contrat, distribution du risque sur le portefeuille, matrice de priorisation (risque × revenu), tableau des clients à prioriser.
- **Simulation client** : formulaire de simulation (âge, ancienneté, contrat, paiements, connexions, NPS...) avec probabilité de churn en temps réel, perte mensuelle attendue, recommandation d'action, jauge visuelle de risque et explication SHAP individuelle (quelles variables poussent ce client vers le churn).
- **Explication** : feature importances, permutation importance, SHAP, courbes ROC et Precision-Recall du modèle final, comparaison des modèles (recall, PR-AUC, erreurs, coût de calcul).

Conçu pour un utilisateur non technique : indicateurs clés affichés en chiffres (pas seulement par couleur), palette contrastée et distinguable, vocabulaire métier plutôt que jargon statistique.

**API REST** : non implémentée dans cette version (extension optionnelle du cahier des charges). Piste documentée : exposer le modèle via `/predict` et `/health` pour découpler le scoring du dashboard et faciliter l'intégration à un CRM.

---

## 10. Itération et retour d'expérience (revue du projet)

Une première version du pipeline (feature engineering à 11 variables, règle de sélection du modèle final fondée sur le seul recall) atteignait un recall de 88,2 % (180/204 churners détectés) avec le modèle final. La revue du projet avec la tutrice a fait apparaître deux limites qui n'étaient pas visibles à la seule lecture des métriques finales.

**Problème 1 — redondance dans le feature engineering.** Trois variables dérivées portaient en réalité un signal déjà capturé ailleurs : `revenue_per_month` était un doublon exact de `monthly_fee` (corrélation = 1.000, conséquence directe de `total_revenue = monthly_fee * tenure_months` dans ce dataset) ; `low_satisfaction` et `inactive_customer` (indicateurs binaires) recoupaient largement `satisfaction_score` et `engagement_score` (scores continus, corrélations -0,49 et -0,43). Ce n'était pas une erreur de calcul, mais un défaut de conception : ces variables compliquaient la lecture des graphiques d'interprétabilité (feature importance, SHAP) sans apporter d'information supplémentaire au modèle.

*Correction apportée* : suppression des 3 variables redondantes, avec vérification préalable par cross-validation répétée (20 entraînements) que le recall du modèle final restait stable — 0,861 avant nettoyage contre 0,862 après, un écart dans le bruit de mesure. Le changement a été vérifié avant d'être appliqué, pas supposé.

**Problème 2 — une règle de sélection incomplète.** La règle initiale choisissait le modèle final sur le seul recall (à précision minimale égale), sans critère de départage en cas d'égalité. Or le nettoyage des variables a fait apparaître une égalité quasi parfaite de recall entre XGBoost et Random Forest (178/204 churners détectés par les deux modèles).

*Correction apportée* : la règle de sélection a été renforcée — en cas d'égalité (ou quasi-égalité, tolérance 1 point de recall), le modèle le plus rapide et le plus sobre à entraîner est désormais préféré, plutôt que de départager arbitrairement sur la seule précision. Cette règle confirme XGBoost (1,71 s d'entraînement) plutôt que Random Forest (10,05 s, 5,9 fois plus lent) pour un recall identique — un argument qui est aussi celui de l'écoresponsabilité.

**Conséquence sur le résultat final** : le recall du modèle retenu passe de 88,2 % (première version) à 87,3 % (version corrigée), un écart de moins d'un point qui reste dans le bruit de mesure. La conclusion métier ne change pas, mais le pipeline est plus simple, plus explicable, et sa règle de sélection est désormais plus défendable devant un jury.

**Réflexion** : cette itération montre l'intérêt d'une revue externe pour challenger des choix qui paraissent corrects sur le papier mais restent fragiles à l'examen — une variable redondante ou une règle de sélection incomplète ne se voient pas toujours dans un score global. Elle illustre aussi une méthode plutôt qu'un correctif ponctuel : chaque changement a été vérifié par ré-entraînement avant d'être appliqué, jamais supposé. C'est cette même logique qui devrait guider toute évolution future du projet (section 11).

---

## 11. Gouvernance, limites et pistes d'amélioration

**Gouvernance** : qualité des données et traçabilité assurées par les notebooks versionnés ; risque de biais lié à la rareté des churners ; surapprentissage contrôlé par validation croisée stratifiée et comparaison multi-modèles ; dérive potentielle des comportements clients dans le temps, non surveillée actuellement.

**Limites principales** :
- précision faible au seuil retenu (21,2 %) — volume important de faux positifs, acceptable seulement si le coût d'une action de rétention reste faible ;
- le choix du seuil dépend du coût réel d'une action CRM, non mesuré dans ce dataset synthétique ;
- certaines variables intuitives (prix, plaintes, type de contrat) n'ont presque aucun lien réel avec le churn ici, probable limite du dataset ;
- absence de monitoring continu ou de suivi de dérive en production.

**Pistes d'amélioration** : calibrer les probabilités et ajouter une matrice de coût métier explicite pour choisir le seuil ; suivre le drift des données et ré-entraîner périodiquement (le faible coût de calcul de XGBoost le permet facilement) ; valider en conditions réelles (A/B test) l'impact des campagnes de rétention déclenchées par le modèle ; exposer le modèle via une API REST pour l'intégrer à un CRM.

---

## 12. Conclusion

Ce projet transforme un dataset client déséquilibré en solution Data Science complète et exploitable : EDA, feature engineering métier, comparaison rigoureuse de 4 familles de modèles, optimisation du seuil de décision et dashboard décisionnel. Le choix de XGBoost (`scale_pos_weight`, seuil 0.20) privilégie la détection des churners (recall 87,3 %), cohérent avec une logique CRM où manquer un client à risque coûte plus cher qu'un contact inutile. Au-delà du score, la valeur du projet repose sur la démarche : métriques adaptées au déséquilibre, vérification par cross-validation de chaque choix de feature engineering, arbitrage explicite entre performance et coût de calcul, et interface rendant la prédiction actionnable par une équipe métier non technique. Le projet illustre enfin une démarche itérative assumée : les limites identifiées lors de la revue du projet (section 10) ont été traitées et vérifiées avant d'être intégrées, plutôt que simplement listées comme améliorations futures.

*Pour le détail complet (méthodologie étape par étape, 38 annexes techniques, glossaire, notebooks) : voir `docs/RAPPORT_PROJET.md` et le dépôt GitHub du projet.*
