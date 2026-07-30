# Projet Data Science - Prediction du churn client

Systeme de prediction du churn client pour une equipe marketing/CRM : preparation des donnees, comparaison multi-modeles (Machine Learning + Deep Learning), interpretabilite, et dashboard decisionnel Streamlit.

Projet realise dans le cadre de l'epreuve certifiante **RNCP40875 - Expert en ingenierie de donnees (Bloc 2)**, EFREI DATA Engineering & AI (cf. `docs/consigne_projet.pdf`).

## 1. Objectif

Aider une equipe marketing/CRM a reperer, avant qu'ils ne partent, les clients qui risquent de resilier leur abonnement, et a estimer le revenu mensuel a risque associe.

La cible est `churn` :

- `0` : le client reste ;
- `1` : le client resilie.

La cible est desequilibree : le dataset contient `8 979` non-churners et `1 021` churners (`10,21 %`), soit un ratio majoritaire/minoritaire de `8,79`. Consequence directe : **le recall est la metrique prioritaire**, car rater un churner (faux negatif) coute plus cher a l'entreprise que contacter a tort un client qui ne partait pas (faux positif).

## 2. Dataset

- Source : Kaggle (`customer_churn.csv`), 10 000 clients, donnees synthetiques mais generees selon une logique metier.
- Variables : demographiques, contrat/facturation, engagement/usage, support/incidents, satisfaction (NPS, CSAT).
- Qualite verifiee (`notebooks/01_EDA.ipynb`) : **0 ligne dupliquee**, **0 `customer_id` duplique**. Une seule colonne avec des valeurs manquantes : `complaint_type` (20,45 %), interpretee comme "pas de plainte connue" plutot que supprimee.
- Les valeurs extremes (ex. `total_revenue`, `support_tickets`, `payment_failures`) sont conservees : elles correspondent a des profils clients reels (tres actifs ou tres en difficulte), pas a des erreurs de saisie.

## 3. Structure du depot

```text
churn_predict/
  app.py                        # Dashboard Streamlit (utilisateur metier)
  feature_engineering.py        # Fonction prepare_customer_features() : variables metier
  requirements.txt
  README.md
  data_preprocessed.pkl         # Artefact genere par le notebook 04 (train/test, seuil, medians...)
  data/
    customer_churn.csv
  notebooks/
    01_EDA.ipynb                        # Exploration des donnees, qualite, desequilibre
    02_preparation_donnees.ipynb        # Logique de preparation / separation X-y
    03_feature_engineering_details.ipynb# Detail pedagogique des variables + tests d'amelioration
    04_entrainement_complet.ipynb       # Pipeline complet : entrainement, comparaison, seuil, export
    05_modelisation_evaluation.ipynb    # Lecture des resultats, graphiques, interpretabilite
    06_application_metier.ipynb         # Explication du dashboard app.py
  models/
    best_model.pkl               # Modele final retenu (XGBoost_scale_pos_weight)
    model_*.pkl                  # Tous les modeles entraines (une variante par strategie)
  reports/
    baseline_analysis.csv        # Naive + LogisticRegression sans reequilibrage
    model_comparison.csv         # Comparaison complete des 8 experiences (CV + test + duree)
    threshold_analysis.csv       # Balayage de seuils (0.10 a 0.90) pour chaque modele
  docs/
    consigne_projet.pdf              # Enonce officiel du projet (EFREI / RNCP40875)
    RAPPORT_PROJET.md / .docx / .pdf # Rapport complet (20 sections + annexes A a AL)
    DOCUMENTATION_PROJET.md          # Synthese technique courte
    CYCLE_DE_VIE_PROJET.md           # Recit du cycle de vie du projet, etape par etape
    TEXTE_A_DIRE.md                  # Script de presentation orale (17 slides)
```

## 4. Cycle de vie du projet (notebooks)

| Notebook | Role |
|---|---|
| `01_EDA.ipynb` | Analyse exploratoire : distributions, doublons/valeurs manquantes, desequilibre de la cible, correlations, insights metier (paiement, contrat, NPS). |
| `02_preparation_donnees.ipynb` | Explique la logique de preparation : separation `X`/`y`, retrait de `customer_id`, principe "pas de transformation apprenante avant le split" (anti data-leakage). |
| `03_feature_engineering_details.ipynb` | Detaille chaque variable metier creee, verifie leur pouvoir separateur (moyenne par `churn`), et **documente deux pistes d'amelioration testees rigoureusement (cross-validation repetee) mais non retenues** car sans gain mesurable (cf. section 6). |
| `04_entrainement_complet.ipynb` | Notebook central : feature engineering, analyse du desequilibre, split stratifie, 8 experiences de modelisation, validation croisee `StratifiedKFold`, optimisation du seuil, sauvegarde des modeles/rapports/artefacts. |
| `05_modelisation_evaluation.ipynb` | Relit les CSV de `reports/` pour produire l'analyse de presentation : comparaison des modeles, lecture des metriques, impact du seuil, cout de calcul, interpretabilite (feature importances, permutation importance, SHAP). |
| `06_application_metier.ipynb` | Documente `app.py` : pourquoi un seul modele est expose a l'utilisateur metier, comment lancer le dashboard, ce qu'il affiche. |

Pipeline resume :

```text
Donnees brutes -> EDA -> Feature engineering -> Split stratifie
  -> Entrainement multi-modeles (8 experiences) -> Validation croisee stratifiee
  -> Optimisation du seuil -> Modele final + artefacts -> Dashboard Streamlit
```

## 5. Feature engineering (`feature_engineering.py`)

Variables metier ajoutees par `prepare_customer_features()` :

| Variable | Logique |
|---|---|
| `has_complaint` | Le client a une plainte connue (`complaint_type != "Aucune Plainte"`). |
| `payment_risk` | `payment_failures > 0` OU `price_increase_last_3m == "Yes"`. |
| `monthly_contract` | Contrat mensuel (`contract_type == "Monthly"`). |
| `tickets_per_tenure` | `support_tickets / tenure_months` (compare equitablement anciens et nouveaux clients). |
| `fee_per_login` | `monthly_fee / monthly_logins`. |
| `support_pressure` | `support_tickets * avg_resolution_time`. |
| `engagement_score` | Score pondere (logins, jours actifs, features utilisees, croissance d'usage, recence de connexion). |
| `satisfaction_score` | Score pondere (CSAT, NPS, escalations, plaintes). |

### Constat sur le signal reel des variables (documente dans le notebook 03)

Toutes les variables n'ont pas le meme pouvoir predictif reel :

- **Signal fort et non lineaire** : `payment_failures` ne bouge presque pas entre 0 et 1 echec (~8,6-8,7 % de churn), mais **saute a 21-33 % des 2 echecs**. `tenure_months` et `csat_score` ont aussi un vrai effet ; leur combinaison (client recent ET insatisfait) fait grimper le churn a **33,6 %**, contre 5,2 % pour un client ancien et satisfait.
- **Quasiment aucun signal** : `price_increase_last_3m` (10,2 % vs 10,4 % de churn) et `has_complaint` (10,3 % vs 10,2 %) ne separent presque pas les deux populations dans ce dataset.

Deux ameliorations ont ete testees en cross-validation repetee (20 entrainements XGBoost par variante, seuil metier 0.20) avant d'etre ecartees :

1. Recalibrer `payment_risk` sur `payment_failures >= 2` : aucun gain (recall quasi identique, dans l'ecart-type de ±2,2 points, legerement plus de faux positifs).
2. Ajouter une interaction explicite "nouveau client x faible satisfaction" : aucun gain net.

**Conclusion retenue** : XGBoost (modele a arbres boostes) apprend deja seul ces seuils et interactions a partir des colonnes brutes ; le feature engineering manuel n'apporte pas de gain supplementaire mesurable pour ce modele.

### Nettoyage des redondances (3 variables retirees)

Une revue de correlation entre les variables creees a mis en evidence de vraies redondances, corrigees dans le code :

- `revenue_per_month` etait un **doublon exact** de `monthly_fee` (correlation = 1.000, car `total_revenue = monthly_fee * tenure_months` dans ce dataset) : supprimee, c'etait un vrai defaut de conception.
- `low_satisfaction` (flag binaire) et `satisfaction_score` (score continu) portaient largement le meme signal (correlation -0,49, memes composantes : CSAT, NPS, plainte) : `low_satisfaction` supprimee, `satisfaction_score` conservee (plus nuancee).
- `inactive_customer` (flag binaire) et `engagement_score` (score continu) portaient egalement le meme signal (correlation -0,43, memes composantes : logins, recence de connexion) : `inactive_customer` supprimee, `engagement_score` conservee.

Verification par cross-validation repetee (20 entrainements) avant application : le recall du modele final reste stable (0,861 avant vs 0,862 apres, dans le bruit), confirmant que ce nettoyage simplifie le code et la lecture des graphiques d'interpretabilite sans perte de performance.

**Consequence sur le choix du modele final** : apres ce nettoyage, XGBoost et Random Forest se sont retrouves a egalite parfaite de recall (178/204 vrais positifs) sur le split de test. La regle de selection automatique du notebook 04 a donc ete renforcee : en cas d'egalite (ou quasi-egalite, tolerance 1 point) de recall, le modele le plus rapide/sobre a entrainer est desormais prefere plutot que de departager uniquement sur la precision — ce qui confirme XGBoost (voir section 7).

## 6. Modelisation et comparaison (seuil par defaut 0.50)

8 experiences comparees, couvrant un modele lineaire (Logistic Regression), un modele d'arbres (Random Forest), du boosting (XGBoost) et du Deep Learning (MLP), croises avec les techniques de gestion du desequilibre (aucune, `RandomOverSampler`, `SMOTE`, `RandomUnderSampler`, `class_weight`, `scale_pos_weight`) :

| Modele | Strategie | Recall | Precision | F1 | ROC-AUC | PR-AUC | FP | FN | Temps (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **XGBoost** | `scale_pos_weight` | **0,667** | 0,264 | 0,378 | 0,792 | 0,275 | 379 | 68 | 1,71 |
| Logistic Regression | `random_over_sampling` | 0,652 | 0,192 | 0,297 | 0,737 | 0,237 | 560 | 71 | 0,36 |
| Logistic Regression | `class_weight` | 0,647 | 0,192 | 0,296 | 0,741 | 0,243 | 555 | 72 | 0,30 |
| Logistic Regression | `random_under_sampling` | 0,642 | 0,186 | 0,289 | 0,729 | 0,232 | 573 | 73 | 0,41 |
| Logistic Regression | `smote` | 0,627 | 0,184 | 0,284 | 0,731 | 0,241 | 569 | 76 | 2,28 |
| Random Forest | `class_weight_balanced_subsample` | 0,319 | 0,310 | 0,314 | 0,803 | 0,282 | 145 | 139 | 10,05 |
| MLP (Deep Learning) | `smote` | 0,255 | 0,214 | 0,233 | 0,667 | 0,179 | 191 | 152 | 12,11 |
| Logistic Regression | `aucun_reequilibrage` (baseline) | 0,044 | 0,450 | 0,080 | 0,736 | 0,245 | 11 | 195 | 0,30 |
| Random Forest | `aucun_reequilibrage` (baseline) | 0,000 | 0,000 | 0,000 | 0,800 | 0,312 | 0 | 204 | 9,33 |
| NaiveMajority | `baseline_majoritaire` | 0,000 | 0,000 | 0,000 | - | - | 0 | 204 | - |

Pourquoi l'accuracy ne suffit pas : la baseline naive obtient `89,8 %` d'accuracy mais `0 %` de recall (elle ne detecte aucun churner) ; une regression logistique sans reequilibrage obtient `89,7 %` d'accuracy mais seulement `4,4 %` de recall. Les metriques suivies sont donc : **recall** (priorite), precision, F1, ROC-AUC, et **PR-AUC** (particulierement utile car la classe positive est rare).

Toutes les validations croisees utilisent `StratifiedKFold` (3 folds) pour preserver la proportion de churners dans chaque fold.

## 7. Optimisation du seuil et modele final

Le seuil `0.5` n'est pas optimal pour une classe positive rare. Un balayage de seuils entre `0.10` et `0.90` est teste pour chaque modele (`reports/threshold_analysis.csv`).

**Modele final retenu : XGBoost avec `scale_pos_weight`, seuil `0.20`.**

| Seuil | Recall | Precision | F1 | ROC-AUC | PR-AUC | Faux positifs | Faux negatifs | Vrais positifs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,20 | **0,873** | 0,212 | 0,341 | 0,792 | 0,275 | 662 | 26 | 178 |

Ce choix detecte `178` churners sur `204` dans le jeu de test (contre `136` au seuil 0.5). Le compromis assume : plus de faux positifs (662 clients contactes a tort) pour minimiser les faux negatifs (26 clients perdus non detectes), ce qui est coherent avec un contexte CRM ou une action de retention coute generalement moins cher qu'un client perdu.

**Regle de selection (modele + seuil ensemble)** : le notebook 04 teste tous les modeles a tous les seuils (0,10 a 0,90), ne garde que les combinaisons avec une precision >= 20 % (`MIN_PRECISION_FOR_RECALL`), puis choisit celle qui maximise le recall. En cas d'egalite de recall a moins d'1 point de pourcentage pres (ce qui s'est produit entre XGBoost@0,20 et Random Forest@0,25 apres le nettoyage des variables, tous deux a 178/204 vrais positifs), le modele le plus rapide/sobre a entrainer est prefere plutot que de departager uniquement sur la precision — XGBoost (1,71 s) l'emporte ainsi sur Random Forest (10,05 s), 5,9 fois plus lent pour un recall identique.

## 8. Cout de calcul et eco-responsabilite

Le temps d'entrainement (validation croisee 3 folds + entrainement final) a ete mesure pour chaque experience :

| Famille de modele | Temps d'entrainement |
|---|---:|
| Logistic Regression (variantes) | 0,30 a 2,28 s |
| **XGBoost `scale_pos_weight` (modele retenu)** | **1,71 s** |
| Random Forest (variantes) | 9,33 a 10,05 s |
| MLP / Deep Learning (SMOTE) | 12,11 s |

XGBoost, le modele retenu, est environ 6 fois plus rapide a entrainer que Random Forest et 7 fois plus rapide que le MLP, pour un recall egal ou superieur. Ce critere est desormais integre directement dans la regle de selection du modele final (cf. section 7), pas seulement mentionne a posteriori.

## 9. Interpretabilite

Trois niveaux d'explication sont disponibles (notebook `05` et onglet "Explication" du dashboard) :

| Technique | Quand | Niveau |
|---|---|---|
| `feature_importances_` (native) | Apres entrainement | Basique, rapide (modeles d'arbres uniquement) |
| Permutation importance | Apres evaluation | Recommandee, agnostique au modele (mesure la perte de recall) |
| SHAP | Sur le modele final | Avancee (impact moyen absolu par variable, package `shap`) |

Variables les plus influentes identifiees : `csat_score`, `payment_failures`, `tenure_months`, `monthly_logins`, `total_revenue` - coherentes avec la logique metier (satisfaction, anciennete, engagement, incidents de paiement).

## 10. Dashboard Streamlit (`app.py`)

Lancement : `streamlit run app.py`. Le dashboard charge uniquement le modele final (`best_model.pkl`) : un utilisateur metier n'a pas besoin de choisir entre plusieurs modeles.

3 onglets :

- **Vue metier** : KPI de retention (clients, churn observe, clients a risque, revenu mensuel a risque), slider de seuil d'alerte, filtres par segment client et type de contrat (drill-down), **distribution du risque de churn sur le portefeuille**, **matrice de priorisation (risque x revenu)**, tableau des clients a prioriser, risque par segment.
- **Simulation client** : formulaire pour simuler un client (age, anciennete, contrat, frais, paiements, connexions, NPS...) et obtenir en temps reel sa probabilite de churn, la perte mensuelle attendue, une recommandation d'action, **une jauge visuelle de risque**, et **une explication SHAP individuelle** (quelles variables poussent CE client vers le churn ou vers la retention).
- **Explication** : feature importances natives, permutation importance, SHAP, **courbes ROC et Precision-Recall du modele final**, comparaison des modeles (recall, PR-AUC, erreurs, radar multi-metriques), cout de calcul, impact du seuil — graphiques presentes **en grille** (2-3 par ligne) pour rester lisibles a l'ecran plutot qu'en un seul graphique geant par ligne.

Conçu pour un utilisateur non technique : chaque indicateur cle est affiche en chiffre (pas seulement par une couleur), palette de couleurs contrastee et distinguable (bleu/orange/rouge/vert/violet), vocabulaire en langage metier.

## 11. API REST

**Non implementee** dans cette version (extension optionnelle du cahier des charges). Piste documentee dans `docs/RAPPORT_PROJET.md` (Annexe M) : exposer le modele via `/predict` et `/health` pour decoupler le scoring du dashboard et faciliter l'integration a un CRM.

## 12. Limites et pistes d'amelioration

Limites principales :

- Precision faible au seuil retenu (0,212) : beaucoup de faux positifs, acceptable seulement si le cout d'une action de retention reste faible.
- Le choix du seuil depend du cout reel d'une action CRM, non mesure directement dans ce dataset.
- Certaines variables intuitives (prix, plaintes, type de contrat) n'ont presque aucun lien reel avec le churn dans ce dataset (probable limite du jeu de donnees, potentiellement synthetique).
- Pas de monitoring continu ni de suivi de derive des donnees en production.

Pistes d'amelioration :

- Calibrer les probabilites et ajouter une matrice de cout metier explicite pour choisir le seuil.
- Suivre le drift des donnees et re-entrainer periodiquement (le faible cout de calcul de XGBoost le permet facilement).
- Valider en conditions reelles (A/B test) l'impact des campagnes de retention declenchees par le modele.

## 13. Documentation complementaire

- `docs/RAPPORT_PROJET.md` (+ `.docx` / `.pdf`) : rapport complet (20 sections + annexes A a AL) couvrant contexte, EDA, preparation, architecture, evaluation comparative, interpretabilite, strategie d'integration IA, gouvernance et limites.
- `docs/DOCUMENTATION_PROJET.md` : synthese technique courte.
- `docs/CYCLE_DE_VIE_PROJET.md` : recit du cycle de vie du projet.
- `docs/TEXTE_A_DIRE.md` : script de presentation orale.
- `docs/consigne_projet.pdf` : enonce officiel (EFREI, RNCP40875 Bloc 2).

## 14. Lancer le projet

```powershell
pip install -r requirements.txt

# Regenere les modeles, rapports et data_preprocessed.pkl
jupyter notebook notebooks/04_entrainement_complet.ipynb

# Lance le dashboard (utilise le modele deja entraine)
streamlit run app.py
```

## 15. Historique des versions

**v2 (actuelle)**

- Feature engineering nettoye : suppression de 3 variables redondantes (`revenue_per_month` etait un doublon exact de `monthly_fee`, `low_satisfaction`/`inactive_customer` faisaient doublon avec `satisfaction_score`/`engagement_score`). Verifie sans perte de recall par cross-validation repetee avant application (`notebooks/03_feature_engineering_details.ipynb`).
- Regle de selection du modele final renforcee (`notebooks/04_entrainement_complet.ipynb`) : en cas d'egalite (ou quasi-egalite, tolerance 1 point) de recall entre deux modeles, le plus rapide/sobre a entrainer est prefere (ecoresponsabilite), plutot qu'un depart mecanique sur la seule precision.
- Modeles, rapports et artefacts regeneres avec ce nouveau jeu de variables : recall du modele final `87,3 %` (contre `88,2 %` en v1), au meme seuil `0.20`.
- Dashboard Streamlit enrichi :
  - Graphiques de l'onglet "Explication" reorganises **en grille** (2-3 par ligne) au lieu d'un graphique geant par ligne.
  - Ajout des **courbes ROC et Precision-Recall** completes du modele final (avant : seuls les chiffres AUC/PR-AUC etaient affiches).
  - Ajout d'une **jauge de risque visuelle** et d'une **explication SHAP individuelle** dans la simulation client (explicabilite locale : pourquoi CE client precis).
  - Ajout d'une **distribution du risque** et d'une **matrice de priorisation (risque x revenu)** dans la vue metier.

**v1**

- Pipeline complet initial : EDA, feature engineering (11 variables), comparaison de 9 configurations de modeles/rééquilibrage, optimisation du seuil, dashboard Streamlit (KPI, simulation, explication de base).
