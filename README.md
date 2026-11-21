# Prédiction de l'attrition bancaire (Customer Churn) — Python & Power BI

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-EB4034)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?logo=powerbi&logoColor=black)

Projet d'intégration réalisé dans le cadre du **Certificat en science des données (SCI 1402)** à l'**Université TÉLUQ**. L'objectif est d'identifier les facteurs statistiquement associés au départ des clients d'une banque ($N = 10\,000$), de bâtir un modèle prédictif robuste et de déployer un outil décisionnel interactif dans Power BI permettant de prioriser les actions de rétention.

---

## 1. Contexte et objectifs d'affaires

L'acquisition d'un nouveau client bancaire coûte nettement plus cher que la rétention d'un client existant. Face à un taux d'attrition global de **20,37 %**, ce projet répond à trois besoins opérationnels :

1. **Comprendre les profils à risque** grâce à des tests d'hypothèses inférentiels rigoureux (sans confondre corrélation et causalité).
2. **Prédire la probabilité de départ** de chaque client à l'aide d'algorithmes d'apprentissage supervisé évalués sur des données hors échantillon.
3. **Outiller les équipes d'affaires** via un rapport Power BI en trois volets reliant l'analyse exploratoire, la performance des modèles et une liste d'intervention prioritaire chiffrée en dollars.

---

## 2. Structure du dépôt

```text
bank-customer-churn/
│
├── notebooks/
│   └── Analyse_Churn_Bancaire_SCI1402.ipynb   # Pipeline complet (EDA, tests statistiques, ML, exports CSV)
│
├── data/
│   ├── Customer-Churn-Records.csv             # Jeu de données brut (10 000 clients)
│   ├── bank_churn_predictions_powerbi.csv     # Prédictions hors échantillon (jeu de test, n = 2 000)
│   ├── comparaison_modeles_powerbi.csv        # Métriques numériques de validation croisée (k = 5)
│   └── importance_xgboost.csv                 # Contribution relative des 16 variables explicatives
│
├── powerbi/
│   └── Dashboard_Churn_Bancaire.pbix          # Tableau de bord interactif (3 pages)
│
├── images/
│   ├── page1_analyse_clients.png              # Capture de la Page 1 (Analyse Clients)
│   ├── page2_performance_ml.png               # Capture de la Page 2 (Performance Modèles ML)
│   └── page3_clients_a_risque.png             # Capture de la Page 3 (Clients à Risque)
│
└── README.md