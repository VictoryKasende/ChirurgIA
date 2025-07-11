# ChirurgIA - Projet de Machine Learning en Chirurgie Générale

Ce projet de Machine Learning vise à analyser les données médicales relatives à la chirurgie générale pour :

1. **Déterminer les causes de décès probables**
2. **Prédire les cas de décès après admission/chirurgie**
3. **Grouper les survivants selon la criticité post-opératoire**

## Structure du Projet

```
ChirurgIA/
├── data/                   # Données d'entrée et traitées
├── notebooks/              # Notebooks Jupyter pour l'exploration
├── src/                    # Code source modulaire
├── models/                 # Modèles entraînés
├── app/                    # Application de déploiement
├── docs/                   # Documentation et article
└── requirements.txt        # Dépendances Python
```

## Installation

1. **Cloner/Naviguer vers le projet** :
   ```bash
   cd /home/victory/Documents/ChirurgIA
   ```

2. **Créer un environnement virtuel** :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   ```

3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

4. **Installer le modèle scispaCy** :
   ```bash
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gz
   ```

## Démarrage Rapide

### 1. Exploration des Données
Commencez par les notebooks dans `notebooks/` :
- `01_data_exploration.ipynb` - Exploration initiale
- `02_data_preprocessing.ipynb` - Nettoyage et préparation
- `03_feature_engineering.ipynb` - Ingénierie des caractéristiques

### 2. Modélisation
- `04_mortality_prediction.ipynb` - Prédiction de mortalité
- `05_cause_analysis.ipynb` - Analyse des causes de décès
- `06_patient_clustering.ipynb` - Clustering des patients

### 3. Déploiement
```bash
cd app/
streamlit run app.py
```

## Méthodologie

### Préparation des Données
- **Données numériques** : Signes vitaux (Temperature, pH, pCO2, etc.)
- **Données textuelles** : Diagnostic, chirurgie, problèmes (ScispaCy)
- **Données catégorielles** : Âge, race, outcome

### Modèles Prévus
- **Classification** : Random Forest, XGBoost, LightGBM
- **Clustering** : K-Means, DBSCAN, Hierarchical
- **Analyse textuelle** : NLP médical avec ScispaCy

### Métriques d'Évaluation
- Précision, Rappel, F1-Score
- ROC-AUC, Precision-Recall AUC
- Silhouette Score pour clustering

## Livrables

1. **Article scientifique** (dans `docs/`)
2. **Code documenté** (dans `src/` et `notebooks/`)
3. **Application de démonstration** (dans `app/`)

## Équipe
Projet réalisé par une équipe de 4 personnes dans le cadre du cours de Machine Learning.

## License
Projet académique - Usage éducatif uniquement.
