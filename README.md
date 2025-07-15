# ChirurgIA - Système d'Intelligence Artificielle pour la Chirurgie Générale

<div align="center">

![ChirurgIA Logo](https://img.shields.io/badge/ChirurgIA-AI%20Medical%20System-blue?style=for-the-badge&logo=medical-cross)

**Système d'IA avancé pour l'analyse prédictive en chirurgie générale**

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

</div>

## 📋 Vue d'Ensemble

ChirurgIA est un système d'intelligence artificielle développé pour révolutionner l'analyse prédictive en chirurgie générale. Le projet combine **Machine Learning avancé**, **traitement de langage naturel médical**, et **API professionnelle** pour fournir :

### 🎯 **Objectifs Principaux**
1. **🩺 Prédiction de Mortalité** - Évaluation du risque de décès post-opératoire
2. **💀 Analyse des Causes de Décès** - Identification des causes probables de mortalité
3. **👥 Clustering des Survivants** - Classification par niveau de criticité post-opératoire

### 🏆 **Performances Atteintes**
- **Prédiction Mortalité** : F1-Score 0.847, AUC-ROC 0.923
- **Analyse Causes** : 13 catégories de causes, 100% confiance
- **Clustering** : 7 clusters, score Silhouette 0.36
- **API Production** : 5/5 endpoints fonctionnels, < 1s de réponse

## 🏗️ Architecture du Projet

```
ChirurgIA/
├── 📊 data/                     # Données médicales
│   ├── chirurgical_data.csv     # Dataset principal (21,000+ patients)
│   └── processed/               # Données prétraitées
├── 📓 notebooks/                # Notebooks d'analyse (100% documentés)
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_mortality_prediction.ipynb
│   ├── 04_death_cause_analysis.ipynb
│   └── 05_survivors_clustering.ipynb
├── 🧠 models/                   # Modèles ML entraînés
│   ├── best_mortality_model_xgboost.pkl
│   ├── death_cause_classifier_improved.pkl
│   ├── survivors_clustering_model.pkl
│   └── *.json                   # Métadonnées et encodeurs
├── 🚀 app/                      # API FastAPI professionnelle
│   └── app.py                   # 5 endpoints de prédiction
├── 🔧 src/                      # Code source modulaire
│   ├── data_preprocessing.py
│   ├── models.py
│   └── predictor_api.py
├── 📚 docs/                     # Documentation complète
└── 🧪 test_api_complete.py      # Tests d'intégration
```

## 🚀 Installation et Démarrage Rapide

### Prérequis
- Python 3.10+
- 8GB RAM minimum
- Environnement Linux/macOS/Windows

### 1. Installation

```bash
# Cloner le projet
git clone https://github.com/VictoryKasende/ChirurgIA.git
cd ChirurgIA

# Créer et activer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou .\venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Installer le modèle médical ScispaCy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gz
```

### 2. Démarrage de l'API

```bash
# Lancer l'API FastAPI
python app/app.py

# L'API sera disponible sur http://localhost:8000
# Documentation automatique : http://localhost:8000/docs
```

### 3. Test de l'API

```bash
# Tester tous les endpoints
python test_api_complete.py
```

## 🔬 Fonctionnalités Avancées

### 📊 **Analyse de Données Médicales**
- **21,000+ patients** analysés
- **17 biomarqueurs** : Temperature, pH, pCO2, pO2, HCO3, BE, Lactate, etc.
- **Données textuelles** : Diagnostic, chirurgie, complications, investigations
- **Preprocessing médical** avec ScispaCy pour extraction d'entités

### 🧠 **Modèles d'Intelligence Artificielle**

#### 🩺 Prédiction de Mortalité (Notebook 03)
- **Modèle** : XGBoost optimisé avec SMOTE
- **Features** : 49 features (17 numériques + 32 dérivées)
- **Performance** : F1-Score 0.847, AUC-ROC 0.923, Sensibilité 84.7%
- **Innovation** : Seuil optimisé (0.35), gestion déséquilibre des classes

#### 💀 Analyse des Causes de Décès (Notebook 04)
- **Modèle** : Classifier amélioré avec vectorisation TF-IDF
- **Categories** : 13 causes (Sepsis, Cardiovasculaire, Cancer, etc.)
- **Features** : 18 features (17 numériques + 1 textuelle)
- **Innovation** : Catégorisation médicale affinée, SMOTE pour classes minoritaires

#### 👥 Clustering des Survivants (Notebook 05)
- **Modèle** : K-Means avec 7 clusters de criticité
- **Features** : 13 features spécialisées (biomarqueurs + scores composites)
- **Performance** : Score Silhouette 0.36, Calinski-Harabasz 2139
- **Innovation** : Scores de criticité post-opératoire (60%-90%)

### 🌐 **API FastAPI Professionnelle**

#### Endpoints Disponibles :

```python
# 1. Prédiction de Mortalité Complète
POST /predict/mortality
{
  "biomarkers": {...},    # 17 biomarqueurs
  "clinical_texts": {...} # Données textuelles
}
# Retourne : probabilité, niveau de risque, facteurs, recommandations

# 2. Prédiction Mortalité Simple  
POST /predict/mortality-simple
{...biomarkers only...}
# Retourne : prédiction rapide avec biomarqueurs uniquement

# 3. Analyse des Causes de Décès
POST /predict/death-cause
{...patient_data...}
# Retourne : cause prédite, top 3 causes, confiance, interprétation

# 4. Clustering des Survivants
POST /analyze/clustering  
{...patient_data...}
# Retourne : cluster, criticité, surveillance, recommandations

# 5. Analyse Complète Intégrée
POST /analyze/complete
{...patient_data...}
# Retourne : toutes les analyses combinées avec logique conditionnelle
```

#### Fonctionnalités API :
- **Validation Pydantic** stricte avec types médicaux
- **Gestion d'erreurs** robuste avec fallbacks
- **Documentation automatique** Swagger/OpenAPI
- **Logging** pour monitoring et debugging
- **CORS** configuré pour intégration frontend

## 📈 Méthodologie Scientifique

### 🔍 **Preprocessing Médical Avancé**
- **Imputation intelligente** des valeurs manquantes par domaine médical
- **Normalisation robuste** (RobustScaler) résistante aux outliers
- **Features engineering** : ratios cliniques, scores APACHE, indicateurs binaires
- **Extraction NLP** : entités médicales avec ScispaCy
- **Gestion déséquilibre** : SMOTE adapté au contexte médical

### 🎯 **Validation et Évaluation**
- **Cross-validation stratifiée** pour données médicales
- **Métriques cliniques** : Sensibilité, Spécificité, VPP, VPN
- **Courbes ROC/PR** pour évaluation seuils cliniques
- **Tests de robustesse** sur profils patients variés
- **Interprétabilité** avec SHAP/LIME pour explicabilité médicale

### 📊 **Gestion des Données**
- **Features numériques** : 17 biomarqueurs standardisés
- **Features textuelles** : 5 champs (Diagnostic, Chirurgie, Problèmes, etc.)
- **Features dérivées** : Ratios, scores composites, indicateurs de seuils
- **Architecture modulaire** : Chaque modèle avec ses features spécifiques

## 🧪 Tests et Validation

### Tests d'Intégration Complets
```bash
# Exécuter la suite de tests
python test_api_complete.py

# Résultats attendus :
✅ Mortality Prediction: 87.2% probabilité critique
✅ Death Cause Analysis: Cancer_Malignité (100% confiance)  
✅ Survivors Clustering: Cluster 1 - Récupération Difficile (70%)
✅ Complete Analysis: Toutes analyses intégrées
```

### Validation Clinique
- **Profils patients réalistes** testés
- **Cohérence des prédictions** inter-modèles
- **Recommandations cliniques** générées automatiquement
- **Facteurs de risque** identifiés précisément

## 📚 Documentation Technique

### Notebooks Détaillés
Chaque notebook contient :
- **Rapport final professionnel** avec métriques
- **Analyse exploratoire** approfondie  
- **Preprocessing documenté** étape par étape
- **Modélisation comparative** multi-algorithmes
- **Validation robuste** avec interprétation clinique

### Code Source
- **Architecture modulaire** dans `src/`
- **Functions réutilisables** pour preprocessing médical
- **Classes ML** avec méthodes d'évaluation intégrées
- **API professionnelle** avec gestion d'erreurs

## 🎯 Cas d'Usage Clinique

### Prédiction de Mortalité
```python
# Patient critique en post-opératoire
biomarkers = {
  "temperature": 38.5, "ph": 7.25, "lactate": 5.2,
  "creatinine": 2.1, "wcc": 18.5, ...
}
# → Prédiction : CRITIQUE (87.2% risque décès)
# → Recommandations : USI urgente, monitoring continu
```

### Analyse des Causes
```python
# Patient en détérioration
clinical_data = {
  "diagnosis": "Septic shock with organ failure",
  "problems": "Severe sepsis, bleeding", ...
}
# → Cause probable : Sepsis_Infection (85% confiance)
# → Actions : Antibiotiques, contrôle hémorragie
```

### Clustering Post-Op
```python
# Survivant en récupération
# → Cluster : Récupération Difficile (70% criticité)
# → Surveillance : Intensive 24h
# → Patients similaires : 7,635 cas analysés
```

## 🏆 Impact et Innovation

### Avancées Techniques
- **Pipeline ML médical** complet de bout en bout
- **API production-ready** avec FastAPI moderne
- **Gestion multi-features** (17/18/49) par modèle
- **Validation Pydantic** avec types médicaux spécialisés
- **Architecture scalable** pour déploiement hospitalier

### Valeur Clinique
- **Aide à la décision** médicale basée sur l'IA
- **Prédiction précoce** des complications post-opératoires  
- **Optimisation des ressources** hospitalières
- **Amélioration des outcomes** patients par anticipation

### Performance Opérationnelle
- **Temps de réponse** < 1 seconde par prédiction
- **Disponibilité** 24/7 via API REST
- **Intégration facile** dans systèmes hospitaliers existants
- **Monitoring** et logging pour maintenance

## 🔧 Configuration et Déploiement

### Variables d'Environnement
```bash
export CHIRURGIA_MODEL_PATH="/path/to/models"
export CHIRURGIA_LOG_LEVEL="INFO"
export CHIRURGIA_API_PORT="8000"
```

### Déploiement Production
```bash
# Avec Docker (recommandé)
docker build -t chirurgia-api .
docker run -p 8000:8000 chirurgia-api

# Avec Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.app:app --bind 0.0.0.0:8000
```

### Monitoring
- **Logs structurés** pour debugging
- **Métriques de performance** par endpoint  
- **Health checks** automatiques
- **Alertes** sur erreurs critiques

## 👥 Équipe et Contributions

### Développement Principal
- **Architecture ML** : Design et implémentation des modèles
- **API Development** : FastAPI professionnelle avec validation
- **Data Science** : Preprocessing médical et feature engineering
- **Documentation** : Notebooks et rapports techniques complets

### Technologies Utilisées
- **Machine Learning** : scikit-learn, XGBoost, imbalanced-learn
- **NLP Médical** : ScispaCy, NLTK, spaCy
- **API Framework** : FastAPI, Pydantic, Uvicorn
- **Data Processing** : pandas, numpy, matplotlib, seaborn
- **Deployment** : Docker, Gunicorn, GitHub Actions

## 📄 Licence et Usage

Ce projet est développé à des fins **éducatives et de recherche**. Pour un usage clinique réel, une validation supplémentaire et une certification médicale seraient nécessaires.

**⚠️ Avertissement Médical** : Ce système est un outil d'aide à la décision et ne remplace pas le jugement clinique professionnel.

---

<div align="center">

**🏥 ChirurgIA - Révolutionner la Chirurgie par l'Intelligence Artificielle**

*Développé avec ❤️ pour améliorer les soins chirurgicaux*

[![GitHub](https://img.shields.io/badge/GitHub-ChirurgIA-181717?style=flat&logo=github)](https://github.com/VictoryKasende/ChirurgIA)
[![Contact](https://img.shields.io/badge/Contact-victorykasende%40egmail.com-blue?style=flat&logo=gmail)](mailto:victorykasende@gmail.com)

</div>
