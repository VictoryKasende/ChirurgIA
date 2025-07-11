# ChirurgIA - Guide de Démarrage Rapide

## 🚀 Pour commencer immédiatement :

### 1. **Configurez l'environnement Python** (PRIORITÉ 1)
```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt

# Installer ScispaCy pour le NLP médical
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gz
```

### 2. **Ajoutez vos données** (PRIORITÉ 2)
- Placez votre fichier CSV dans le dossier `data/`
- Nommez-le `chirurgical_data.csv` ou modifiez le chemin dans les notebooks

### 3. **Commencez l'exploration** ✅ **TERMINÉ**
```bash
# Ouvrir le notebook principal
jupyter notebook notebooks/01_data_exploration.ipynb
```
**✅ Exploration terminée ! Résultats clés :**
- 21,997 patients, 73 variables  
- Taux de mortalité : 4.76%
- Signes vitaux et données textuelles riches
- Tous les 3 objectifs sont réalisables

### 4. **Prochaine étape : Preprocessing** (PRIORITÉ ACTUELLE)
```bash
# Créer et ouvrir le notebook de preprocessing
jupyter notebook notebooks/02_data_preprocessing.ipynb
```
```bash
cd app/
streamlit run app.py
```

## 📋 Prochaines étapes recommandées :

### Phase 1 : Exploration (Semaine 1-2)
- [ ] Charger et explorer vos vraies données
- [ ] Comprendre la distribution des variables
- [ ] Identifier les valeurs manquantes et aberrantes
- [ ] Analyser la balance des classes (survivants vs décès)

### Phase 2 : Préparation (Semaine 2-3)
- [ ] Nettoyer les données numériques (signes vitaux)
- [ ] Traiter les textes médicaux avec ScispaCy
- [ ] Gérer les valeurs manquantes
- [ ] Créer de nouvelles features

### Phase 3 : Modélisation (Semaine 3-4)
- [ ] **Objectif 1** : Modèle de prédiction de mortalité
- [ ] **Objectif 2** : Analyse des causes de décès
- [ ] **Objectif 3** : Clustering des survivants

### Phase 4 : Évaluation et Déploiement (Semaine 4-5)
- [ ] Évaluer les performances des modèles
- [ ] Interpréter les résultats (SHAP, LIME)
- [ ] Finaliser l'application
- [ ] Rédiger l'article

## 🔧 Structure de travail recommandée :

1. **notebooks/** : Pour l'exploration et le prototypage
2. **src/** : Code modulaire réutilisable
3. **models/** : Modèles entraînés sauvegardés
4. **app/** : Application de démonstration
5. **docs/** : Documentation et article final

## ⚠️ Points d'attention :

- **Données sensibles** : Respectez la confidentialité médicale
- **Validation** : Utilisez la validation croisée stratifiée
- **Interprétabilité** : Les modèles médicaux doivent être explicables
- **Classes déséquilibrées** : Utilisez SMOTE ou des métriques adaptées

## 🆘 En cas de problème :

1. **Erreur ScispaCy** : Vérifiez l'installation du modèle médical
2. **Dépendances manquantes** : Réinstallez requirements.txt
3. **Données introuvables** : Vérifiez le chemin dans les notebooks
4. **Performance lente** : Réduisez la taille des données pour les tests

---

**Bonne chance avec votre projet ChirurgIA ! 🏥**
