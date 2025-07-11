<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Instructions Copilot pour ChirurgIA

## Contexte du Projet
Ce projet de Machine Learning analyse des données médicales de chirurgie générale avec trois objectifs :
1. Déterminer les causes de décès probables
2. Prédire les cas de décès après admission/chirurgie  
3. Grouper les survivants selon la criticité post-opératoire

## Données Disponibles
- **Attributs numériques** : Signes vitaux (Temperature, pH, pCO2, pO2, HCO3, BE, Lactate, Na, K, Cl, Urea, Creatinine, HGT, WCC, HGB, PLT, INR, ABG, U+E, FBC)
- **Attributs textuels** : Diagnosis, Surgery, Problems, Investigation
- **Attributs catégoriels** : Âge, race, date d'admission, Outcome, causeofDeath

## Bibliothèques Spécialisées
- **ScispaCy** : Pour l'extraction d'entités médicales des textes
- **Imbalanced-learn** : Pour gérer les classes déséquilibrées
- **SHAP/LIME** : Pour l'interprétabilité des modèles

## Bonnes Pratiques à Suivre
1. **Préprocessing médical** : Utiliser ScispaCy pour extraire les termes médicaux
2. **Gestion des valeurs manquantes** : Stratégies spécifiques au domaine médical
3. **Classes déséquilibrées** : Techniques de rééchantillonnage appropriées
4. **Validation** : Cross-validation stratifiée pour les données médicales
5. **Interprétabilité** : Les modèles doivent être explicables pour usage médical

## Structure de Code Préférée
- Code modulaire dans `src/`
- Notebooks pour exploration dans `notebooks/`
- Fonctions réutilisables pour le preprocessing médical
- Classes pour les modèles ML avec méthodes d'évaluation intégrées
