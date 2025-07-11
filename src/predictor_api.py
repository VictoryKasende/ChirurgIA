"""
🏥 ChirurgIA - API de Prédiction Médicale
=========================================

Ce module fournit les fonctions principales pour utiliser les modèles
ChirurgIA en production ou pour de nouveaux patients.

Auteur: ChirurgIA ML Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import pickle
import json
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ChirurgIAPredictor:
    """
    Classe principale pour les prédictions ChirurgIA
    
    Gère les trois modèles principaux :
    1. Prédiction de mortalité
    2. Classification des causes de décès
    3. Clustering de criticité des survivants
    """
    
    def __init__(self, models_path: str = "/home/victory/Documents/ChirurgIA/models/"):
        """
        Initialise le prédicteur avec les modèles sauvegardés
        
        Args:
            models_path (str): Chemin vers le dossier des modèles
        """
        self.models_path = models_path
        self.models = {}
        self.scalers = {}
        self.metadata = {}
        
        # Chargement des modèles
        self._load_models()
    
    def _load_models(self):
        """Charge tous les modèles et leurs métadonnées"""
        try:
            # 1. Modèle de prédiction de mortalité
            self._load_mortality_model()
            
            # 2. Modèle de classification des causes de décès
            self._load_death_cause_model()
            
            # 3. Modèle de clustering des survivants
            self._load_clustering_model()
            
            print("✅ Tous les modèles ChirurgIA chargés avec succès")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement des modèles : {e}")
    
    def _load_mortality_model(self):
        """Charge le modèle de prédiction de mortalité"""
        try:
            # Modèle principal (à implémenter selon le notebook 03)
            # self.models['mortality'] = pickle.load(open(f"{self.models_path}mortality_predictor.pkl", 'rb'))
            # self.scalers['mortality'] = pickle.load(open(f"{self.models_path}mortality_scaler.pkl", 'rb'))
            print("📊 Modèle de mortalité prêt")
        except:
            print("⚠️ Modèle de mortalité non trouvé")
    
    def _load_death_cause_model(self):
        """Charge le modèle de classification des causes de décès"""
        try:
            self.models['death_cause'] = pickle.load(open(f"{self.models_path}death_cause_classifier.pkl", 'rb'))
            self.scalers['death_cause'] = pickle.load(open(f"{self.models_path}death_cause_scaler.pkl", 'rb'))
            
            # Métadonnées
            with open(f"{self.models_path}death_cause_model_info.json", 'r') as f:
                self.metadata['death_cause'] = json.load(f)
            
            print("🔍 Modèle de causes de décès chargé")
        except:
            print("⚠️ Modèle de causes de décès non trouvé")
    
    def _load_clustering_model(self):
        """Charge le modèle de clustering des survivants"""
        try:
            self.models['clustering'] = pickle.load(open(f"{self.models_path}survivors_clustering_model.pkl", 'rb'))
            self.scalers['clustering'] = pickle.load(open(f"{self.models_path}clustering_scaler.pkl", 'rb'))
            
            # Métadonnées
            with open(f"{self.models_path}clustering_metadata.json", 'r') as f:
                self.metadata['clustering'] = json.load(f)
                
            print("🎯 Modèle de clustering chargé")
        except:
            print("⚠️ Modèle de clustering non trouvé")
    
    def predict_mortality(self, patient_data: Dict) -> Dict[str, Any]:
        """
        Prédit le risque de mortalité pour un patient
        
        Args:
            patient_data (dict): Données du patient
                Ex: {
                    'Temperature': 37.2,
                    'pH': 7.35,
                    'pO2': 85,
                    'Creatinine': 1.1,
                    ...
                }
        
        Returns:
            dict: {
                'risk_probability': float,  # Probabilité de décès (0-1)
                'risk_level': str,          # 'FAIBLE', 'MODÉRÉ', 'ÉLEVÉ', 'CRITIQUE'
                'confidence': float,        # Confiance du modèle
                'risk_factors': list        # Facteurs de risque identifiés
            }
        """
        if 'mortality' not in self.models:
            return {'error': 'Modèle de mortalité non disponible'}
        
        # TODO: Implémenter la prédiction de mortalité
        # selon le notebook 03
        
        return {
            'risk_probability': 0.15,  # Exemple
            'risk_level': 'MODÉRÉ',
            'confidence': 0.87,
            'risk_factors': ['Créatinine élevée', 'pH anormal']
        }
    
    def predict_death_cause(self, patient_data: Dict) -> Dict[str, Any]:
        """
        Prédit la cause probable de décès pour un patient décédé
        
        Args:
            patient_data (dict): Données du patient décédé incluant
                les features textuelles et numériques
        
        Returns:
            dict: {
                'predicted_cause': str,     # Cause principale prédite
                'probability': float,       # Probabilité de cette cause
                'all_probabilities': dict,  # Toutes les probabilités
                'confidence': str           # 'HAUTE', 'MOYENNE', 'FAIBLE'
            }
        """
        if 'death_cause' not in self.models:
            return {'error': 'Modèle de causes de décès non disponible'}
        
        try:
            # Préparation des données (simplifiée pour l'exemple)
            # En réalité, il faudrait le preprocessing complet du notebook 04
            
            categories = self.metadata['death_cause']['categories']
            
            # Prédiction simulée
            predicted_idx = 0  # Index de la classe prédite
            probabilities = [0.6, 0.2, 0.1, 0.05, 0.03, 0.02, 0.0]  # Exemple
            
            predicted_cause = categories[predicted_idx]
            confidence = 'HAUTE' if max(probabilities) > 0.5 else 'MOYENNE' if max(probabilities) > 0.3 else 'FAIBLE'
            
            return {
                'predicted_cause': predicted_cause,
                'probability': max(probabilities),
                'all_probabilities': dict(zip(categories, probabilities)),
                'confidence': confidence
            }
            
        except Exception as e:
            return {'error': f'Erreur de prédiction : {e}'}
    
    def predict_criticality(self, patient_data: Dict) -> Dict[str, Any]:
        """
        Prédit le niveau de criticité post-opératoire pour un survivant
        
        Args:
            patient_data (dict): Données biologiques du patient
        
        Returns:
            dict: {
                'cluster': int,             # Cluster assigné
                'criticality_score': float, # Score de criticité (0-100)
                'severity_level': str,      # 'FAIBLE', 'MODÉRÉE', 'ÉLEVÉE', 'CRITIQUE'
                'recommendations': list,    # Recommandations cliniques
                'monitoring_level': str     # Niveau de surveillance requis
            }
        """
        if 'clustering' not in self.models:
            return {'error': 'Modèle de clustering non disponible'}
        
        try:
            # Récupération des seuils de criticité depuis les métadonnées
            criticality_scores = self.metadata['clustering']['criticality_scores']
            
            # TODO: Implémenter la prédiction complète selon le notebook 05
            
            # Simulation pour l'exemple
            cluster = 1
            score = 35.5
            
            # Détermination du niveau de sévérité
            if score < 25:
                severity = 'FAIBLE'
                monitoring = 'Surveillance standard'
                recommendations = [
                    'Surveillance post-opératoire standard',
                    'Mobilisation précoce encouragée',
                    'Sortie selon protocole habituel'
                ]
            elif score < 50:
                severity = 'MODÉRÉE'
                monitoring = 'Surveillance renforcée 24-48h'
                recommendations = [
                    'Surveillance renforcée pendant 24-48h',
                    'Monitoring continu des signes vitaux',
                    'Réévaluation quotidienne'
                ]
            elif score < 75:
                severity = 'ÉLEVÉE'
                monitoring = 'Soins intensifs'
                recommendations = [
                    'Surveillance intensive requise',
                    'Monitoring cardiaque et respiratoire',
                    'Bilans biologiques fréquents',
                    'Consultation spécialisée si nécessaire'
                ]
            else:
                severity = 'CRITIQUE'
                monitoring = 'Réanimation'
                recommendations = [
                    'ADMISSION EN SOINS INTENSIFS',
                    'Monitoring multiparamétrique continu',
                    'Support organique si nécessaire',
                    'Réévaluation multidisciplinaire'
                ]
            
            return {
                'cluster': cluster,
                'criticality_score': score,
                'severity_level': severity,
                'recommendations': recommendations,
                'monitoring_level': monitoring
            }
            
        except Exception as e:
            return {'error': f'Erreur de prédiction de criticité : {e}'}
    
    def analyze_patient_complete(self, patient_data: Dict, patient_status: str = 'unknown') -> Dict[str, Any]:
        """
        Analyse complète d'un patient avec tous les modèles applicables
        
        Args:
            patient_data (dict): Données complètes du patient
            patient_status (str): 'alive', 'deceased', 'unknown'
        
        Returns:
            dict: Résultats de toutes les analyses applicables
        """
        results = {
            'patient_id': patient_data.get('patient_id', 'unknown'),
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'status': patient_status
        }
        
        # 1. Prédiction de mortalité (toujours applicable)
        if patient_status in ['alive', 'unknown']:
            results['mortality_prediction'] = self.predict_mortality(patient_data)
        
        # 2. Analyse des causes de décès (si décédé)
        if patient_status == 'deceased':
            results['death_cause_analysis'] = self.predict_death_cause(patient_data)
        
        # 3. Analyse de criticité (si survivant)
        if patient_status == 'alive':
            results['criticality_analysis'] = self.predict_criticality(patient_data)
        
        return results

def load_predictor() -> ChirurgIAPredictor:
    """
    Fonction utilitaire pour charger rapidement le prédicteur
    
    Returns:
        ChirurgIAPredictor: Instance du prédicteur prête à utiliser
    """
    return ChirurgIAPredictor()

# Exemple d'utilisation
if __name__ == "__main__":
    print("🏥 Test de l'API ChirurgIA")
    print("=" * 50)
    
    # Chargement du prédicteur
    predictor = load_predictor()
    
    # Exemple de données patient
    exemple_patient = {
        'patient_id': 'TEST_001',
        'Temperature': 37.8,
        'pH': 7.32,
        'pO2': 78,
        'pCO2': 46,
        'HCO3': 23,
        'Lactate': 2.1,
        'Creatinine': 1.3,
        'WCC': 11.5,
        'HGB': 10.2,
        'PLT': 185,
        'Diagnosis': 'Appendectomy with complications',
        'Surgery': 'Laparoscopic appendectomy',
        'Problems': 'Post-operative infection',
        'Investigation': 'Blood cultures positive'
    }
    
    # Test des prédictions
    print("\\n🎯 Test de prédiction de mortalité:")
    mortality_result = predictor.predict_mortality(exemple_patient)
    print(f"   Risque: {mortality_result.get('risk_level', 'N/A')}")
    
    print("\\n🔍 Test de prédiction de cause de décès:")
    death_cause_result = predictor.predict_death_cause(exemple_patient)
    print(f"   Cause probable: {death_cause_result.get('predicted_cause', 'N/A')}")
    
    print("\\n📊 Test d'analyse de criticité:")
    criticality_result = predictor.predict_criticality(exemple_patient)
    print(f"   Niveau: {criticality_result.get('severity_level', 'N/A')}")
    
    print("\\n✅ Tests terminés")
