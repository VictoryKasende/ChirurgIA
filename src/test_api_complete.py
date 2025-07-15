#!/usr/bin/env python3
"""
Test de l'API ChirurgIA avec données complètes
Démontre l'utilisation des biomarqueurs + données textuelles
"""

import requests
import json

API_BASE = "http://localhost:8000"

def test_mortality_prediction_complete():
    """Test de prédiction de mortalité avec données complètes"""
    
    # Données patient complètes (biomarqueurs + textes cliniques)
    patient_data = {
        "biomarkers": {
            "temperature": 38.5,
            "ph": 7.25,
            "pco2": 45.0,
            "po2": 85.0,
            "hco3": 18.0,
            "be": -6.0,
            "lactate": 5.2,
            "na": 142.0,
            "k": 4.1,
            "cl": 105.0,
            "urea": 8.5,
            "creatinine": 2.1,
            "hgt": 180.0,
            "wcc": 18.5,
            "hgb": 95.0,
            "plt": 125.0,
            "inr": 1.8
        },
        "clinical_texts": {
            "diagnosis": "Acute appendicitis with severe complications and sepsis",
            "surgery": "Emergency laparoscopic appendectomy with drainage",
            "problems": "Post-operative infection, bleeding, sepsis development",
            "investigations": "CT scan showing perforation, blood cultures positive",
            "clinical_course": "Patient admitted in emergency, ICU transfer required"
        }
    }
    
    print("🔍 Test de prédiction de mortalité complète...")
    response = requests.post(f"{API_BASE}/predict/mortality", json=patient_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Prédiction: {result['prediction']}")
        print(f"📊 Probabilité de décès: {result['probability_death']:.1%}")
        print(f"📈 Niveau de risque: {result['risk_level']}")
        print(f"⚠️ Facteurs de risque: {', '.join(result['risk_factors'])}")
        print(f"💡 Recommandations: {len(result['recommendations'])} recommandations")
        return result
    else:
        print(f"❌ Erreur: {response.status_code} - {response.text}")
        return None

def test_mortality_prediction_simple():
    """Test de prédiction de mortalité avec biomarqueurs uniquement"""
    
    # Données biomarqueurs uniquement
    biomarkers_data = {
        "temperature": 37.2,
        "ph": 7.38,
        "pco2": 42.0,
        "po2": 95.0,
        "hco3": 24.0,
        "be": 1.0,
        "lactate": 1.8,
        "na": 140.0,
        "k": 4.0,
        "cl": 102.0,
        "urea": 5.2,
        "creatinine": 0.9,
        "hgt": 110.0,
        "wcc": 7.5,
        "hgb": 135.0,
        "plt": 280.0,
        "inr": 1.1
    }
    
    print("\n🔍 Test de prédiction de mortalité simple...")
    response = requests.post(f"{API_BASE}/predict/mortality-simple", json=biomarkers_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Prédiction: {result['prediction']}")
        print(f"📊 Probabilité de survie: {result['probability_survival']:.1%}")
        print(f"📈 Niveau de risque: {result['risk_level']}")
        return result
    else:
        print(f"❌ Erreur: {response.status_code} - {response.text}")
        return None

def test_death_cause_prediction():
    """Test de prédiction des causes de décès"""
    
    # Patient à haut risque avec contexte clinique
    patient_data = {
        "biomarkers": {
            "temperature": 39.1,
            "ph": 7.15,
            "pco2": 50.0,
            "po2": 65.0,
            "hco3": 15.0,
            "be": -10.0,
            "lactate": 8.5,
            "na": 148.0,
            "k": 5.2,
            "cl": 108.0,
            "urea": 15.0,
            "creatinine": 3.5,
            "hgt": 220.0,
            "wcc": 25.0,
            "hgb": 75.0,
            "plt": 85.0,
            "inr": 2.8
        },
        "clinical_texts": {
            "diagnosis": "Septic shock secondary to pneumonia with multiple organ failure",
            "surgery": "Emergency exploratory surgery",
            "problems": "Severe sepsis, respiratory failure, renal dysfunction, bleeding",
            "investigations": "Chest X-ray pneumonia, lactate elevated, blood cultures positive",
            "clinical_course": "Rapid deterioration, ICU admission, mechanical ventilation"
        }
    }
    
    print("\n🧬 Test de prédiction des causes de décès...")
    response = requests.post(f"{API_BASE}/predict/death-cause", json=patient_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"💀 Cause prédite: {result['predicted_cause']}")
        print(f"📊 Confiance: {result['confidence']:.1%}")
        print("🔝 Top 3 causes:")
        for i, cause in enumerate(result['top_3_causes'], 1):
            print(f"   {i}. {cause['cause']}: {cause['probability']:.1%}")
        print(f"🏥 Interprétation: {result['clinical_interpretation']}")
        return result
    else:
        print(f"❌ Erreur: {response.status_code} - {response.text}")
        return None

def test_clustering_analysis():
    """Test de clustering des survivants"""
    
    # Patient survivant pour clustering
    patient_data = {
        "biomarkers": {
            "temperature": 37.8,
            "ph": 7.35,
            "pco2": 44.0,
            "po2": 88.0,
            "hco3": 22.0,
            "be": -2.0,
            "lactate": 3.1,
            "na": 141.0,
            "k": 4.2,
            "cl": 103.0,
            "urea": 6.8,
            "creatinine": 1.4,
            "hgt": 140.0,
            "wcc": 12.0,
            "hgb": 105.0,
            "plt": 180.0,
            "inr": 1.4
        },
        "clinical_texts": {
            "diagnosis": "Cholecystitis with mild complications",
            "surgery": "Elective laparoscopic cholecystectomy",
            "problems": "Minor post-operative pain, no major complications",
            "investigations": "Ultrasound confirming gallstones, routine blood work",
            "clinical_course": "Stable recovery, good pain control"
        }
    }
    
    print("\n👥 Test de clustering des survivants...")
    response = requests.post(f"{API_BASE}/analyze/clustering", json=patient_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"🎯 Cluster: {result['cluster']} - {result['cluster_name']}")
        print(f"📊 Score de criticité: {result['criticality_score']:.1f}%")
        print(f"⚠️ Niveau de sévérité: {result['severity_level']}")
        print(f"👨‍⚕️ Surveillance: {result['monitoring_level']}")
        print(f"👥 Patients similaires: {result['similar_patients_count']}")
        print(f"💡 Recommandations: {len(result['recommendations'])} recommandations")
        return result
    else:
        print(f"❌ Erreur: {response.status_code} - {response.text}")
        return None

def test_complete_analysis():
    """Test d'analyse complète"""
    
    # Patient avec profil mixte
    patient_data = {
        "biomarkers": {
            "temperature": 38.2,
            "ph": 7.32,
            "pco2": 46.0,
            "po2": 82.0,
            "hco3": 20.0,
            "be": -3.0,
            "lactate": 3.8,
            "na": 143.0,
            "k": 4.3,
            "cl": 106.0,
            "urea": 7.2,
            "creatinine": 1.6,
            "hgt": 165.0,
            "wcc": 14.5,
            "hgb": 110.0,
            "plt": 200.0,
            "inr": 1.5
        },
        "clinical_texts": {
            "diagnosis": "Perforated diverticulitis with peritonitis",
            "surgery": "Emergency sigmoid resection with primary anastomosis",
            "problems": "Localized infection, moderate inflammatory response",
            "investigations": "CT scan showing perforation and fluid collection",
            "clinical_course": "Challenging surgery, post-op monitoring required"
        }
    }
    
    print("\n🔬 Test d'analyse complète...")
    response = requests.post(f"{API_BASE}/analyze/complete", json=patient_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"🆔 Patient ID: {result['patient_id']}")
        
        if result['mortality_prediction']:
            mort = result['mortality_prediction']
            print(f"💀 Mortalité: {mort['prediction']} ({mort['probability_death']:.1%})")
        
        if result['death_cause_analysis']:
            cause = result['death_cause_analysis']
            print(f"🧬 Cause probable: {cause['predicted_cause']}")
        
        if result['clustering_analysis']:
            cluster = result['clustering_analysis']
            print(f"👥 Cluster: {cluster['cluster_name']} ({cluster['criticality_score']:.1f}%)")
        
        return result
    else:
        print(f"❌ Erreur: {response.status_code} - {response.text}")
        return None

if __name__ == "__main__":
    print("🏥 Test de l'API ChirurgIA avec données complètes")
    print("=" * 60)
    
    # Tests séquentiels
    test_mortality_prediction_complete()
    test_mortality_prediction_simple()
    test_death_cause_prediction()
    test_clustering_analysis()
    test_complete_analysis()
    
    print("\n✅ Tests terminés!")
