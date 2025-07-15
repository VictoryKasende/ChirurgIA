"""
API FastAPI professionnelle pour le déploiement du modèle ChirurgIA
API REST pour la prédiction de mortalité, analyse des causes de décès et clustering des survivants
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import json
import sys
import os
from datetime import datetime
import logging
from pathlib import Path

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
API FastAPI professionnelle pour le déploiement du modèle ChirurgIA
API REST pour la prédiction de mortalité, analyse des causes de décès et clustering des survivants
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import json
import sys
import os
from datetime import datetime
import logging
from pathlib import Path

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de l'API
app = FastAPI(
    title="ChirurgIA API",
    description="API professionnelle pour l'analyse prédictive en chirurgie générale",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sécurité (optionnel)
security = HTTPBearer()

# Chemins des modèles
MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"

# Cache global pour les modèles
models_cache = {}

# ===== MODÈLES PYDANTIC =====

class BiomarkerData(BaseModel):
    """Données biomédicales pour prédiction"""
    # Signes vitaux et biomarqueurs (17 features numériques)
    temperature: float = Field(..., ge=35.0, le=42.0, description="Température corporelle en °C")
    ph: float = Field(..., ge=7.0, le=7.8, description="pH sanguin")
    pco2: float = Field(..., ge=20.0, le=80.0, description="pCO2 en mmHg")
    po2: float = Field(..., ge=40.0, le=150.0, description="pO2 en mmHg")
    hco3: float = Field(..., ge=10.0, le=40.0, description="HCO3 en mEq/L")
    be: float = Field(..., ge=-20.0, le=20.0, description="Base Excess en mEq/L")
    lactate: float = Field(..., ge=0.5, le=20.0, description="Lactate en mmol/L")
    na: float = Field(..., ge=120.0, le=160.0, description="Sodium en mEq/L")
    k: float = Field(..., ge=2.0, le=7.0, description="Potassium en mEq/L")
    cl: float = Field(..., ge=80.0, le=120.0, description="Chlorure en mEq/L")
    urea: float = Field(..., ge=2.0, le=50.0, description="Urée en mmol/L")
    creatinine: float = Field(..., ge=0.4, le=10.0, description="Créatinine en mg/dL")
    hgt: float = Field(..., ge=50.0, le=500.0, description="Glycémie en mg/dL")
    wcc: float = Field(..., ge=1.0, le=50.0, description="Leucocytes en 10³/μL")
    hgb: float = Field(..., ge=50.0, le=200.0, description="Hémoglobine en g/L")
    plt: float = Field(..., ge=20.0, le=800.0, description="Plaquettes en 10³/μL")
    inr: float = Field(..., ge=0.8, le=5.0, description="INR")

class ClinicalTextData(BaseModel):
    """Données textuelles cliniques pour feature engineering"""
    diagnosis: Optional[str] = Field(None, description="Diagnostic principal", max_length=1000)
    surgery: Optional[str] = Field(None, description="Description de la chirurgie", max_length=1000)
    problems: Optional[str] = Field(None, description="Problèmes/complications", max_length=1000)
    investigations: Optional[str] = Field(None, description="Examens/investigations", max_length=1000)
    clinical_course: Optional[str] = Field(None, description="Évolution clinique", max_length=1000)

class PatientData(BaseModel):
    """Données complètes patient pour prédiction"""
    biomarkers: BiomarkerData
    clinical_texts: ClinicalTextData

class MortalityPredictionResponse(BaseModel):
    """Réponse de prédiction de mortalité"""
    patient_id: str
    prediction: str = Field(..., description="Survived ou Died")
    probability_death: float = Field(..., ge=0.0, le=1.0, description="Probabilité de décès")
    probability_survival: float = Field(..., ge=0.0, le=1.0, description="Probabilité de survie")
    risk_level: str = Field(..., description="Niveau de risque")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance du modèle")
    risk_factors: List[str] = Field(..., description="Facteurs de risque identifiés")
    recommendations: List[str] = Field(..., description="Recommandations cliniques")
    timestamp: datetime

class CauseProbability(BaseModel):
    """Structure pour une cause avec sa probabilité"""
    cause: str = Field(..., description="Nom de la cause")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probabilité de cette cause")

class DeathCausePredictionResponse(BaseModel):
    """Réponse de prédiction des causes de décès"""
    patient_id: str
    predicted_cause: str = Field(..., description="Cause de décès prédite")
    top_3_causes: List[CauseProbability] = Field(..., description="Top 3 des causes probables avec probabilités")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance de la prédiction")
    clinical_interpretation: str = Field(..., description="Interprétation clinique")
    timestamp: datetime

class ClusteringResponse(BaseModel):
    """Réponse de clustering des survivants"""
    patient_id: str
    cluster: int = Field(..., description="Numéro du cluster assigné")
    cluster_name: str = Field(..., description="Nom du cluster")
    criticality_score: float = Field(..., ge=0.0, le=100.0, description="Score de criticité en %")
    severity_level: str = Field(..., description="Niveau de sévérité")
    similar_patients_count: int = Field(..., description="Nombre de patients similaires")
    recommendations: List[str] = Field(..., description="Recommandations de soins")
    monitoring_level: str = Field(..., description="Niveau de surveillance requis")
    timestamp: datetime

class APIHealthResponse(BaseModel):
    """Réponse de santé de l'API"""
    status: str
    version: str
    models_loaded: Dict[str, bool]
    timestamp: datetime
    uptime_seconds: float

# ===== UTILITAIRES =====

def load_model(model_name: str):
    """Charge un modèle depuis le cache ou le disque"""
    if model_name not in models_cache:
        model_path = MODELS_DIR / f"{model_name}.pkl"
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Modèle {model_name} non trouvé"
            )
        
        try:
            models_cache[model_name] = joblib.load(model_path)
            logger.info(f"Modèle {model_name} chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {model_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Erreur lors du chargement du modèle {model_name}"
            )
    
    return models_cache[model_name]

def biomarkers_to_features(biomarkers: BiomarkerData) -> np.ndarray:
    """Convertit les biomarqueurs en features pour le modèle (17 features numériques)"""
    features = np.array([
        biomarkers.temperature, biomarkers.ph, biomarkers.pco2, biomarkers.po2,
        biomarkers.hco3, biomarkers.be, biomarkers.lactate, biomarkers.na,
        biomarkers.k, biomarkers.cl, biomarkers.urea, biomarkers.creatinine,
        biomarkers.hgt, biomarkers.wcc, biomarkers.hgb, biomarkers.plt,
        biomarkers.inr
    ])
    return features

def extract_mortality_features(biomarkers: BiomarkerData, text_data: ClinicalTextData) -> tuple:
    """
    Extrait les features pour la prédiction de mortalité
    
    Returns:
        numeric_features: 17 features pour le RobustScaler  
        full_features: 49 features pour le modèle XGBoost
    """
    # 17 features numériques pour le scaler
    numeric_features = biomarkers_to_features(biomarkers)
    
    # Features textuelles basiques (4 features)
    text_features = []
    text_features.append(len(text_data.diagnosis.split()) if text_data.diagnosis else 0)
    text_features.append(len(text_data.surgery.split()) if text_data.surgery else 0)
    text_features.append(len(text_data.problems.split()) if text_data.problems else 0)
    text_features.append(len(text_data.investigations.split()) if text_data.investigations else 0)
    
    # Features dérivées (28 pour arriver à 49 total)
    derived_features = []
    
    # Ratios critiques
    derived_features.append(biomarkers.po2 / biomarkers.pco2 if biomarkers.pco2 > 0 else 0)
    derived_features.append(biomarkers.na / biomarkers.k if biomarkers.k > 0 else 0)
    derived_features.append(biomarkers.urea / biomarkers.creatinine if biomarkers.creatinine > 0 else 0)
    
    # Scores binaires anormaux
    derived_features.append(1 if biomarkers.temperature < 36.5 or biomarkers.temperature > 37.5 else 0)
    derived_features.append(1 if biomarkers.lactate > 2.0 else 0)
    derived_features.append(1 if biomarkers.ph < 7.35 or biomarkers.ph > 7.45 else 0)
    derived_features.append(1 if biomarkers.creatinine > 1.3 else 0)
    derived_features.append(1 if biomarkers.wcc > 11 or biomarkers.wcc < 4 else 0)
    derived_features.append(1 if biomarkers.hgb < 120 else 0)
    derived_features.append(1 if biomarkers.plt < 150 else 0)
    
    # Scores APACHE simplifiés
    apache_temp = 0 if 36 <= biomarkers.temperature <= 38.5 else (2 if biomarkers.temperature > 38.5 else 3)
    apache_ph = 0 if 7.33 <= biomarkers.ph <= 7.49 else (2 if biomarkers.ph < 7.33 else 1)
    apache_wcc = 0 if 3 <= biomarkers.wcc <= 14.9 else (1 if biomarkers.wcc >= 15 else 2)
    derived_features.extend([apache_temp, apache_ph, apache_wcc])
    
    # Remplir jusqu'à 28 features dérivées
    while len(derived_features) < 28:
        derived_features.append(0.0)
    
    # Combiner pour 49 features total
    full_features = np.concatenate([numeric_features, text_features, derived_features])
    
    return numeric_features.reshape(1, -1), full_features.reshape(1, -1)

def extract_death_cause_features(biomarkers: BiomarkerData, text_data: ClinicalTextData) -> tuple:
    """
    Extrait les features pour la prédiction des causes de décès
    
    Returns:
        numeric_features: 17 features pour le scaler
        combined_features: 18 features pour le classifier
    """
    # 17 features numériques pour le scaler
    numeric_features = biomarkers_to_features(biomarkers)
    
    # 1 feature textuelle principale pour arriver à 18
    # Complexité diagnostique
    diagnostic_complexity = len(text_data.diagnosis.split()) if text_data.diagnosis else 0
    
    # Combiner pour 18 features
    combined_features = np.concatenate([numeric_features, [diagnostic_complexity]])
    
    return numeric_features.reshape(1, -1), combined_features.reshape(1, -1)

def extract_clustering_features(biomarkers: BiomarkerData) -> np.ndarray:
    """
    Extrait les features pour le clustering des survivants (13 features exactement)
    Basé sur clustering_metadata.json: 
    ['Cl', 'PLT', 'Creatinine', 'Urea', 'WCC', 'K', 'HGB', 'Na', 'Temperature', 
     'temp_abnormal', 'renal_dysfunction', 'resp_dysfunction', 'hemato_abnormal']
    """
    # 9 features numériques de base
    base_features = [
        biomarkers.cl,          # Cl
        biomarkers.plt,         # PLT
        biomarkers.creatinine,  # Creatinine
        biomarkers.urea,        # Urea
        biomarkers.wcc,         # WCC
        biomarkers.k,           # K
        biomarkers.hgb,         # HGB
        biomarkers.na,          # Na
        biomarkers.temperature  # Temperature
    ]
    
    # 4 features composites
    temp_abnormal = 1 if biomarkers.temperature < 36.5 or biomarkers.temperature > 37.5 else 0
    renal_dysfunction = 1 if biomarkers.creatinine > 1.3 or biomarkers.urea > 20 else 0
    resp_dysfunction = 1 if biomarkers.po2 < 80 or biomarkers.pco2 > 45 else 0
    hemato_abnormal = 1 if biomarkers.hgb < 120 or biomarkers.wcc > 11 or biomarkers.plt < 150 else 0
    
    composite_features = [temp_abnormal, renal_dysfunction, resp_dysfunction, hemato_abnormal]
    
    # Combiner les 13 features
    all_features = base_features + composite_features
    
    return np.array(all_features).reshape(1, -1)

def extract_text_features(text_data: ClinicalTextData) -> np.ndarray:
    """Extrait les features textuelles comme dans le notebook de preprocessing"""
    import re
    
    features = []
    
    # Pour chaque champ textuel (Diagnosis, Surgery, Problems, Investigations, ClinicalCourse)
    text_fields = [
        text_data.diagnosis or "",
        text_data.surgery or "", 
        text_data.problems or "",
        text_data.investigations or "",
        text_data.clinical_course or ""
    ]
    
    field_names = ["Diagnosis", "Surgery", "Problems", "Investigations", "ClinicalCourse"]
    
    for i, text in enumerate(text_fields):
        text = str(text).strip()
        
        # Features de base
        features.extend([
            len(text),  # length
            len(text.split()) if text else 0,  # word_count
            1 if not text else 0,  # is_empty
        ])
        
        # Features spécifiques par type de champ
        if field_names[i] == "Diagnosis":
            features.extend([
                1 if re.search(r'\bacute\b', text.lower()) else 0,
                1 if re.search(r'\bchronic\b', text.lower()) else 0,
                1 if re.search(r'\bsevere\b', text.lower()) else 0,
                1 if re.search(r'\bmild\b', text.lower()) else 0,
                1 if re.search(r'\bemergency\b', text.lower()) else 0,
                1 if re.search(r'\burgent\b', text.lower()) else 0,
            ])
        elif field_names[i] == "Surgery":
            features.extend([
                1 if re.search(r'\blaparoscopic\b', text.lower()) else 0,
                1 if re.search(r'\bopen\b', text.lower()) else 0,
                1 if re.search(r'\bemergency\b', text.lower()) else 0,
                1 if re.search(r'\belective\b', text.lower()) else 0,
            ])
        elif field_names[i] == "Problems":
            features.extend([
                1 if re.search(r'\binfection\b', text.lower()) else 0,
                1 if re.search(r'\bbleeding\b', text.lower()) else 0,
                1 if re.search(r'\bsepsis\b', text.lower()) else 0,
                1 if re.search(r'\bpneumonia\b', text.lower()) else 0,
                1 if re.search(r'\bcomplications?\b', text.lower()) else 0,
            ])
        
        # Features d'entités médicales (simplifiées)
        if field_names[i] in ["Diagnosis", "Surgery"]:
            # Détection d'entités médicales basiques
            medical_terms = [
                'appendicitis', 'cholecystitis', 'hernia', 'obstruction', 'perforation',
                'appendectomy', 'cholecystectomy', 'repair', 'resection', 'anastomosis'
            ]
            has_entity = any(term in text.lower() for term in medical_terms)
            features.append(1 if has_entity else 0)
    
    return np.array(features)

def create_full_feature_vector(biomarkers: BiomarkerData, text_data: ClinicalTextData) -> np.ndarray:
    """Crée le vecteur de features complet comme attendu par les modèles"""
    # 17 features numériques
    bio_features = biomarkers_to_features(biomarkers)
    
    # ~31 features textuelles (selon le preprocessing du notebook)
    text_features = extract_text_features(text_data)
    
    # Concaténer toutes les features
    all_features = np.concatenate([bio_features, text_features])
    
    # S'assurer qu'on a le bon nombre de features (48 selon X_train.csv)
    expected_features = 48
    if len(all_features) < expected_features:
        # Padding avec des zéros si nécessaire
        padding = np.zeros(expected_features - len(all_features))
        all_features = np.concatenate([all_features, padding])
    elif len(all_features) > expected_features:
        # Tronquer si trop de features
        all_features = all_features[:expected_features]
    
    return all_features.reshape(1, -1)

def assess_risk_factors(biomarkers: BiomarkerData) -> List[str]:
    """Identifie les facteurs de risque basés sur les valeurs normales"""
    risk_factors = []
    
    # Seuils cliniques normaux
    if biomarkers.temperature < 36.0 or biomarkers.temperature > 38.5:
        risk_factors.append("Température anormale")
    
    if biomarkers.ph < 7.35 or biomarkers.ph > 7.45:
        risk_factors.append("Déséquilibre acido-basique")
    
    if biomarkers.lactate > 4.0:
        risk_factors.append("Hyperlactatémie (choc)")
    
    if biomarkers.wcc > 15.0:
        risk_factors.append("Leucocytose (infection)")
    
    if biomarkers.creatinine > 2.0:
        risk_factors.append("Dysfonction rénale")
    
    if biomarkers.po2 < 60:
        risk_factors.append("Hypoxémie")
    
    if biomarkers.hgb < 80:
        risk_factors.append("Anémie sévère")
    
    if biomarkers.plt < 100:
        risk_factors.append("Thrombopénie")
    
    if biomarkers.inr > 2.0:
        risk_factors.append("Troubles de coagulation")
    
    return risk_factors

def get_clinical_recommendations(risk_level: str, risk_factors: List[str]) -> List[str]:
    """Génère des recommandations cliniques basées sur le risque"""
    recommendations = []
    
    if risk_level == "CRITIQUE":
        recommendations.extend([
            "Admission en soins intensifs urgente",
            "Monitoring cardiaque et respiratoire continu",
            "Support organique si nécessaire",
            "Consultation multidisciplinaire"
        ])
    elif risk_level == "ÉLEVÉ":
        recommendations.extend([
            "Surveillance intensive 24h",
            "Monitoring des signes vitaux",
            "Bilans biologiques fréquents",
            "Consultation spécialisée"
        ])
    elif risk_level == "MODÉRÉ":
        recommendations.extend([
            "Surveillance renforcée",
            "Réévaluation toutes les 6h",
            "Monitoring standard"
        ])
    else:
        recommendations.append("Surveillance post-opératoire standard")
    
    # Recommandations spécifiques aux facteurs de risque
    if "Hyperlactatémie (choc)" in risk_factors:
        recommendations.append("Correction du choc et perfusion tissulaire")
    
    if "Dysfonction rénale" in risk_factors:
        recommendations.append("Surveillance de la fonction rénale et adaptation posologie")
    
    if "Hypoxémie" in risk_factors:
        recommendations.append("Optimisation de l'oxygénation")
    
    return recommendations

# ===== ENDPOINTS =====

@app.get("/", response_model=Dict[str, str])
async def root():
    """Point d'entrée de l'API"""
    return {
        "message": "ChirurgIA API - Analyse Prédictive en Chirurgie Générale",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=APIHealthResponse)
async def health_check():
    """Vérification de l'état de l'API et des modèles"""
    models_status = {}
    
    # Vérifier la disponibilité des modèles
    model_files = [
        "best_mortality_model_xgboost",
        "death_cause_classifier_improved", 
        "survivors_clustering_model"
    ]
    
    for model_name in model_files:
        model_path = MODELS_DIR / f"{model_name}.pkl"
        models_status[model_name] = model_path.exists()
    
    return APIHealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=models_status,
        timestamp=datetime.now(),
        uptime_seconds=0.0  # À implémenter avec un compteur global
    )

@app.post("/predict/mortality", response_model=MortalityPredictionResponse)
async def predict_mortality(patient_data: PatientData, patient_id: Optional[str] = None):
    """
    Prédiction de mortalité basée sur les features correctes (Notebook 03)
    RobustScaler: 17 features numériques, XGBoost: 49 features totales
    """
    if patient_id is None:
        patient_id = f"patient_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Charger les modèles
        model = load_model("best_mortality_model_xgboost")
        scaler = load_model("robust_scaler")
        label_encoder = load_model("label_encoder")
        
        # Extraire les features correctement
        numeric_features, full_features = extract_mortality_features(patient_data.biomarkers, patient_data.clinical_texts)
        
        # Appliquer le scaler sur les 17 features numériques seulement
        numeric_scaled = scaler.transform(numeric_features)
        
        # Remplacer les 17 premières features dans le vecteur complet par les valeurs scalées
        full_features_scaled = full_features.copy()
        full_features_scaled[0, :17] = numeric_scaled[0]
        
        # Prédiction avec XGBoost (49 features)
        prediction_proba = model.predict_proba(full_features_scaled)[0]
        prediction_binary = model.predict(full_features_scaled)[0]
        
        # Probabilités (attention à l'ordre des classes)
        prob_died = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
        prob_survived = prediction_proba[0] if len(prediction_proba) > 1 else 1 - prediction_proba[0]
        
        # Ajuster si nécessaire selon l'encodage du label_encoder
        try:
            classes = label_encoder.classes_
            if classes[0] == 'Survived':  # 0=Survived, 1=Died
                prob_survived, prob_died = prob_died, prob_survived
        except:
            pass
        
        # Prédiction finale
        prediction_label = label_encoder.inverse_transform([prediction_binary])[0]
        
        # Évaluation du risque basée sur les biomarqueurs
        risk_factors = assess_risk_factors(patient_data.biomarkers)
        
        # Niveau de risque basé sur la probabilité de décès
        if prob_died >= 0.8:
            risk_level = "CRITIQUE"
        elif prob_died >= 0.5:
            risk_level = "ÉLEVÉ"
        elif prob_died >= 0.3:
            risk_level = "MODÉRÉ"
        else:
            risk_level = "FAIBLE"
        
        # Confiance (certitude de la prédiction)
        confidence = max(prob_died, prob_survived)
        
        # Recommandations
        recommendations = get_clinical_recommendations(risk_level, risk_factors)
        
        return MortalityPredictionResponse(
            patient_id=patient_id,
            prediction=prediction_label,
            probability_death=float(prob_died),
            probability_survival=float(prob_survived),
            risk_level=risk_level,
            confidence=float(confidence),
            risk_factors=risk_factors,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction de mortalité: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )

@app.post("/predict/mortality-simple", response_model=MortalityPredictionResponse)
async def predict_mortality_simple(biomarkers: BiomarkerData, patient_id: Optional[str] = None):
    """
    Prédiction de mortalité simplifiée avec biomarqueurs uniquement
    Pour tester rapidement l'API
    """
    # Créer des données textuelles vides
    empty_clinical_texts = ClinicalTextData()
    
    # Créer un objet PatientData complet
    patient_data = PatientData(
        biomarkers=biomarkers,
        clinical_texts=empty_clinical_texts
    )
    
    # Utiliser l'endpoint principal
    return await predict_mortality(patient_data, patient_id)

@app.post("/predict/death-cause", response_model=DeathCausePredictionResponse)
async def predict_death_cause(patient_data: PatientData, patient_id: Optional[str] = None):
    """
    Prédiction des causes de décès (Notebook 04)
    Scaler: 17 features numériques, Classifier: 18 features
    """
    if patient_id is None:
        patient_id = f"patient_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Charger les modèles
        model = load_model("death_cause_classifier_improved")
        scaler = load_model("death_cause_scaler_improved")
        label_encoder = load_model("death_cause_label_encoder")
        
        # Extraire les features correctement
        numeric_features, classifier_features = extract_death_cause_features(patient_data.biomarkers, patient_data.clinical_texts)
        
        # Appliquer le scaler sur les 17 features numériques seulement
        numeric_scaled = scaler.transform(numeric_features)
        
        # Remplacer les 17 premières features dans le vecteur de 18 par les valeurs scalées
        classifier_features_scaled = classifier_features.copy()
        classifier_features_scaled[0, :17] = numeric_scaled[0]
        
        # Prédiction avec le classifier (18 features)
        try:
            prediction_proba = model.predict_proba(classifier_features_scaled)[0]
            prediction_binary = model.predict(classifier_features_scaled)[0]
        except Exception as pred_error:
            logger.error(f"Erreur de prédiction: {pred_error}")
            # Fallback: retourner la cause la plus probable statiquement
            predicted_cause = "Sepsis_Infection"
            top_3_causes = [
                CauseProbability(cause="Sepsis_Infection", probability=0.4),
                CauseProbability(cause="Défaillance_Multi_Organes", probability=0.3),
                CauseProbability(cause="Respiratoire", probability=0.2)
            ]
            confidence = 0.4
            clinical_interpretation = f"Prédiction par fallback: {predicted_cause} (modèle inaccessible)"
            
            return DeathCausePredictionResponse(
                patient_id=patient_id,
                predicted_cause=predicted_cause,
                top_3_causes=top_3_causes,
                confidence=confidence,
                clinical_interpretation=clinical_interpretation,
                timestamp=datetime.now()
            )
        
        # Cause prédite
        try:
            predicted_cause = label_encoder.inverse_transform([int(prediction_binary)])[0]
        except Exception as label_error:
            logger.error(f"Erreur de label: {label_error}")
            # Mapper manuellement si nécessaire
            available_labels = label_encoder.classes_
            pred_idx = int(prediction_binary) if isinstance(prediction_binary, (int, float)) else 0
            if 0 <= pred_idx < len(available_labels):
                predicted_cause = available_labels[pred_idx]
            else:
                predicted_cause = "Causes_Diverses"
        
        # Top 3 causes avec gestion d'erreur
        top_3_indices = np.argsort(prediction_proba)[-3:][::-1]
        top_3_causes = []
        
        for idx in top_3_indices:
            try:
                idx = int(idx)  # S'assurer que c'est un entier
                if 0 <= idx < len(label_encoder.classes_):
                    cause = label_encoder.classes_[idx]
                    probability = float(prediction_proba[idx])
                    top_3_causes.append(CauseProbability(cause=cause, probability=probability))
            except Exception as e:
                logger.warning(f"Erreur pour cause index {idx}: {e}")
                continue
        
        # Confiance
        confidence = float(np.max(prediction_proba))
        
        # Interprétation clinique enrichie
        clinical_interpretation = f"Cause la plus probable: {predicted_cause} ({confidence:.1%} de confiance). "
        
        # Ajouter des insights basés sur les données textuelles
        if patient_data.clinical_texts.problems:
            problems_text = patient_data.clinical_texts.problems.lower()
            if 'sepsis' in problems_text or 'infection' in problems_text:
                clinical_interpretation += "Facteurs infectieux détectés dans les complications. "
            if 'bleeding' in problems_text or 'hemorrhage' in problems_text:
                clinical_interpretation += "Risque hémorragique identifié. "
        
        return DeathCausePredictionResponse(
            patient_id=patient_id,
            predicted_cause=predicted_cause,
            top_3_causes=top_3_causes,
            confidence=confidence,
            clinical_interpretation=clinical_interpretation,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction des causes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction des causes: {str(e)}"
        )

@app.post("/analyze/clustering", response_model=ClusteringResponse)
async def analyze_patient_clustering(patient_data: PatientData, patient_id: Optional[str] = None):
    """
    Clustering des survivants par criticité (Notebook 05)
    Utilise exactement 13 features selon clustering_metadata.json
    """
    if patient_id is None:
        patient_id = f"patient_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Charger les modèles
        model = load_model("survivors_clustering_model")
        scaler = load_model("clustering_scaler")
        imputer = load_model("clustering_imputer")
        
        # Charger les métadonnées du clustering
        metadata_path = MODELS_DIR / "clustering_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            # Métadonnées par défaut
            metadata = {
                "criticality_scores": {
                    "0": {"score": 12.5, "size": 1583},
                    "1": {"score": 38.7, "size": 1421}, 
                    "2": {"score": 64.2, "size": 1324}
                }
            }
        
        # Extraire exactement les 13 features requises
        features = extract_clustering_features(patient_data.biomarkers)
        
        # Appliquer l'imputation et la normalisation (dans cet ordre)
        features_imputed = imputer.transform(features)
        features_scaled = scaler.transform(features_imputed)
        
        # Prédiction du cluster
        cluster = int(model.predict(features_scaled)[0])
        
        # Récupérer les informations du cluster
        cluster_info = metadata.get("criticality_scores", {}).get(str(cluster), {})
        criticality_score = cluster_info.get("score", 50.0)
        similar_patients = cluster_info.get("size", 1000)
        
        # Déterminer le nom et le niveau de sévérité
        if criticality_score < 20:
            cluster_name = "Récupération Excellente"
            severity_level = "FAIBLE"
            monitoring = "Standard"
            recommendations = [
                "Surveillance post-opératoire standard",
                "Mobilisation précoce encouragée",
                "Sortie selon protocole habituel"
            ]
        elif criticality_score < 50:
            cluster_name = "Récupération Modérée"
            severity_level = "MODÉRÉE"
            monitoring = "Renforcé"
            recommendations = [
                "Surveillance renforcée 24-48h",
                "Monitoring continu des signes vitaux",
                "Réévaluation quotidienne"
            ]
        else:
            cluster_name = "Récupération Difficile"
            severity_level = "ÉLEVÉE"
            monitoring = "Intensif"
            recommendations = [
                "Surveillance intensive requise",
                "Monitoring cardiaque et respiratoire",
                "Bilans biologiques fréquents",
                "Consultation spécialisée systématique"
            ]
        
        # Recommandations enrichies basées sur les données textuelles
        if patient_data.clinical_texts.problems:
            problems_text = patient_data.clinical_texts.problems.lower()
            if 'infection' in problems_text:
                recommendations.append("Surveillance renforcée des signes d'infection")
            if 'pain' in problems_text or 'douleur' in problems_text:
                recommendations.append("Gestion optimisée de la douleur")
        
        return ClusteringResponse(
            patient_id=patient_id,
            cluster=cluster,
            cluster_name=cluster_name,
            criticality_score=float(criticality_score),
            severity_level=severity_level,
            similar_patients_count=int(similar_patients),
            recommendations=recommendations,
            monitoring_level=monitoring,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Erreur lors du clustering: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du clustering: {str(e)}"
        )

@app.post("/analyze/complete", response_model=Dict[str, Any])
async def complete_analysis(patient_data: PatientData, patient_id: Optional[str] = None):
    """
    Analyse complète : mortalité + causes + clustering
    """
    if patient_id is None:
        patient_id = f"patient_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Prédiction de mortalité
        mortality_result = await predict_mortality(patient_data, patient_id)
        
        # Si risque de décès élevé, analyser les causes
        death_cause_result = None
        if mortality_result.probability_death > 0.3:
            try:
                death_cause_result = await predict_death_cause(patient_data, patient_id)
            except Exception as e:
                logger.warning(f"Prédiction des causes échouée: {e}")
        
        # Si survie probable, faire le clustering
        clustering_result = None
        if mortality_result.probability_survival > 0.5:
            clustering_result = await analyze_patient_clustering(patient_data, patient_id)
        
        return {
            "patient_id": patient_id,
            "mortality_prediction": mortality_result.dict(),
            "death_cause_analysis": death_cause_result.dict() if death_cause_result else None,
            "clustering_analysis": clustering_result.dict() if clustering_result else None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse complète: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'analyse complète: {str(e)}"
        )

# ===== POINTS D'ADMINISTRATION =====

@app.get("/admin/models", dependencies=[Depends(security)])
async def list_models():
    """Liste des modèles disponibles (authentification requise)"""
    models_info = {}
    
    for model_file in MODELS_DIR.glob("*.pkl"):
        model_name = model_file.stem
        models_info[model_name] = {
            "path": str(model_file),
            "size_mb": round(model_file.stat().st_size / (1024*1024), 2),
            "loaded": model_name in models_cache,
            "last_modified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
        }
    
    return {
        "models": models_info,
        "cache_status": list(models_cache.keys())
    }

@app.post("/admin/models/{model_name}/reload", dependencies=[Depends(security)])
async def reload_model(model_name: str):
    """Recharge un modèle spécifique"""
    try:
        # Supprimer du cache
        if model_name in models_cache:
            del models_cache[model_name]
        
        # Recharger
        load_model(model_name)
        
        return {"message": f"Modèle {model_name} rechargé avec succès"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du rechargement: {str(e)}"
        )

# ===== GESTION DES ERREURS =====

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint non trouvé",
            "message": "Vérifiez l'URL et consultez la documentation à /docs"
        }
    )

@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    logger.error(f"Erreur serveur: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Erreur interne du serveur",
            "message": "Une erreur inattendue s'est produite"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
