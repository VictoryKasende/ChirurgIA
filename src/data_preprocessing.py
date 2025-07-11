"""
Module de préparation et nettoyage des données médicales
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
import scispacy
import spacy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer


class MedicalDataPreprocessor:
    """
    Classe pour le préprocessing des données médicales chirurgicales.
    
    Gère :
    - Nettoyage des données numériques (signes vitaux)
    - Traitement des textes médicaux avec ScispaCy
    - Gestion des valeurs manquantes
    - Encodage des variables catégorielles
    """
    
    def __init__(self, scispacy_model: str = "en_core_sci_sm"):
        """
        Initialise le preprocesseur.
        
        Args:
            scispacy_model: Nom du modèle ScispaCy à utiliser
        """
        self.scispacy_model = scispacy_model
        self.nlp = None
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
        # Attributs numériques (signes vitaux)
        self.numeric_features = [
            'Temperature', 'pH', 'pCO2', 'pO2', 'HCO3', 'BE', 'Lactate',
            'Na', 'K', 'Cl', 'Urea', 'Creatinine', 'HGT', 'WCC', 'HGB', 
            'PLT', 'INR', 'ABG', 'U+E', 'FBC'
        ]
        
        # Attributs textuels
        self.text_features = ['Diagnosis', 'Surgery', 'Problems', 'Investigation']
        
        # Attributs catégoriels
        self.categorical_features = ['Age', 'Race', 'Outcome']
    
    def load_scispacy_model(self):
        """Charge le modèle ScispaCy pour le traitement des textes médicaux."""
        try:
            self.nlp = spacy.load(self.scispacy_model)
            print(f"Modèle ScispaCy '{self.scispacy_model}' chargé avec succès.")
        except OSError:
            print(f"Erreur : Modèle '{self.scispacy_model}' non trouvé.")
            print("Installation requise : pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gz")
            raise
    
    def clean_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données numériques (signes vitaux).
        
        Args:
            df: DataFrame contenant les données
            
        Returns:
            DataFrame avec données numériques nettoyées
        """
        df_clean = df.copy()
        
        for feature in self.numeric_features:
            if feature in df_clean.columns:
                # Convertir en numérique, forcer les erreurs en NaN
                df_clean[feature] = pd.to_numeric(df_clean[feature], errors='coerce')
                
                # Supprimer les valeurs aberrantes (méthode IQR)
                Q1 = df_clean[feature].quantile(0.25)
                Q3 = df_clean[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_clean[feature] = df_clean[feature].clip(lower_bound, upper_bound)
        
        return df_clean
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extrait les entités médicales d'un texte avec ScispaCy.
        
        Args:
            text: Texte médical à analyser
            
        Returns:
            Dictionnaire avec les entités extraites par type
        """
        if not self.nlp:
            self.load_scispacy_model()
        
        if pd.isna(text) or text == "":
            return {"entities": [], "diseases": [], "procedures": []}
        
        doc = self.nlp(str(text))
        
        entities = {
            "entities": [],
            "diseases": [],
            "procedures": []
        }
        
        for ent in doc.ents:
            entities["entities"].append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
            
            # Catégoriser selon le type d'entité
            if ent.label_ in ["DISEASE", "SYMPTOM"]:
                entities["diseases"].append(ent.text)
            elif ent.label_ in ["PROCEDURE", "TREATMENT"]:
                entities["procedures"].append(ent.text)
        
        return entities
    
    def process_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Traite les caractéristiques textuelles avec ScispaCy.
        
        Args:
            df: DataFrame contenant les données
            
        Returns:
            DataFrame avec nouvelles features extraites du texte
        """
        df_processed = df.copy()
        
        for feature in self.text_features:
            if feature in df.columns:
                # Extraire les entités médicales
                entities_list = df[feature].apply(self.extract_medical_entities)
                
                # Créer de nouvelles features
                df_processed[f"{feature}_num_entities"] = entities_list.apply(
                    lambda x: len(x["entities"])
                )
                df_processed[f"{feature}_num_diseases"] = entities_list.apply(
                    lambda x: len(x["diseases"])
                )
                df_processed[f"{feature}_num_procedures"] = entities_list.apply(
                    lambda x: len(x["procedures"])
                )
                
                # Stocker les entités textuelles pour analyse ultérieure
                df_processed[f"{feature}_entities"] = entities_list
        
        return df_processed
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "knn") -> pd.DataFrame:
        """
        Gère les valeurs manquantes selon la stratégie choisie.
        
        Args:
            df: DataFrame à traiter
            strategy: Stratégie d'imputation ("mean", "median", "knn")
            
        Returns:
            DataFrame avec valeurs manquantes imputées
        """
        df_imputed = df.copy()
        
        # Imputation pour les features numériques
        numeric_cols = [col for col in self.numeric_features if col in df.columns]
        
        if numeric_cols:
            if strategy == "knn":
                imputer = KNNImputer(n_neighbors=5)
            else:
                imputer = SimpleImputer(strategy=strategy)
            
            df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
            self.imputers['numeric'] = imputer
        
        # Imputation pour les features catégorielles
        categorical_cols = [col for col in self.categorical_features if col in df.columns]
        
        for col in categorical_cols:
            mode_value = df_imputed[col].mode()[0] if not df_imputed[col].mode().empty else "Unknown"
            df_imputed[col].fillna(mode_value, inplace=True)
        
        return df_imputed
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode les features catégorielles.
        
        Args:
            df: DataFrame à encoder
            
        Returns:
            DataFrame avec features catégorielles encodées
        """
        df_encoded = df.copy()
        
        for feature in self.categorical_features:
            if feature in df.columns:
                encoder = LabelEncoder()
                df_encoded[f"{feature}_encoded"] = encoder.fit_transform(
                    df_encoded[feature].astype(str)
                )
                self.encoders[feature] = encoder
        
        return df_encoded
    
    def normalize_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise les features numériques.
        
        Args:
            df: DataFrame à normaliser
            
        Returns:
            DataFrame avec features numériques normalisées
        """
        df_normalized = df.copy()
        
        numeric_cols = [col for col in self.numeric_features if col in df.columns]
        
        if numeric_cols:
            scaler = StandardScaler()
            df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])
            self.scalers['numeric'] = scaler
        
        return df_normalized
    
    def preprocess_pipeline(self, df: pd.DataFrame, 
                          normalize: bool = True,
                          handle_missing: bool = True,
                          process_text: bool = True) -> pd.DataFrame:
        """
        Pipeline complet de préprocessing.
        
        Args:
            df: DataFrame à traiter
            normalize: Normaliser les features numériques
            handle_missing: Gérer les valeurs manquantes
            process_text: Traiter les features textuelles
            
        Returns:
            DataFrame complètement préprocessé
        """
        print("Début du préprocessing...")
        
        # 1. Nettoyage des données numériques
        df_processed = self.clean_numeric_data(df)
        print("✓ Nettoyage des données numériques terminé")
        
        # 2. Traitement des textes médicaux
        if process_text:
            df_processed = self.process_text_features(df_processed)
            print("✓ Traitement des textes médicaux terminé")
        
        # 3. Gestion des valeurs manquantes
        if handle_missing:
            df_processed = self.handle_missing_values(df_processed)
            print("✓ Gestion des valeurs manquantes terminée")
        
        # 4. Encodage des variables catégorielles
        df_processed = self.encode_categorical_features(df_processed)
        print("✓ Encodage des variables catégorielles terminé")
        
        # 5. Normalisation des features numériques
        if normalize:
            df_processed = self.normalize_numeric_features(df_processed)
            print("✓ Normalisation des features numériques terminée")
        
        print("Préprocessing terminé avec succès!")
        return df_processed


def load_data(file_path: str) -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV.
    
    Args:
        file_path: Chemin vers le fichier de données
        
    Returns:
        DataFrame avec les données chargées
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return df
    except FileNotFoundError:
        print(f"Erreur : Fichier '{file_path}' non trouvé.")
        raise
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        raise


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Génère un résumé des données.
    
    Args:
        df: DataFrame à analyser
        
    Returns:
        Dictionnaire avec le résumé des données
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.to_dict(),
        "numeric_summary": df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else {},
        "categorical_summary": {
            col: df[col].value_counts().head().to_dict() 
            for col in df.select_dtypes(include=['object']).columns
        }
    }
    
    return summary
