"""
Module pour les modèles de Machine Learning spécialisés en données médicales
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    silhouette_score, adjusted_rand_score
)
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class MortalityPredictor:
    """
    Modèle de prédiction de mortalité pour les patients chirurgicaux.
    """
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialise le prédicteur de mortalité.
        
        Args:
            model_type: Type de modèle ("random_forest", "xgboost", "lightgbm", "logistic")
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.is_fitted = False
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialise le modèle selon le type choisi."""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                scale_pos_weight=1
            )
        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == "logistic":
            self.model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        else:
            raise ValueError(f"Type de modèle non supporté : {self.model_type}")
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = "Outcome",
                    balance_data: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prépare les données pour l'entraînement.
        
        Args:
            df: DataFrame avec les données
            target_col: Nom de la colonne cible
            balance_data: Appliquer SMOTE pour équilibrer les classes
            
        Returns:
            Tuple (X, y) avec features et target
        """
        # Séparer features et target
        feature_cols = [col for col in df.columns if col != target_col and '_entities' not in col]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_col]
        
        # Encoder la target si nécessaire
        if y.dtype == 'object':
            y = (y == 'Death').astype(int)  # Assumant 'Death' comme classe positive
        
        self.feature_names = list(X.columns)
        
        # Équilibrer les données si demandé
        if balance_data:
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
            print(f"Données équilibrées avec SMOTE : {len(y)} échantillons")
        
        return X.values, y.values
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_split: float = 0.2) -> Dict[str, float]:
        """
        Entraîne le modèle de prédiction.
        
        Args:
            X: Features d'entraînement
            y: Target d'entraînement
            validation_split: Proportion pour la validation
            
        Returns:
            Métriques de performance
        """
        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, 
            random_state=42, stratify=y
        )
        
        # Entraînement
        print(f"Entraînement du modèle {self.model_type}...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Prédictions
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Calcul des métriques
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred),
            "roc_auc": roc_auc_score(y_val, y_pred_proba)
        }
        
        print("Métriques de validation :")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, List[float]]:
        """
        Effectue une validation croisée.
        
        Args:
            X: Features
            y: Target
            cv: Nombre de folds
            
        Returns:
            Scores de validation croisée
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_scores = {}
        
        for score in scoring:
            scores = cross_val_score(self.model, X, y, cv=skf, scoring=score)
            cv_scores[score] = scores.tolist()
            print(f"{score}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_scores
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Récupère l'importance des features.
        
        Returns:
            DataFrame avec l'importance des features
        """
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas encore entraîné")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            raise ValueError("Le modèle ne supporte pas l'importance des features")
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def explain_prediction(self, X_sample: np.ndarray, 
                          method: str = "shap") -> Any:
        """
        Explique une prédiction avec SHAP ou LIME.
        
        Args:
            X_sample: Échantillon à expliquer
            method: Méthode d'explication ("shap" ou "lime")
            
        Returns:
            Explication de la prédiction
        """
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas encore entraîné")
        
        if method == "shap":
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)
            return shap_values
        else:
            raise NotImplementedError(f"Méthode {method} non implémentée")
    
    def save_model(self, filepath: str):
        """Sauvegarde le modèle."""
        if not self.is_fitted:
            raise ValueError("Le modèle n'est pas encore entraîné")
        
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, filepath)
        print(f"Modèle sauvegardé : {filepath}")
    
    def load_model(self, filepath: str):
        """Charge un modèle sauvegardé."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_fitted = True
        print(f"Modèle chargé : {filepath}")


class CauseOfDeathAnalyzer:
    """
    Analyseur des causes de décès basé sur les données textuelles et numériques.
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.cause_encoder = None
        self.is_fitted = False
    
    def prepare_cause_data(self, df: pd.DataFrame, 
                          cause_col: str = "causeofDeath") -> Tuple[np.ndarray, np.ndarray]:
        """
        Prépare les données pour l'analyse des causes de décès.
        
        Args:
            df: DataFrame avec les données
            cause_col: Nom de la colonne des causes de décès
            
        Returns:
            Tuple (X, y) avec features et causes
        """
        # Filtrer uniquement les cas de décès
        death_cases = df[df['Outcome'] == 'Death'].copy()
        
        if death_cases.empty:
            raise ValueError("Aucun cas de décès trouvé dans les données")
        
        # Supprimer les cas sans cause de décès
        death_cases = death_cases.dropna(subset=[cause_col])
        
        # Préparer les features
        feature_cols = [col for col in death_cases.columns 
                       if col not in [cause_col, 'Outcome'] and '_entities' not in col]
        X = death_cases[feature_cols].select_dtypes(include=[np.number])
        
        # Encoder les causes de décès
        from sklearn.preprocessing import LabelEncoder
        self.cause_encoder = LabelEncoder()
        y = self.cause_encoder.fit_transform(death_cases[cause_col])
        
        self.feature_names = list(X.columns)
        
        return X.values, y
    
    def train_cause_classifier(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Entraîne un classificateur pour les causes de décès.
        
        Args:
            X: Features
            y: Causes de décès encodées
            
        Returns:
            Métriques de performance
        """
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Entraînement
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Évaluation
        y_pred = self.model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average='macro'),
            "f1_weighted": f1_score(y_test, y_pred, average='weighted')
        }
        
        print("Métriques du classificateur de causes :")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics


class PatientClusterer:
    """
    Clustering des patients survivants selon la criticité post-opératoire.
    """
    
    def __init__(self, n_clusters: int = 3, method: str = "kmeans"):
        """
        Initialise le clusterer.
        
        Args:
            n_clusters: Nombre de clusters souhaité
            method: Méthode de clustering ("kmeans", "dbscan")
        """
        self.n_clusters = n_clusters
        self.method = method
        self.model = None
        self.feature_names = None
        self.is_fitted = False
        
        self._initialize_clusterer()
    
    def _initialize_clusterer(self):
        """Initialise l'algorithme de clustering."""
        if self.method == "kmeans":
            self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
        elif self.method == "dbscan":
            self.model = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Méthode de clustering non supportée : {self.method}")
    
    def prepare_survivor_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prépare les données des survivants pour le clustering.
        
        Args:
            df: DataFrame avec les données
            
        Returns:
            Features des survivants
        """
        # Filtrer les survivants
        survivors = df[df['Outcome'] != 'Death'].copy()
        
        if survivors.empty:
            raise ValueError("Aucun survivant trouvé dans les données")
        
        # Sélectionner les features pertinentes
        feature_cols = [col for col in survivors.columns 
                       if '_entities' not in col and col != 'Outcome']
        X = survivors[feature_cols].select_dtypes(include=[np.number])
        
        self.feature_names = list(X.columns)
        
        return X.values
    
    def fit_cluster(self, X: np.ndarray) -> np.ndarray:
        """
        Effectue le clustering.
        
        Args:
            X: Features des survivants
            
        Returns:
            Labels des clusters
        """
        labels = self.model.fit_predict(X)
        self.is_fitted = True
        
        # Calculer les métriques
        if len(set(labels)) > 1:
            silhouette = silhouette_score(X, labels)
            print(f"Score de silhouette : {silhouette:.4f}")
        
        unique_labels = set(labels)
        print(f"Nombre de clusters trouvés : {len(unique_labels)}")
        for label in unique_labels:
            count = sum(labels == label)
            print(f"  Cluster {label}: {count} patients")
        
        return labels
    
    def analyze_clusters(self, X: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
        """
        Analyse les caractéristiques de chaque cluster.
        
        Args:
            X: Features
            labels: Labels des clusters
            
        Returns:
            DataFrame avec les statistiques par cluster
        """
        df_analysis = pd.DataFrame(X, columns=self.feature_names)
        df_analysis['cluster'] = labels
        
        # Statistiques par cluster
        cluster_stats = df_analysis.groupby('cluster').agg(['mean', 'std', 'count'])
        
        return cluster_stats
    
    def plot_clusters(self, X: np.ndarray, labels: np.ndarray, 
                     features_to_plot: List[str] = None):
        """
        Visualise les clusters.
        
        Args:
            X: Features
            labels: Labels des clusters
            features_to_plot: Features à utiliser pour la visualisation
        """
        if features_to_plot is None:
            # Prendre les 2 premières features
            features_to_plot = self.feature_names[:2]
        
        if len(features_to_plot) < 2:
            print("Pas assez de features pour la visualisation 2D")
            return
        
        feature_indices = [self.feature_names.index(f) for f in features_to_plot]
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            X[:, feature_indices[0]], 
            X[:, feature_indices[1]], 
            c=labels, 
            cmap='viridis',
            alpha=0.7
        )
        plt.xlabel(features_to_plot[0])
        plt.ylabel(features_to_plot[1])
        plt.title('Clustering des Patients Survivants')
        plt.colorbar(scatter)
        plt.show()


def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, 
                             y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Évalue les performances d'un modèle de classification.
    
    Args:
        y_true: Vraies valeurs
        y_pred: Prédictions
        y_pred_proba: Probabilités prédites
        
    Returns:
        Dictionnaire avec les métriques
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1": f1_score(y_true, y_pred, average='weighted')
    }
    
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str] = None):
    """
    Affiche la matrice de confusion.
    
    Args:
        y_true: Vraies valeurs
        y_pred: Prédictions
        class_names: Noms des classes
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de Confusion')
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies Valeurs')
    plt.show()
