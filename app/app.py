"""
Application Streamlit pour le déploiement du modèle ChirurgIA
Application simple pour prédire la mortalité et analyser les données chirurgicales
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configuration de la page
st.set_page_config(
    page_title="ChirurgIA - Analyse Prédictive Chirurgicale",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin: 1rem 0;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.warning-box {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 0.3rem;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def main():
    # En-tête principal
    st.markdown('<h1 class="main-header">🏥 ChirurgIA</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Système d\'Analyse Prédictive pour la Chirurgie Générale</p>', unsafe_allow_html=True)
    
    # Sidebar pour la navigation
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Choisir une section :",
        ["🏠 Accueil", "📊 Analyse des Données", "🔮 Prédiction de Mortalité", 
         "🧬 Analyse des Causes", "👥 Clustering des Patients", "📋 À Propos"]
    )
    
    if page == "🏠 Accueil":
        show_home()
    elif page == "📊 Analyse des Données":
        show_data_analysis()
    elif page == "🔮 Prédiction de Mortalité":
        show_mortality_prediction()
    elif page == "🧬 Analyse des Causes":
        show_cause_analysis()
    elif page == "👥 Clustering des Patients":
        show_patient_clustering()
    elif page == "📋 À Propos":
        show_about()

def show_home():
    """Page d'accueil avec présentation du projet"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">🎯 Objectifs du Projet</h2>', unsafe_allow_html=True)
        
        objectives = [
            ("🔍 Détermination des causes de décès", "Analyse des patterns dans les données de mortalité"),
            ("📈 Prédiction de mortalité", "Modèles ML pour prédire les risques post-opératoires"),
            ("👥 Clustering des survivants", "Segmentation selon la criticité post-chirurgicale")
        ]
        
        for title, desc in objectives:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3 class="sub-header">📊 Statistiques Générales</h3>', unsafe_allow_html=True)
        
        # Métriques factices pour la démonstration
        st.metric("Patients analysés", "1,000", "📈 +50")
        st.metric("Taux de survie", "85%", "📈 +2%")
        st.metric("Précision du modèle", "92%", "📈 +5%")
        
        # Graphique de démonstration
        demo_data = pd.DataFrame({
            'Mois': ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun'],
            'Patients': [150, 180, 220, 190, 240, 280]
        })
        
        fig = px.line(demo_data, x='Mois', y='Patients', 
                     title="Évolution du nombre de patients")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Section d'avertissement
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Avertissement Médical</h4>
        <p>Cette application est destinée à des fins de recherche et d'éducation uniquement. 
        Les prédictions ne doivent jamais remplacer l'avis médical professionnel. 
        Consultez toujours un professionnel de santé qualifié pour toute décision médicale.</p>
    </div>
    """, unsafe_allow_html=True)

def show_data_analysis():
    """Page d'analyse des données"""
    st.markdown('<h2 class="sub-header">📊 Analyse des Données</h2>', unsafe_allow_html=True)
    
    # Option de chargement des données
    st.markdown("### 📁 Chargement des Données")
    
    uploaded_file = st.file_uploader(
        "Choisir un fichier CSV", 
        type=['csv'],
        help="Téléchargez votre fichier de données chirurgicales"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        # Aperçu des données
        st.markdown("### 👀 Aperçu des Données")
        st.dataframe(df.head(), use_container_width=True)
        
        # Statistiques descriptives
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Statistiques Numériques")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe())
            else:
                st.info("Aucune colonne numérique trouvée")
        
        with col2:
            st.markdown("#### ❌ Valeurs Manquantes")
            missing_data = df.isnull().sum()
            missing_df = pd.DataFrame({
                'Colonne': missing_data.index,
                'Valeurs manquantes': missing_data.values,
                'Pourcentage': (missing_data.values / len(df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Valeurs manquantes'] > 0]
            
            if not missing_df.empty:
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("Aucune valeur manquante !")
        
        # Visualisations
        if 'Outcome' in df.columns:
            st.markdown("### 📊 Distribution des Outcomes")
            outcome_counts = df['Outcome'].value_counts()
            
            fig = px.pie(
                values=outcome_counts.values,
                names=outcome_counts.index,
                title="Répartition Survivants vs Décès"
            )
            st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Données de démonstration
        st.info("📝 Téléchargez un fichier CSV ou utilisez les données de démonstration ci-dessous")
        
        if st.button("🔄 Générer des données de démonstration"):
            demo_df = generate_demo_data()
            st.session_state['demo_data'] = demo_df
            st.success("✅ Données de démonstration générées !")
            st.dataframe(demo_df.head(), use_container_width=True)

def generate_demo_data():
    """Génère des données de démonstration"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'Age': np.random.choice(['18-30', '31-50', '51-70', '71+'], n_samples),
        'Temperature': np.random.normal(37.0, 1.5, n_samples),
        'pH': np.random.normal(7.4, 0.1, n_samples),
        'Lactate': np.random.exponential(2, n_samples),
        'Surgery': np.random.choice(['Appendectomy', 'Cholecystectomy', 'Hernia repair'], n_samples),
        'Outcome': np.random.choice(['Alive', 'Death'], n_samples, p=[0.85, 0.15])
    }
    
    return pd.DataFrame(data)

def show_mortality_prediction():
    """Page de prédiction de mortalité"""
    st.markdown('<h2 class="sub-header">🔮 Prédiction de Mortalité</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Cette section permet de prédire le risque de mortalité post-opératoire 
    basé sur les signes vitaux et les caractéristiques du patient.
    """)
    
    # Formulaire de saisie
    st.markdown("### 📝 Saisie des Données Patient")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Données Démographiques")
        age_group = st.selectbox("Groupe d'âge", ['18-30', '31-50', '51-70', '71+'])
        race = st.selectbox("Origine", ['White', 'Black', 'Hispanic', 'Asian', 'Other'])
    
    with col2:
        st.markdown("#### 🩺 Signes Vitaux")
        temperature = st.slider("Température (°C)", 35.0, 42.0, 37.0, 0.1)
        ph = st.slider("pH", 7.0, 7.8, 7.4, 0.01)
        lactate = st.slider("Lactate (mmol/L)", 0.5, 10.0, 2.0, 0.1)
    
    with col3:
        st.markdown("#### 🔬 Paramètres Biologiques")
        wcc = st.slider("GB (10³/μL)", 2.0, 20.0, 8.0, 0.5)
        creatinine = st.slider("Créatinine (mg/dL)", 0.5, 5.0, 1.0, 0.1)
        hgb = st.slider("Hémoglobine (g/dL)", 80, 180, 130, 5)
    
    # Prédiction (simulée)
    if st.button("🔍 Prédire le Risque", type="primary"):
        # Simulation d'une prédiction
        risk_factors = []
        risk_score = 0
        
        if temperature > 38.5 or temperature < 36.0:
            risk_factors.append("Température anormale")
            risk_score += 20
        
        if ph < 7.35 or ph > 7.45:
            risk_factors.append("pH déséquilibré")
            risk_score += 15
        
        if lactate > 4.0:
            risk_factors.append("Lactate élevé")
            risk_score += 25
        
        if wcc > 15.0:
            risk_factors.append("Leucocytose")
            risk_score += 10
        
        if creatinine > 2.0:
            risk_factors.append("Dysfonction rénale")
            risk_score += 20
        
        # Affichage des résultats
        st.markdown("### 📊 Résultats de la Prédiction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score de risque
            if risk_score < 20:
                risk_level = "🟢 FAIBLE"
                risk_color = "success"
            elif risk_score < 50:
                risk_level = "🟡 MODÉRÉ"
                risk_color = "warning"
            else:
                risk_level = "🔴 ÉLEVÉ"
                risk_color = "error"
            
            st.metric("Score de Risque", f"{risk_score}/100")
            st.markdown(f"**Niveau de Risque : {risk_level}**")
        
        with col2:
            if risk_factors:
                st.markdown("**⚠️ Facteurs de Risque Identifiés :**")
                for factor in risk_factors:
                    st.markdown(f"• {factor}")
            else:
                st.success("✅ Aucun facteur de risque majeur détecté")
        
        # Graphique de risque
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Score de Risque"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgreen"},
                    {'range': [20, 50], 'color': "yellow"},
                    {'range': [50, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_cause_analysis():
    """Page d'analyse des causes de décès"""
    st.markdown('<h2 class="sub-header">🧬 Analyse des Causes de Décès</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Cette section analyse les causes de décès les plus fréquentes 
    et les facteurs associés.
    """)
    
    # Données de démonstration pour les causes
    causes_data = {
        'Cause': ['Septic shock', 'Cardiac arrest', 'Respiratory failure', 
                 'Multiple organ failure', 'Surgical complications', 'Pneumonia'],
        'Nombre': [45, 32, 28, 25, 18, 12],
        'Pourcentage': [28.1, 20.0, 17.5, 15.6, 11.3, 7.5]
    }
    
    causes_df = pd.DataFrame(causes_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Distribution des Causes")
        fig = px.bar(causes_df, x='Cause', y='Nombre', 
                    title="Nombre de Décès par Cause")
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🥧 Répartition Proportionnelle")
        fig = px.pie(causes_df, values='Nombre', names='Cause',
                    title="Répartition des Causes de Décès")
        st.plotly_chart(fig, use_container_width=True)
    
    # Tableau détaillé
    st.markdown("### 📋 Tableau Détaillé")
    st.dataframe(causes_df, use_container_width=True)
    
    # Analyse temporelle (simulation)
    st.markdown("### 📈 Évolution Temporelle")
    months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun']
    
    temporal_data = pd.DataFrame({
        'Mois': months * 3,
        'Cause': ['Septic shock'] * 6 + ['Cardiac arrest'] * 6 + ['Respiratory failure'] * 6,
        'Cas': [8, 12, 10, 15, 8, 11, 5, 8, 6, 10, 7, 9, 4, 7, 5, 8, 6, 7]
    })
    
    fig = px.line(temporal_data, x='Mois', y='Cas', color='Cause',
                 title="Évolution des Principales Causes de Décès")
    st.plotly_chart(fig, use_container_width=True)

def show_patient_clustering():
    """Page de clustering des patients"""
    st.markdown('<h2 class="sub-header">👥 Clustering des Patients Survivants</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Cette section groupe les patients survivants selon leur criticité 
    post-opératoire et leurs caractéristiques cliniques.
    """)
    
    # Simulation de données de clustering
    np.random.seed(42)
    n_patients = 200
    
    # Génération de 3 clusters
    cluster_data = []
    
    # Cluster 1: Récupération excellente
    for i in range(70):
        cluster_data.append({
            'Patient_ID': f'P{i+1}',
            'Temperature': np.random.normal(36.8, 0.3),
            'Lactate': np.random.normal(1.5, 0.3),
            'Length_of_Stay': np.random.normal(3, 1),
            'Complications': np.random.choice([0, 1], p=[0.9, 0.1]),
            'Cluster': 'Récupération Excellente'
        })
    
    # Cluster 2: Récupération modérée
    for i in range(70, 140):
        cluster_data.append({
            'Patient_ID': f'P{i+1}',
            'Temperature': np.random.normal(37.5, 0.5),
            'Lactate': np.random.normal(2.5, 0.5),
            'Length_of_Stay': np.random.normal(7, 2),
            'Complications': np.random.choice([0, 1], p=[0.7, 0.3]),
            'Cluster': 'Récupération Modérée'
        })
    
    # Cluster 3: Récupération difficile
    for i in range(140, 200):
        cluster_data.append({
            'Patient_ID': f'P{i+1}',
            'Temperature': np.random.normal(38.2, 0.7),
            'Lactate': np.random.normal(4.0, 1.0),
            'Length_of_Stay': np.random.normal(15, 5),
            'Complications': np.random.choice([0, 1], p=[0.4, 0.6]),
            'Cluster': 'Récupération Difficile'
        })
    
    cluster_df = pd.DataFrame(cluster_data)
    
    # Visualisations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Distribution des Clusters")
        cluster_counts = cluster_df['Cluster'].value_counts()
        fig = px.pie(values=cluster_counts.values, names=cluster_counts.index,
                    title="Répartition des Patients par Cluster")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Caractéristiques par Cluster")
        fig = px.box(cluster_df, x='Cluster', y='Length_of_Stay',
                    title="Durée de Séjour par Cluster")
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot 2D
    st.markdown("### 🔍 Visualisation 2D des Clusters")
    fig = px.scatter(cluster_df, x='Temperature', y='Lactate', 
                    color='Cluster', size='Length_of_Stay',
                    title="Clustering basé sur Température et Lactate",
                    hover_data=['Patient_ID', 'Complications'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques par cluster
    st.markdown("### 📈 Statistiques par Cluster")
    cluster_stats = cluster_df.groupby('Cluster').agg({
        'Temperature': ['mean', 'std'],
        'Lactate': ['mean', 'std'],
        'Length_of_Stay': ['mean', 'std'],
        'Complications': 'mean'
    }).round(2)
    
    st.dataframe(cluster_stats, use_container_width=True)

def show_about():
    """Page à propos"""
    st.markdown('<h2 class="sub-header">📋 À Propos de ChirurgIA</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎓 Projet Académique
    
    **ChirurgIA** est un projet de Machine Learning développé dans le cadre d'un cours universitaire 
    sur l'analyse de données médicales en chirurgie générale.
    
    ### 🎯 Objectifs Pédagogiques
    
    - **Analyse de données médicales réelles** : Comprendre les défis spécifiques du domaine médical
    - **Application du Machine Learning** : Mise en pratique d'algorithmes sur des données sensibles
    - **Développement d'applications** : Création d'interfaces utilisateur pour le déploiement
    - **Interprétabilité des modèles** : Importance de l'explicabilité en contexte médical
    
    ### 🛠️ Technologies Utilisées
    
    - **Python** : Langage principal de développement
    - **Scikit-learn** : Algorithmes de Machine Learning
    - **ScispaCy** : Traitement du langage naturel médical
    - **Streamlit** : Interface web interactive
    - **Plotly** : Visualisations interactives
    - **Pandas/NumPy** : Manipulation et analyse des données
    
    ### 📚 Méthodologie
    
    1. **Exploration des données** : Analyse descriptive et visualisation
    2. **Préprocessing** : Nettoyage et préparation des données médicales
    3. **Feature Engineering** : Extraction d'entités médicales avec NLP
    4. **Modélisation** : 
       - Classification pour la prédiction de mortalité
       - Analyse textuelle pour les causes de décès
       - Clustering pour la segmentation des patients
    5. **Évaluation** : Métriques adaptées au domaine médical
    6. **Déploiement** : Application web pour la démonstration
    
    ### ⚠️ Limitations et Avertissements
    
    - **Usage académique uniquement** : Ce projet est à des fins éducatives
    - **Données simulées** : Les données utilisées sont fictives pour la démonstration
    - **Pas d'usage clinique** : Ne pas utiliser pour de vraies décisions médicales
    - **Validation requise** : Tout modèle médical nécessite une validation clinique rigoureuse
    
    ### 👥 Équipe de Développement
    
    Projet réalisé par une équipe de 4 étudiants en Data Science.
    
    ### 📞 Contact
    
    Pour toute question sur ce projet académique, veuillez contacter votre encadrant pédagogique.
    """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>© 2024 ChirurgIA - Projet Académique de Machine Learning</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
