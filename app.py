import streamlit as st
import pandas as pd
import joblib
import os

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Maladie Cardiaque",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Fonction de nettoyage (nécessaire pour le pipeline)
def clean_famhist(X):
    """Nettoie la colonne famhist"""
    X = X.copy()
    X["famhist"] = (
        X["famhist"]
        .str.strip()
        .str.lower()
        .replace({"present": "present", "absent": "absent"})
    )
    return X

# Chargement du modèle avec gestion d'erreur
@st.cache_resource
def load_model():
    """Charge le modèle ML sauvegardé"""
    try:
        model_path = "Model3.pkl"
        if not os.path.exists(model_path):
            st.error(f"❌ Fichier {model_path} introuvable!")
            return None
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# ============= INTERFACE PRINCIPALE =============

st.title("🫀 Prédiction du Risque de Maladie Cardiaque")
st.markdown("""
Cette application utilise un modèle de **Machine Learning** (KNN optimisé avec ACP) 
pour prédire le risque de maladie cardiaque coronarienne (CHD).

---
""")

# Chargement du modèle
model = load_model()

if model is None:
    st.stop()



# ============= FORMULAIRE DE SAISIE =============

st.subheader(" Saisir les données du patient")

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        sbp = st.number_input(
            "Pression Systolique (SBP)",
            min_value=80,
            max_value=250,
            value=138,
            help="Valeur normale : 90-140 mmHg"
        )
        
        ldl = st.number_input(
            "Cholestérol LDL",
            min_value=0.0,
            max_value=1600.0,
            value=440.0,
            step=10.0,
            help="Valeur normale : < 100 mg/dL"
        )
        
        adiposity = st.number_input(
            "Adiposité",
            min_value=10,
            max_value=4500,
            value=2326,
            help="Indice de masse grasse"
        )
    
    with col2:
        obesity = st.number_input(
            "Obésité",
            min_value=10.0,
            max_value=5000.0,
            value=2373.0,
            step=10.0,
            help="Indice d'obésité corporelle"
        )
        
        age = st.number_input(
            "Âge",
            min_value=15,
            max_value=100,
            value=43,
            help="Âge du patient en années"
        )
        
        famhist = st.selectbox(
            "Antécédents Familiaux",
            options=["Present", "Absent"],
            help="Y a-t-il des antécédents de maladie cardiaque dans la famille ?"
        )
    
    submitted = st.form_submit_button("🔍 Prédire le Risque", use_container_width=True)

# ============= PRÉDICTION =============

if submitted:
    # Création du DataFrame d'entrée
    input_data = pd.DataFrame({
        "sbp": [sbp],
        "ldl": [ldl],
        "adiposity": [adiposity],
        "famhist": [famhist.lower()],
        "obesity": [obesity],
        "age": [age]
    })
    
 
    
    # Prédiction
    try:
        with st.spinner("Analyse en cours..."):
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]
            proba_chd = proba[1]  # Probabilité de CHD = 1
        
        # Affichage des résultats
        st.markdown("---")
        st.subheader(" Résultats de l'Analyse")
        
        # Création de 3 colonnes pour les métriques
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Prédiction", 
                     "RISQUE ÉLEVÉ" if prediction == 1 else "RISQUE FAIBLE",
                     delta=None)
        
        with col2:
            st.metric("Probabilité CHD", 
                     f"{proba_chd:.1%}",
                     delta=None)
        
        with col3:
            st.metric("Confiance", 
                     f"{max(proba):.1%}",
                     delta=None)
        
        # Message d'alerte coloré
        if prediction == 1:
            st.error(f"""
            ⚠️ **RISQUE ÉLEVÉ DÉTECTÉ**
            
            Le modèle prédit un risque **élevé** de maladie cardiaque coronarienne.
            
            Probabilité : **{proba_chd:.1%}**
            
            💡 **Recommandations :**
            - Consulter rapidement un cardiologue
            - Surveiller la pression artérielle
            - Adopter un mode de vie sain
            """)
        else:
            st.success(f"""
            ✅ **RISQUE FAIBLE**
            
            Le modèle prédit un risque **faible** de maladie cardiaque coronarienne.
            
            Probabilité de CHD : **{proba_chd:.1%}**
            
            💡 **Recommandations :**
            - Maintenir un mode de vie sain
            - Contrôles réguliers recommandés
            - Continuer la prévention
            """)
        
        # Barre de progression visuelle
        st.markdown("### 📊 Niveau de risque")
        st.progress(proba_chd)
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")

# ============= FOOTER =============
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    Développé avec ❤️ | Modèle basé sur le dataset CHD.csv
    <br>
    <em>Cette application est fournie à des fins éducatives uniquement et ne remplace pas un avis médical professionnel.</em>
</div>
""", unsafe_allow_html=True)