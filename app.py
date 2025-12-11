"""
AISCA - Application Streamlit principale
"""
import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="AISCA - Cartographie des Compétences",
    page_icon="🎯",
    layout="wide"
)

# Page d'accueil
st.title("🎯 AISCA")
st.subheader("Agent Intelligent Sémantique et Génératif pour la Cartographie des Compétences")

st.markdown("""
### Bienvenue !

Cette application vous aide à :
- 📊 Évaluer vos compétences
- 🎯 Obtenir des recommandations de métiers
- 📈 Identifier les compétences à développer

---
**Status**: 🚧 En développement
""")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choisir une page",
        ["Accueil", "Questionnaire", "Résultats"]
    )

if page == "Accueil":
    st.info("👈 Utilisez le menu latéral pour naviguer")
elif page == "Questionnaire":
    st.warning("⚠️ Module en développement")
elif page == "Résultats":
    st.warning("⚠️ Module en développement")
