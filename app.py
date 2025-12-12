"""
Application Streamlit - AISCA Skills Assessment
Questionnaire mixte : Likert, texte libre, QCM, cases à cocher
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict
import json

from genai.hybrid_generator import HybridGenerator
from rag.job_recommender import JobRecommender
from nlp.scoring_blocks import BlockScorer

# Configuration de la page
st.set_page_config(
    page_title="AISCA - Skills Assessment",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Questions mixtes
QUESTIONS_CONFIG = [
    {
        "id": 1,
        "type": "likert",
        "question": "Évaluez votre niveau en Python",
        "options": ["1 - Débutant", "2 - Élémentaire", "3 - Intermédiaire", "4 - Avancé", "5 - Expert"],
        "context": "Python programming for data science and analysis"
    },
    {
        "id": 2,
        "type": "text",
        "question": "Décrivez votre expérience avec l'analyse de données (outils, projets, durée)",
        "placeholder": "Ex: J'ai 2 ans d'expérience en analyse de données avec pandas, numpy...",
        "context": "data analysis experience"
    },
    {
        "id": 3,
        "type": "checkbox",
        "question": "Quels outils de visualisation maîtrisez-vous ?",
        "options": ["Matplotlib", "Seaborn", "Plotly", "Tableau", "Power BI", "D3.js", "Aucun"],
        "context": "data visualization tools"
    },
    {
        "id": 4,
        "type": "likert",
        "question": "Évaluez votre niveau en Machine Learning",
        "options": ["1 - Débutant", "2 - Élémentaire", "3 - Intermédiaire", "4 - Avancé", "5 - Expert"],
        "context": "machine learning and predictive modeling"
    },
    {
        "id": 5,
        "type": "guided",
        "question": "Avez-vous déjà utilisé des techniques de tokenization en NLP ?",
        "options": ["Oui, régulièrement", "Oui, occasionnellement", "Non, jamais", "Je ne sais pas ce que c'est"],
        "followup": "Si oui, décrivez brièvement votre expérience",
        "context": "natural language processing and tokenization"
    },
    {
        "id": 6,
        "type": "multiple_choice",
        "question": "Quelle base de données utilisez-vous principalement ?",
        "options": ["MySQL/PostgreSQL", "MongoDB", "Oracle", "SQL Server", "BigQuery", "Aucune"],
        "context": "database management and SQL"
    },
    {
        "id": 7,
        "type": "checkbox",
        "question": "Sélectionnez vos compétences en Machine Learning",
        "options": [
            "Régression linéaire/logistique",
            "Arbres de décision",
            "Random Forest",
            "XGBoost/LightGBM",
            "Neural Networks",
            "Deep Learning",
            "Aucune"
        ],
        "context": "machine learning algorithms and techniques"
    },
    {
        "id": 8,
        "type": "likert",
        "question": "Évaluez votre niveau en statistiques",
        "options": ["1 - Débutant", "2 - Élémentaire", "3 - Intermédiaire", "4 - Avancé", "5 - Expert"],
        "context": "statistics and hypothesis testing"
    },
    {
        "id": 9,
        "type": "text",
        "question": "Décrivez un projet data science dont vous êtes fier",
        "placeholder": "Ex: J'ai développé un modèle de prédiction de churn avec 85% de précision...",
        "context": "data science project experience"
    },
    {
        "id": 10,
        "type": "checkbox",
        "question": "Quels services cloud avez-vous utilisés ?",
        "options": ["AWS", "Azure", "Google Cloud Platform", "IBM Cloud", "Aucun"],
        "context": "cloud computing platforms"
    }
]


def initialize_session_state():
    """Initialise l'état de la session."""
    if 'responses' not in st.session_state:
        st.session_state.responses = {}
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'detailed_analysis' not in st.session_state:
        st.session_state.detailed_analysis = None
    if 'block_scores' not in st.session_state:
        st.session_state.block_scores = None
    if 'job_recommendations' not in st.session_state:
        st.session_state.job_recommendations = None


def convert_responses_to_text(responses: Dict) -> List[str]:
    """Convertit les réponses structurées en texte pour SBERT."""
    texts = []
    
    for q_id, response in responses.items():
        q_config = next((q for q in QUESTIONS_CONFIG if q['id'] == q_id), None)
        if not q_config:
            continue
        
        context = q_config['context']
        
        if q_config['type'] == 'likert':
            level = response.split('-')[0].strip()
            text = f"{context}: level {level} out of 5, {response}"
            texts.append(text)
        
        elif q_config['type'] == 'text':
            if response and response.strip():
                text = f"{context}: {response}"
                texts.append(text)
        
        elif q_config['type'] == 'checkbox':
            if isinstance(response, list):
                selected = [opt for opt in response if opt != "Aucun"]
                if selected:
                    text = f"{context}: experience with {', '.join(selected)}"
                    texts.append(text)
        
        elif q_config['type'] == 'multiple_choice':
            if response != "Aucune" and response != "Aucun":
                text = f"{context}: proficient in {response}"
                texts.append(text)
        
        elif q_config['type'] == 'guided':
            if isinstance(response, dict):
                answer = response.get('main', '')
                followup = response.get('followup', '')
                if answer == "Oui, régulièrement" or answer == "Oui, occasionnellement":
                    text = f"{context}: {answer}"
                    if followup:
                        text += f". {followup}"
                    texts.append(text)
    
    return texts


def render_header():
    """Affiche l'en-tête."""
    st.markdown('<h1 class="main-header">🎓 AISCA Skills Assessment</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; font-size: 1.2rem; color: #666;'>
    Évaluez vos compétences en Data Science & IA
    </p>
    """, unsafe_allow_html=True)
    st.markdown("---")


def render_question(q_config):
    """Affiche une question selon son type."""
    q_id = q_config['id']
    q_type = q_config['type']
    question = q_config['question']
    
    st.markdown(f"### Question {q_id}/10")
    st.write(f"**{question}**")
    
    if q_type == 'likert':
        response = st.radio(
            label=f"q_{q_id}",
            options=q_config['options'],
            key=f"q_{q_id}",
            label_visibility="collapsed"
        )
        st.session_state.responses[q_id] = response
    
    elif q_type == 'text':
        response = st.text_area(
            label=f"q_{q_id}",
            placeholder=q_config.get('placeholder', ''),
            height=100,
            key=f"q_{q_id}",
            label_visibility="collapsed"
        )
        st.session_state.responses[q_id] = response
    
    elif q_type == 'checkbox':
        st.write("*Sélectionnez toutes les options qui s'appliquent*")
        selected = []
        for option in q_config['options']:
            if st.checkbox(option, key=f"q_{q_id}_{option}"):
                selected.append(option)
        st.session_state.responses[q_id] = selected
    
    elif q_type == 'multiple_choice':
        response = st.radio(
            label=f"q_{q_id}",
            options=q_config['options'],
            key=f"q_{q_id}",
            label_visibility="collapsed"
        )
        st.session_state.responses[q_id] = response
    
    elif q_type == 'guided':
        main_answer = st.radio(
            label=f"q_{q_id}_main",
            options=q_config['options'],
            key=f"q_{q_id}_main",
            label_visibility="collapsed"
        )
        
        followup_answer = ""
        if main_answer in ["Oui, régulièrement", "Oui, occasionnellement"]:
            st.write(f"*{q_config.get('followup', '')}*")
            followup_answer = st.text_input(
                label=f"q_{q_id}_followup",
                key=f"q_{q_id}_followup",
                label_visibility="collapsed"
            )
        
        st.session_state.responses[q_id] = {
            'main': main_answer,
            'followup': followup_answer
        }


def render_questionnaire():
    """Affiche le questionnaire complet."""
    st.markdown("## 📝 Questionnaire de Compétences")
    
    st.info("💡 Répondez à toutes les questions pour obtenir une évaluation précise.")
    
    for q_config in QUESTIONS_CONFIG:
        render_question(q_config)
        st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Analyser mes compétences", key="analyze_btn", use_container_width=True, type="primary"):
            success = process_responses()
            if success:
                st.session_state.analysis_complete = True
                st.rerun()


def process_responses():
    """Traite les réponses avec gestion d'erreurs détaillée."""
    user_texts = convert_responses_to_text(st.session_state.responses)
    
    if not user_texts:
        st.error("❌ Aucune réponse détectée. Veuillez répondre à au moins une question.")
        return False
    
    with st.spinner("🔄 Analyse en cours..."):
        try:
            # Initialiser les composants
            st.session_state.generator = HybridGenerator(use_cache=True, prefer_local=True)
            st.session_state.job_recommender = JobRecommender()
            st.session_state.block_scorer = BlockScorer()
            
            st.session_state.user_texts = user_texts
            
            # Calcul des analyses
            st.session_state.block_scores = st.session_state.block_scorer.calculate_block_scores(user_texts)
            st.session_state.job_recommendations = st.session_state.job_recommender.get_top_recommendations(user_texts, n=3)
            st.session_state.detailed_analysis = st.session_state.block_scorer.get_detailed_analysis(user_texts)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'analyse: {e}")
            import traceback
            with st.expander("🔍 Détails de l'erreur"):
                st.code(traceback.format_exc())
            return False


def render_results():
    """Affiche les résultats."""
    if not st.session_state.detailed_analysis:
        st.error("Erreur : Analyse non disponible")
        if st.button("← Retour au questionnaire"):
            st.session_state.analysis_complete = False
            st.rerun()
        return
    
    st.markdown("## 📊 Résultats de votre Analyse")
    
    coverage_score = st.session_state.detailed_analysis['coverage_score']
    interpretation = st.session_state.detailed_analysis['interpretation']
    
    st.markdown(f"""
    <div class="score-card">
        <h2>Score Global de Couverture</h2>
        <h1 style="font-size: 4rem; margin: 1rem 0;">{coverage_score:.1%}</h1>
        <p style="font-size: 1.3rem;">{interpretation}</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Scores par Bloc",
        "💼 Métiers Recommandés", 
        "🎯 Plan de Progression",
        "👤 Bio Professionnelle"
    ])
    
    with tab1:
        render_block_scores()
    
    with tab2:
        render_job_recommendations()
    
    with tab3:
        render_progression_plan()
    
    with tab4:
        render_professional_bio()


def render_block_scores():
    """Affiche les scores par bloc."""
    if not st.session_state.block_scores:
        st.warning("Scores non disponibles")
        return
    
    st.markdown("### 📊 Analyse par Bloc de Compétences")
    
    block_data = []
    for block_name, data in st.session_state.block_scores.items():
        block_data.append({
            'Bloc': block_name,
            'Score': data['score'] * 100
        })
    
    df = pd.DataFrame(block_data).sort_values('Score', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df['Score'],
        y=df['Bloc'],
        orientation='h',
        marker=dict(color=df['Score'], colorscale='RdYlGn', showscale=True),
        text=df['Score'].round(1),
        texttemplate='%{text}%',
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Scores par Bloc de Compétences",
        xaxis_title="Score (%)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_job_recommendations():
    """Affiche les métiers recommandés."""
    if not st.session_state.job_recommendations:
        st.warning("Recommandations non disponibles")
        return
    
    st.markdown("### 💼 Top 3 Métiers Recommandés")
    
    for i, job in enumerate(st.session_state.job_recommendations, 1):
        st.markdown(f"#### {i}. {job['title']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="📊 Score de Correspondance",
                value=f"{job['score']:.1%}"
            )
        
        with col2:
            st.metric(
                label="✅ Compétences Validées",
                value=f"{job['num_matched_competencies']}/{job['num_required_competencies']}"
            )
        
        st.info(f"**Niveau de préparation:** {job['readiness']}")
        
        if job.get('matched_competencies'):
            with st.expander("✅ Vos compétences validées"):
                for comp in job['matched_competencies'][:5]:
                    st.write(f"- {comp}")
        
        if job.get('missing_competencies'):
            with st.expander("📚 Compétences à développer"):
                for comp in job['missing_competencies'][:5]:
                    st.write(f"- {comp}")
        
        st.markdown("---")


def render_progression_plan():
    """Génère le plan de progression."""
    st.markdown("### 🎯 Plan de Progression Personnalisé")
    
    if st.button("✨ Générer mon plan avec l'IA", key="gen_plan_btn"):
        with st.spinner("Génération en cours..."):
            try:
                plan = st.session_state.generator.generate_progression_plan(st.session_state.user_texts)
                st.session_state.progression_plan = plan
            except Exception as e:
                st.error(f"Erreur: {e}")
    
    if 'progression_plan' in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state.progression_plan)
        st.download_button(
            label="📥 Télécharger le plan",
            data=st.session_state.progression_plan,
            file_name="plan_progression.txt",
            mime="text/plain",
            key="download_plan_btn"
        )


def render_professional_bio():
    """Génère la bio."""
    st.markdown("### 👤 Bio Professionnelle")
    
    if st.button("✨ Générer ma bio avec l'IA", key="gen_bio_btn"):
        with st.spinner("Génération en cours..."):
            try:
                jobs = [j['title'] for j in st.session_state.job_recommendations]
                bio = st.session_state.generator.generate_professional_bio(st.session_state.user_texts, jobs)
                st.session_state.professional_bio = bio
            except Exception as e:
                st.error(f"Erreur: {e}")
    
    if 'professional_bio' in st.session_state:
        st.markdown("---")
        st.info(st.session_state.professional_bio)
        st.download_button(
            label="📥 Télécharger la bio",
            data=st.session_state.professional_bio,
            file_name="bio_professionnelle.txt",
            mime="text/plain",
            key="download_bio_btn"
        )


def render_sidebar():
    """Sidebar."""
    with st.sidebar:
        st.markdown("### 🎓 AISCA")
        st.markdown("**Skills Assessment System**")
        
        st.markdown("---")
        st.markdown("### 📚 À propos")
        st.info("""
        Système d'évaluation utilisant :
        
        - 🧠 SBERT (analyse sémantique)
        - 🎯 RAG (recommandations)
        - ✨ IA générative
        """)
        
        if st.session_state.analysis_complete:
            st.success("✅ Analyse terminée")
            
            st.markdown("---")
            if st.button("🔄 Nouvelle Analyse", key="reset_btn", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()


def main():
    """Main."""
    initialize_session_state()
    render_header()
    render_sidebar()
    
    if not st.session_state.analysis_complete:
        render_questionnaire()
    else:
        render_results()


if __name__ == "__main__":
    main()
