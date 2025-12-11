# 🎓 AISCA - AI Skills & Career Assessment

Système intelligent d'évaluation de compétences et recommandation de carrière utilisant l'IA générative et l'analyse sémantique.

## 📌 Description

Application d'analyse sémantique des compétences combinant :
- **SBERT** (Sentence-BERT) pour l'analyse sémantique locale
- **RAG** (Retrieval-Augmented Generation) pour les recommandations
- **IA Générative** (Ollama/Gemini) pour la génération de contenu personnalisé
- **Streamlit** pour l'interface utilisateur interactive

## 🚀 Installation

### Prérequis
- Python 3.9+
- pip ou conda

### Étapes

1. **Cloner le repository**
git clone https://github.com/haciyil38/Projet_IA_G-n-rative_AISCA
cd iagen

text

2. **Créer l'environnement virtuel**
python -m venv venv
source venv/bin/activate # Sur macOS/Linux

ou
venv\Scripts\activate # Sur Windows

text

3. **Installer les dépendances**
pip install -r requirements.txt

text

4. **Configurer les variables d'environnement**
cp .env.example .env

Éditer .env avec votre clé API Gemini (optionnel)
text

5. **Encoder le référentiel de compétences**
python encode_repository.py

text

6. **Installer Ollama (optionnel, pour LLM local)**
macOS
brew install ollama

Démarrer Ollama
ollama serve

Télécharger un modèle
ollama pull llama3.2

text

## 🎯 Utilisation

### Lancer l'application Streamlit
streamlit run app.py

text

## 📁 Structure du projet

iagen/
├── app.py # Application Streamlit principale
├── config.py # Configuration centralisée
├── embeddings.py # Gestion embeddings SBERT
├── encode_repository.py # Pré-calcul des embeddings
├── requirements.txt # Dépendances Python
├── .env.example # Template variables d'environnement
│
├── data/
│ ├── repository.json # Référentiel des compétences
│ └── repo_embeddings.npz # Embeddings pré-calculés (généré)
│
├── genai/ # Module IA Générative
│ ├── init.py
│ ├── client.py # Client Google Gemini
│ ├── ollama_client.py # Client Ollama (LLM local)
│ ├── hybrid_generator.py # Générateur hybride multi-provider
│ ├── cache_manager.py # Système de cache intelligent
│ └── generator.py # Générateur Gemini original
│
├── nlp/ # Module NLP & Scoring
│ ├── init.py
│ ├── scoring.py # Calcul de similarité sémantique
│ └── scoring_blocks.py # Scoring par blocs de compétences
│
├── rag/ # Module RAG (Retrieval-Augmented Generation)
│ ├── init.py
│ ├── retriever.py # Récupération des compétences pertinentes
│ ├── context_builder.py # Construction du contexte enrichi
│ └── job_recommender.py # Recommandation de métiers
│
└── tests/ # Tests unitaires
├── init.py
├── test_embeddings.py # Tests des embeddings SBERT
├── test_scoring.py # Tests du système de scoring
├── test_rag.py # Tests du système RAG
└── test_genai.py # Tests de l'IA générative

text

## 🧪 Tests

Tous les tests
pytest tests/ -v

Tests spécifiques
pytest tests/test_embeddings.py -v
pytest tests/test_rag.py -v

text

## ⚙️ Configuration

### Providers IA disponibles

1. **Ollama (Local - Recommandé pour développement)**
   - Gratuit et illimité
   - Fonctionne hors ligne
   - Installation : `brew install ollama`

2. **Google Gemini (Cloud - Recommandé pour production)**
   - Nécessite clé API 
   - Configuration dans `.env`
   - Obtenir clé : https://ai.google.dev/

### Système de fallback

Le système hybride bascule automatiquement :
1. **Ollama** (priorité) → local, rapide
2. **Gemini** (backup) → cloud, performant
3. **Templates** (fallback) → toujours fonctionnel

## 📊 Fonctionnalités

✅ **Questionnaire interactif** (10 questions)  
✅ **Analyse sémantique SBERT** (matching compétences)  
✅ **Scoring par blocs** de compétences  
✅ **Top 3 métiers recommandés** avec scores  
✅ **Plan de progression** personnalisé  
✅ **Bio professionnelle** générée par IA  
✅ **Visualisations interactives** (Plotly)  
✅ **Système de cache** (optimisation coûts)  

## 🎓 Exigences du projet

Ce projet répond aux exigences suivantes :

- **EF1** : Architecture RAG complète
- **EF2** : Embeddings SBERT pour analyse sémantique
- **EF3** : Scoring et recommandations top 3
- **EF4** : IA générative (enrichissement + génération)
- **EF5** : Interface Streamlit interactive

## 🤝 Contribution

Projet académique - EFREI Paris 2025-2026

## 📝 Licence

Projet éducatif dans le but de notre mastère

## 👥 Auteur

Haci
Neïl

