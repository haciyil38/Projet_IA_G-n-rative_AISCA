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
git clone <votre-repo>
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

### Architecture du système

┌─────────────────┐
│ Utilisateur │
│ (Questionnaire)│
└────────┬────────┘
│
▼
┌─────────────────┐
│ SBERT Local │
│ (Embeddings) │
└────────┬────────┘
│
▼
┌─────────────────┐
│ RAG System │
│ (Retrieval) │
└────────┬────────┘
│
▼
┌─────────────────┐
│ Hybrid GenAI │
│ Ollama/Gemini │
└────────┬────────┘
│
▼
┌─────────────────┐
│ Résultats │
│ Plans + Bios │
└─────────────────┘

text

## 📁 Structure du projet

iagen/
├── app.py # Application Streamlit principale
├── config.py # Configuration centralisée
├── embeddings.py # Gestion embeddings SBERT
├── encode_repository.py # Pré-calcul embeddings
├── requirements.txt # Dépendances Python
├── .env.example # Template variables d'environnement
├── data/
│ ├── repository.json # Référentiel compétences
│ └── repo_embeddings.npz # Embeddings pré-calculés (généré)
├── genai/
│ ├── client.py # Client Gemini
│ ├── ollama_client.py # Client Ollama (local)
│ ├── hybrid_generator.py # Générateur hybride
│ ├── cache_manager.py # Système de cache
│ └── generator.py # Générateur original
├── nlp/
│ ├── scoring.py # Calcul similarité
│ └── scoring_blocks.py # Scoring par blocs
├── rag/
│ ├── retriever.py # Récupération compétences
│ ├── context_builder.py # Construction contexte
│ └── job_recommender.py # Recommandation métiers
└── tests/
├── test_embeddings.py
├── test_scoring.py
└── test_rag.py

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
   - Nécessite clé API (300$ crédits gratuits)
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

Projet académique - EFREI Paris 2024-2025

## 📝 Licence

Projet éducatif - Tous droits réservés

## 👥 Auteur

Haci Yilmazer - EFREI Paris

## 📞 Support

Pour toute question sur le projet, consultez la documentation dans `/docs`
