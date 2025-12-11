# AISCA - AI Skills & Career Assessment

Système intelligent d'évaluation de compétences et recommandation de carrière utilisant l'IA générative et l'analyse sémantique.

## Description

Application d'analyse sémantique des compétences combinant :
- **SBERT** (Sentence-BERT) pour l'analyse sémantique locale
- **RAG** (Retrieval-Augmented Generation) pour les recommandations
- **IA Générative** (Ollama/Gemini) pour la génération de contenu personnalisé
- **Streamlit** pour l'interface utilisateur interactive

## Installation

### Prérequis
- Python 3.9+
- pip ou conda

### Étapes

**1. Cloner le repository**
```
git clone <votre-repo>
cd iagen
```

**2. Créer l'environnement virtuel**
```
python -m venv venv
source venv/bin/activate  # Sur macOS/Linux
# ou
venv\Scripts\activate  # Sur Windows
```

**3. Installer les dépendances**
```
pip install -r requirements.txt
```

**4. Configurer les variables d'environnement**
```
cp .env.example .env
# Éditer .env avec votre clé API Gemini (optionnel)
```

**5. Encoder le référentiel de compétences**
```
python encode_repository.py
```

**6. Installer Ollama (optionnel, pour LLM local)**
```
# macOS
brew install ollama

# Démarrer Ollama
ollama serve

# Télécharger un modèle
ollama pull llama3.2
```

## Utilisation

### Lancer l'application Streamlit
```
streamlit run app.py
```

### Architecture du système

```
┌─────────────────┐
│   Utilisateur   │
│  (Questionnaire)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   SBERT Local   │
│   (Embeddings)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   RAG System    │
│   (Retrieval)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hybrid GenAI   │
│  Ollama/Gemini  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Résultats    │
│  Plans + Bios   │
└─────────────────┘
```

## Structure du projet

```
iagen/
│
├── app.py                          # Application Streamlit principale
├── config.py                       # Configuration centralisée
├── embeddings.py                   # Gestion embeddings SBERT
├── encode_repository.py            # Pré-calcul des embeddings
├── requirements.txt                # Dépendances Python
├── .env.example                    # Template variables d'environnement
│
├── data/
│   ├── repository.json             # Référentiel des compétences
│   └── repo_embeddings.npz         # Embeddings pré-calculés
│
├── genai/
│   ├── __init__.py
│   ├── client.py                   # Client Google Gemini
│   ├── ollama_client.py            # Client Ollama (LLM local)
│   ├── hybrid_generator.py         # Générateur hybride
│   ├── cache_manager.py            # Système de cache
│   └── generator.py                # Générateur Gemini original
│
├── nlp/
│   ├── __init__.py
│   ├── scoring.py                  # Calcul de similarité
│   └── scoring_blocks.py           # Scoring par blocs
│
├── rag/
│   ├── __init__.py
│   ├── retriever.py                # Récupération compétences
│   ├── context_builder.py          # Construction contexte
│   └── job_recommender.py          # Recommandation métiers
│
└── tests/
    ├── __init__.py
    ├── test_embeddings.py          # Tests SBERT
    ├── test_scoring.py             # Tests scoring
    ├── test_rag.py                 # Tests RAG
    └── test_genai.py               # Tests IA générative
```

### Description des modules

#### Module racine
- **app.py** : Interface Streamlit avec questionnaire interactif
- **config.py** : Configuration centralisée (API keys, chemins, paramètres)
- **embeddings.py** : Gestionnaire SBERT pour encodage sémantique
- **encode_repository.py** : Script de pré-calcul des embeddings

#### data/
Données et embeddings
- **repository.json** : Référentiel complet des compétences AISCA
- **repo_embeddings.npz** : Embeddings pré-calculés (optimisation)

#### genai/
IA Générative
- **client.py** : Client API Google Gemini
- **ollama_client.py** : Client Ollama pour LLM local
- **hybrid_generator.py** : Système hybride avec fallback automatique
- **cache_manager.py** : Cache intelligent pour optimiser les coûts

#### nlp/
Analyse sémantique
- **scoring.py** : Calcul de similarité cosinus avec SBERT
- **scoring_blocks.py** : Scoring pondéré par blocs (Σ(Wi×Si) / ΣWi)

#### rag/
Recommandations
- **retriever.py** : Extraction des compétences pertinentes
- **context_builder.py** : Construction du contexte enrichi
- **job_recommender.py** : Top 3 métiers recommandés avec scores

#### tests/
Tests unitaires
- Tests complets de tous les modules avec pytest

## Tests

```
# Tous les tests
pytest tests/ -v

# Tests spécifiques
pytest tests/test_embeddings.py -v
pytest tests/test_rag.py -v
```

## Configuration

### Providers IA disponibles

**1. Ollama (Local - Recommandé pour développement)**
- Gratuit et illimité
- Fonctionne hors ligne
- Installation : `brew install ollama`

**2. Google Gemini (Cloud - Recommandé pour production)**
- Nécessite clé API
- Configuration dans `.env`
- Obtenir clé : https://ai.google.dev/

### Système de fallback intelligent

Le système hybride bascule automatiquement :
1. **Ollama** (priorité) - local, rapide
2. **Gemini** (backup) - cloud, performant
3. **Templates** (fallback) - toujours fonctionnel

## Fonctionnalités

| Fonctionnalité | Status |
|----------------|--------|
| Questionnaire interactif (10 questions) | Implémenté |
| Analyse sémantique SBERT | Implémenté |
| Scoring par blocs de compétences | Implémenté |
| Top 3 métiers recommandés | Implémenté |
| Plan de progression personnalisé | Implémenté |
| Bio professionnelle générée par IA | Implémenté |
| Visualisations interactives (Plotly) | Implémenté |
| Système de cache (optimisation coûts) | Implémenté |

## Exigences du projet

Ce projet répond aux exigences suivantes :

| Exigence | Description | Status |
|----------|-------------|--------|
| **EF1** | Architecture RAG complète | Implémenté |
| **EF2** | Embeddings SBERT pour analyse sémantique | Implémenté |
| **EF3** | Scoring et recommandations top 3 | Implémenté |
| **EF4** | IA générative (enrichissement + génération) | Implémenté |
| **EF5** | Interface Streamlit interactive | Implémenté |

## Technologies utilisées

- **Backend** : Python 3.9+
- **Embeddings** : SBERT (sentence-transformers)
- **IA Générative** : Ollama (Llama 3.2) + Google Gemini
- **Frontend** : Streamlit
- **Visualisations** : Plotly
- **Tests** : pytest
- **Cache** : Système JSON local

## Déploiement

### Local
```
streamlit run app.py
```

### Streamlit Cloud
1. Push le code sur GitHub
2. Connectez-vous sur https://streamlit.io/cloud
3. Déployez depuis votre repository

## Contribution

Projet académique - EFREI Paris 2024-2025

**Équipe :**
- Haci
- Neïl

## Licence

Projet éducatif dans le but de notre Mastère - Tous droits réservés



